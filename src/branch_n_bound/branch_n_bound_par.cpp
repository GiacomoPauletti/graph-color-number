#include "branch_n_bound_par.hpp"

#define ALLGATHER_WAIT_TIME 6  // Sleep time for MPI_Allgather
#define TIMEOUT_CHECK_WAIT_TIME 5  // Sleep time for timeout checker

// tags for MPI communication
#define TAG_WORK_REQUEST 1
#define TAG_WORK_RESPONSE 2
#define TAG_SOLUTION_FOUND 4
#define TAG_IDLE 5
#define TAG_WORK_STEALING 6

std::atomic<bool> terminate_flag = false;
std::mutex queue_mutex;	 // avoid concurrent access to the queue
std::mutex log_mutex;

std::mutex branching_mutex;
std::mutex task_mutex;

std::mutex cout_mutex;

/**
 * @brief Checks if the timeout has been reached based on the start time and the
 * given timeout duration.
 *
 * @param start_time The start time point from which to calculate the elapsed
 * time.
 * @param timeout_seconds The timeout duration in seconds.
 * @return true if the elapsed time is greater than or equal to the timeout,
 * false otherwise.
 */
bool BranchNBoundPar::CheckTimeout(
    const std::chrono::steady_clock::time_point& start_time,
    int timeout_seconds) {
	// TODO: check if its better using only mpi_wtime instead of chrono
	auto current_time = MPI_Wtime();
	double start_time_seconds =
	    std::chrono::duration<double>(start_time.time_since_epoch())
		.count();
	auto elapsed_seconds = current_time - start_time_seconds;
	return elapsed_seconds >= timeout_seconds;
}


void BranchNBoundPar::Log_par(const std::string& message, int depth) {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (_log_file.is_open()) {
        // Get the current MPI walltime
        double timestamp = MPI_Wtime();

        // Indentation based on depth
        std::string indentation(depth * 2, ' ');

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int thread_id = omp_get_thread_num();

        // Log the message with the timestamp
        _log_file << indentation << "[Rank " << rank
                      << " | Thread " << thread_id << "] "
                      << "[Time " << timestamp << "] " << message << std::endl;
    }
}

void printMessage(const std::string& msg) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << msg << std::endl;
}

/**
 * @brief Sends a serialized Branch object to a specified destination in the MPI
 * communication.
 *
 * This function serializes the Branch object and sends it over MPI to the
 * specified destination with the provided tag and communicator.
 *
 * @param b The Branch object to be sent.
 * @param dest The destination rank for the message.
 * @param tag The MPI tag for the message.
 * @param comm The MPI communicator used for communication.
 */
void sendBranch(const Branch& b, int dest, int tag, MPI_Comm comm) {
	std::vector<char> buffer = b.serialize();
	int size = buffer.size();
	MPI_Request request[2];	
	int completed = 0;

	MPI_Isend(&size, 1, MPI_INT, dest, tag, comm, &request[0]);
    
	MPI_Isend(buffer.data(), size, MPI_BYTE, dest, tag, comm, &request[1]);

	while (!terminate_flag.load(std::memory_order_relaxed)) {
        MPI_Testall(2, request, &completed, MPI_STATUSES_IGNORE);
        if (completed) return;
    }

	MPI_Cancel(&request[0]);
    MPI_Cancel(&request[1]);
    MPI_Request_free(&request[0]);
    MPI_Request_free(&request[1]);
}

/**
 * @brief Receives a serialized Branch object from a specified source in the MPI
 * communication.
 *
 * This function receives the serialized Branch object from the specified source
 * and reconstructs the Branch object by deserializing the received data.
 *
 * @param source The source rank from which to receive the message.
 * @param tag The MPI tag for the message.
 * @param comm The MPI communicator used for communication.
 * @return The deserialized Branch object.
 */
Branch recvBranch(int source, int tag, MPI_Comm comm) {
	MPI_Status status[2];
    MPI_Request request[2];
    int size = 0;
    int flag = 0;

	MPI_Irecv(&size, 1, MPI_INT, source, tag, comm, &request[0]);

	while (!terminate_flag.load(std::memory_order_relaxed)) {
        MPI_Test(&request[0], &flag, &status[0]);
        if (flag) break;
	}

	if (!flag) {
        MPI_Cancel(&request[0]);
        MPI_Request_free(&request[0]);
        return Branch();
    }
	
	std::vector<char> buffer(size, 0);

	MPI_Irecv(buffer.data(), size, MPI_BYTE, source, tag, comm, &request[1]);
    flag = 0;

	while (!terminate_flag.load(std::memory_order_relaxed)) {
        MPI_Test(&request[1], &flag, &status[1]);
        if (flag) break;
    }

	if (!flag) {
        MPI_Cancel(&request[1]);
        MPI_Request_free(&request[1]);
        return Branch();
    }

	return Branch::deserialize(buffer);
}

/**
 * thread_0_terminator - Listens for termination signals (solution found or
 * timeout) and terminates the execution of the process accordingly. This
 * function is called by the second thread of the master and the worker
 * processes.
 *
 * This function is responsible for detecting whether a solution has been found
 * or a timeout has occurred. If the master (rank 0) receives a solution, it
 * broadcasts the solution status to all processes. If the timeout occurs, it
 * broadcasts the timeout signal. Worker nodes listen for these termination
 * signals and exit the loop if any of the conditions are met.
 *
 * Parameters:
 *   my_rank (int)           : The rank of the current process.
 *   p (int)                 : The total number of processes in the MPI
 * communicator. global_start_time (int) : The global start time, used to check
 * for timeouts. timeout_seconds (int)   : The timeout duration (in seconds)
 * after which the timeout signal is sent.
 */
void BranchNBoundPar::thread_0_terminator(int my_rank, int p, int global_start_time, int timeout_seconds, double &optimum_time) {
	int solution_found = 0;
	int timeout_signal = 0;

	std::vector<int> idle_status(p, 1); // Array to keep track of idle status of workers
	idle_status[0] = 0; // Root process is not idle
	while (true) {
		if (my_rank == 0) {
			// Master listens for solution found (Non-blocking)
			MPI_Status status_solution;
			MPI_Status status_idle;
			int flag_solution = 0;
			int flag_idle = 0;

			// Check if timeout is reached, broadcast timeout signal
			if (MPI_Wtime() - global_start_time >= timeout_seconds) {
				timeout_signal = 1;
				Log_par("[TERMINATION]: Timeout reached.", 0);
			}

			MPI_Iprobe(MPI_ANY_SOURCE, TAG_SOLUTION_FOUND, MPI_COMM_WORLD, &flag_solution, &status_solution);
			// Check if a solution is being communicated
			if (flag_solution) {
				unsigned short solution = 0;
				MPI_Request recv_request;
				MPI_Irecv(&solution, 1, MPI_UNSIGNED_SHORT, status_solution.MPI_SOURCE, TAG_SOLUTION_FOUND, MPI_COMM_WORLD, &recv_request);

				int completed = 0;
				MPI_Status status_sol_completed;
				while (!completed) {
					if(terminate_flag.load()) break;
					MPI_Test(&recv_request, &completed, &status_sol_completed);
					usleep(1000);
				}

				solution_found = 1;
				Log_par("[TERMINATION]: Solution found communicated.", 0);
				optimum_time = MPI_Wtime() - global_start_time;
			}

			// Listen for idle status updates from workers
			while (true) {
				int flag_idle = 0;
				MPI_Iprobe(MPI_ANY_SOURCE, TAG_IDLE, MPI_COMM_WORLD, &flag_idle, &status_idle);
			
				if (!flag_idle) break;

				//std::cout << "Master received idle status" << std::endl;
				//printMessage("Master received idle status from " + std::to_string(status_idle.MPI_SOURCE));
				int worker_idle_status = 0;


				MPI_Request recv_request;
				MPI_Irecv(&worker_idle_status, 1, MPI_INT, status_idle.MPI_SOURCE, TAG_IDLE, MPI_COMM_WORLD, &recv_request);

				int completed = 0;
				MPI_Status status_sol_completed;
				while (!completed) {
					if(terminate_flag.load()) break;
					MPI_Test(&recv_request, &completed, &status_sol_completed);
					usleep(1000);
				}
				
				if(completed) idle_status[status_idle.MPI_SOURCE] = worker_idle_status;
			}

            // Check if all workers are idle
            if (std::all_of(idle_status.begin(), idle_status.end(), [](int status) { return status == 1; })) {
                solution_found = 1;
				optimum_time = MPI_Wtime() - global_start_time;
				Log_par("[TERMINATION]: All processes idle.", 0);
            }

		}
		// Worker nodes listen for termination signals (solution or timeout)
		MPI_Bcast(&solution_found, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&timeout_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// Exit if solution or timeout is detected
		if (solution_found || timeout_signal) {
			terminate_flag.store(true, std::memory_order_relaxed);
			break;
		}
		usleep(10000);	// Prevent CPU overload (10 ms)
	}
}

/**
 * thread_1_solution_gatherer - Periodically gathers the best upper bound
 * (best_ub) from all worker processes and updates the global best_ub. This
 * function is called by the first thread of the master and the worker
 * processes.
 *
 * This function is run by the master process and performs an allgather
 * operation to collect the best upper bound from all worker nodes every
 * ALLGATHER_WAIT_TIME seconds. The best upper bound is updated by taking the
 * minimum of the current best_ub and the gathered values.
 *
 * Parameters:
 *   p (int)           : The number of processes in the MPI communicator.
 *   best_ub (int*)    : Pointer to the variable holding the best upper bound.
 */
void BranchNBoundPar::thread_1_solution_gatherer(int p, std::atomic<unsigned short>& best_ub) { 
    std::vector<unsigned short> all_best_ub(p);
    auto last_gather_time = MPI_Wtime();
    MPI_Request request;
	int request_active = 0;
	
    while (!terminate_flag.load(std::memory_order_relaxed)) {
        auto current_time = MPI_Wtime();
        auto elapsed_time = current_time - last_gather_time;

        if (elapsed_time >= ALLGATHER_WAIT_TIME) {
            unsigned short local_best_ub = best_ub.load(); // safe read

			if (terminate_flag.load(std::memory_order_relaxed)) {
                return;
            }

            // Start non-blocking allgather
            MPI_Iallgather(&local_best_ub, 1, MPI_UNSIGNED_SHORT, all_best_ub.data(), 1, MPI_UNSIGNED_SHORT, MPI_COMM_WORLD, &request);
			request_active = 1;

            // Wait for completion with timeout handling (or simply test it periodically)
            MPI_Status status;
            while (true) {
                int flag = 0;
                MPI_Test(&request, &flag, &status);
                if (flag) break;  // The operation is completed

                // If termination flag is set, cancel the request to avoid deadlock
                if (terminate_flag.load(std::memory_order_relaxed)) {
					if (request_active && flag) {
                        MPI_Cancel(&request);
                        MPI_Request_free(&request);
                    }
                    return;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
			request_active = 0;

            // Update the best upper bound for other threads in shared memory
			Log_par("[UPDATE] Gathered best_ub " + std::to_string(best_ub), 0);
            best_ub.store(*std::min_element(all_best_ub.begin(), all_best_ub.end()));  // safe write

            // Reset the timer
            last_gather_time = current_time;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
	if (request_active) {
        MPI_Cancel(&request);
        MPI_Request_free(&request);
    }
}


/**
 * thread_2_employer - Listens for work requests from other worker processes and
 * distributes available work from the local queue. This function is called by
 * the third thread of the worker processes.
 *
 * This function is responsible for handling work requests from other worker
 * processes. It listens for incoming work requests using non-blocking MPI
 * communication. If a work request is received and there is work available in
 * the local queue, it sends a branch to the requesting process. If no work is
 * available, it sends a response indicating that no work is available.
 *
 * Parameters:
 *   queue_mutex (std::mutex&) : Mutex to protect concurrent access to the work
 * queue.
 *   queue (BranchQueue&)      : The local work queue containing branches to be
 * processed.
 */
void BranchNBoundPar::thread_2_employer(std::mutex& queue_mutex, BranchQueue& queue) {
    MPI_Status status;
    int request_signal = 0;
    MPI_Request request;

    while (!terminate_flag.load(std::memory_order_relaxed)) {
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_WORK_REQUEST, MPI_COMM_WORLD, &request_signal, &status);
        if (request_signal) {
            int destination_rank = status.MPI_SOURCE;
            int response = 0;

            std::lock_guard<std::mutex> lock(queue_mutex);
            if (queue.size() > 1) {
                response = 1;
                Branch branch = std::move(const_cast<Branch&>(queue.top()));
                queue.pop();

                MPI_Isend(&response, 1, MPI_INT, destination_rank, TAG_WORK_RESPONSE, MPI_COMM_WORLD, &request);
                MPI_Request_free(&request);
                sendBranch(branch, destination_rank, TAG_WORK_STEALING, MPI_COMM_WORLD);
            } else {
                MPI_Isend(&response, 1, MPI_INT, destination_rank, TAG_WORK_RESPONSE, MPI_COMM_WORLD, &request);
                MPI_Request_free(&request);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

/**
 * request_work - Requests work from other worker processes when the local queue is empty.
 *
 * This function is responsible for requesting work from other worker processes
 * when the local queue is empty. It uses non-blocking MPI communication to send
 * a work request to a randomly selected worker and waits for a response. If work
 * is available, it receives the branch and adds it to the local queue.
 *
 * Parameters:
 *   my_rank (int)            : The rank of the current process.
 *   p (int)                  : The total number of processes in the MPI communicator.
 *   queue (BranchQueue&)     : The local work queue containing branches to be processed.
 *   queue_mutex (std::mutex&): Mutex to protect concurrent access to the work queue.
 *   current (Branch&)        : The branch object to store the received work.
 *
 * Returns:
 *   bool : True if work was successfully received and added to the queue, false otherwise.
 */
bool request_work(int my_rank, int p, BranchQueue& queue, std::mutex& queue_mutex, Branch& current) {
    int target_worker = my_rank;
    while (target_worker == my_rank) target_worker = (rand() % p); // Randomly select a worker to request work from

    MPI_Status status;
    int response = 0;
    MPI_Request send_request, recv_request;

    //printMessage("Rank: " + std::to_string(my_rank) + " Sending work request to " + std::to_string(target_worker));
    MPI_Isend(nullptr, 0, MPI_INT, target_worker, TAG_WORK_REQUEST, MPI_COMM_WORLD, &send_request);
    MPI_Request_free(&send_request);

    MPI_Irecv(&response, 1, MPI_INT, target_worker, TAG_WORK_RESPONSE, MPI_COMM_WORLD, &recv_request);

    double start_time = MPI_Wtime();
    while (true) {
        int flag = 0;
        MPI_Test(&recv_request, &flag, &status);
        if (flag) break;  // The operation is completed

        // If termination flag is set, cancel the request to avoid deadlock
        if (terminate_flag.load(std::memory_order_relaxed)) {
            //printMessage("Rank: " + std::to_string(my_rank) + " Termination flag set, canceling work request");
            MPI_Cancel(&recv_request);
            MPI_Request_free(&recv_request);
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (response == 1) { // Work is available
        //printMessage("Rank: " + std::to_string(my_rank) + " Work available from " + std::to_string(target_worker));
        current = recvBranch(target_worker, TAG_WORK_STEALING, MPI_COMM_WORLD);
        //printMessage("Rank: " + std::to_string(my_rank) + " Work received from " + std::to_string(target_worker));
        std::lock_guard<std::mutex> lock(queue_mutex);
        queue.push(std::move(current));
        return true;
    } else {
        //printMessage("Rank: " + std::to_string(my_rank) + " No work available from " + std::to_string(target_worker));
    }
    return false;
}

int BranchNBoundPar::Solve(Graph& g, double &optimum_time, int timeout_seconds) {
	// Start the timeout timer.
	auto global_start_time = MPI_Wtime();

	optimum_time  = -1.0;
	terminate_flag.store(false);

	BranchQueue queue;
	int my_rank;
	int p;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	// Initialize big enough best_ub for all processes.
	std::atomic<unsigned short> best_ub = USHRT_MAX;
	unsigned short ub1 = USHRT_MAX;
	unsigned short ub2 = USHRT_MAX;

	MPI_Status status_recv;
	Branch branch_recv;

	// OpenMP Parallel Region
	/*
	idea is assign specific threads to specific tasks, in particular the
	first three threads are used for gathering best_ub, checking for
	termination and listening for work requests, then the remaining threads are used
	for -one to keep popping from the queue, work_stealing, generate
	omp_tasks(method create_task) and add new branches to the queue -others
	to compute the omp_tasks
	*/
	omp_set_num_threads(4);
	#pragma omp parallel default(shared)
	{
		int tid = omp_get_thread_num();

		if (tid == 0) { // Checks if solution has been found or timeout. 
			thread_0_terminator(my_rank, p, global_start_time, timeout_seconds, optimum_time);
			//printMessage("Rank: " + std::to_string(my_rank) + " Exited thread_0_terminator func.");
			//std::cout << "Rank: " << my_rank << " Exited thread_0_terminator func." << std::endl;
		}else if (tid == 1) { // Updates (gathers) best_ub from time to time.
			thread_1_solution_gatherer(p, best_ub);
			//printMessage("Rank: " + std::to_string(my_rank) + " Exited thread_1_solution_gatherer func.");
			//std::cout << "Rank: " << my_rank << " Exited thread_1_solution_gatherer func." << std::endl;
		}else if (tid == 2) { // Employer thread employs workers by answering their work requests
			thread_2_employer(queue_mutex, queue);
			//printMessage("Rank: " + std::to_string(my_rank) + " Exited thread_2_employer func.");
			//std::cout << "Rank: " << my_rank << " Exited thread_2_employer func." << std::endl;
		}else if (tid == 3) { // TODO: Let more threads do these computations in parallel
			
			Branch current;
			//std::cout << "Rank: " << my_rank << " Starting... " << std::endl;
			//printMessage("Rank: " + std::to_string(my_rank) + " Starting...");

			// Initialize bounds
			int lb = _clique_strat.FindClique(g);
			unsigned short ub;
			_color_strat.Color(g, ub);
			best_ub.store(ub);
	
			// Log initial bounds
			Log_par("[INITIALIZATION] Initial bounds: lb = " + std::to_string(lb) +
				", ub = " + std::to_string(ub), 0);

			std::unique_lock<std::mutex> lock(queue_mutex, std::defer_lock);
			queue.push(Branch(g.Clone(), lb, ub, 1));	// Initial branch with depth 1

			bool first_iteration = true;

			bool has_work = false;
			while (!terminate_flag.load()) {
				has_work = false;
				//std::cout << "Rank " << my_rank << "  Queue empty: " << queue.empty() << " ."<< std::endl;
				{	
					std::lock_guard<std::mutex> lock(queue_mutex);
					if (!queue.empty()) { 
						current = std::move(const_cast<Branch&>(queue.top()));
						queue.pop();
						has_work = true;
					}
				}
				
				// If no work and already passed the initial distributing phase, request work.
				if (!has_work) {
					//#pragma omp single // Only a single thread asks for work.
					//{
					// Notify the root process that this worker is idle
					int idle_status = 1;
					MPI_Send(&idle_status, 1, MPI_INT, 0, TAG_IDLE, MPI_COMM_WORLD);
					// Start requesting work.
					//std::cout << "Rank: " << my_rank << " requesting work..." << std::endl;
					Log_par("[REQUEST] Requesting work...", current.depth);
					//printMessage("Rank: " + std::to_string(my_rank) + " requesting work...");
					while (!terminate_flag.load() && !request_work(my_rank, p, queue, queue_mutex, current)) {
						std::this_thread::sleep_for(std::chrono::milliseconds(10));
					}
					// Work received. Notify the root process that this worker is not idle anymore.
					if(terminate_flag.load()) break;
					idle_status = 0;
					MPI_Send(&idle_status, 1, MPI_INT, 0, TAG_IDLE, MPI_COMM_WORLD);
					Log_par("[REQUEST] Work received.", current.depth);				
					//}
					continue;
				}

				auto current_G = std::move(current.g);
				int current_lb = current.lb;
				unsigned short current_ub = current.ub;

				Log_par("[BRANCH] Processing node: lb = " + std::to_string(current_lb) +
						", ub = " + std::to_string(current_ub), current.depth);

				if (current_lb == current_ub) {
					// If at root (original graph, first iteration), solution found.
					if(first_iteration){
						Log_par(
							"[FOUND] Chromatic number "
							"found (very first computation at root): " + std::to_string(current_lb), current.depth);
						best_ub.store(current_ub);
						MPI_Send(&current_ub, 1, MPI_UNSIGNED_SHORT, 0, TAG_SOLUTION_FOUND, MPI_COMM_WORLD);
						break;
					}
					// If not at root (original graph, first iteration), prune .
					best_ub.store(std::min(current_ub, best_ub.load()));
					Log_par(
						"[PRUNE] Branch pruned at "
						"depth " + std::to_string(current.depth) +
						": lb = " + std::to_string(current_lb) +
						" == ub = " + std::to_string(current_ub),
						current.depth);
					continue;
				}

				// Prune
				if (current_lb >= best_ub.load()) {
					Log_par(
						"[PRUNE] Branch pruned at "
						"depth " + std::to_string(current.depth) +
						": lb = " + std::to_string(current_lb) +
						" >= best_ub = " + std::to_string(best_ub.load()),
						current.depth);
					continue;
				}

				// Start branching 
                std::unique_lock<std::mutex> lock_branching(branching_mutex);
                int u, v;
                std::tie(u, v) = _branching_strat.ChooseVertices(*current_G);
                lock_branching.unlock();
                Log_par("[BRANCH] Branching on vertices: u = " + std::to_string(u) +
                        ", v = " + std::to_string(v),
                        current.depth);

                if (u == -1 || v == -1) {
                    best_ub.store(std::min<unsigned short>(current_G->GetNumVertices(), best_ub.load()));
                    continue;
                }

                std::unique_lock<std::mutex> lock_task(task_mutex);
				
                if (current.depth < my_rank+1) {
					// Keep adding edges for the first `my_rank` levels
					auto G_new = current_G->Clone();
					G_new->AddEdge(u, v);
					int lb2 = _clique_strat.FindClique(*G_new);
					_color_strat.Color(*G_new, ub2);
					
					Log_par("[Add Edge] depth " + std::to_string(current.depth) + 
							", lb = " + std::to_string(lb2) + 
							", ub = " + std::to_string(ub2), current.depth);
				
					{
						std::lock_guard<std::mutex> lock(queue_mutex);
						queue.push(Branch(std::move(G_new), lb2, ub2, current.depth + 1));
					}
				} else if (current.depth == my_rank+1) {
					// Merge vertices once when `current.depth == my_rank`
					auto G_merge = current_G->Clone();
					G_merge->MergeVertices(u, v);
					int lb1 = _clique_strat.FindClique(*G_merge);
					_color_strat.Color(*G_merge, ub1);
				
					Log_par("[Merge] depth " + std::to_string(current.depth) + 
							", lb = " + std::to_string(lb1) + 
							", ub = " + std::to_string(ub2), current.depth);
				
					{
						std::lock_guard<std::mutex> lock(queue_mutex);
						queue.push(Branch(std::move(G_merge), lb1, ub1, current.depth + 1));
					}
				} else {
					// After merging, branch in both directions
					auto G1 = current_G->Clone();
					G1->MergeVertices(u, v);
					int lb1 = _clique_strat.FindClique(*G1);
					_color_strat.Color(*G1, ub1);
					{
						std::lock_guard<std::mutex> lock(queue_mutex);
						queue.push(Branch(std::move(G1), lb1, ub1, current.depth + 1));
					}
				
					auto G2 = current_G->Clone();
					G2->AddEdge(u, v);
					int lb2 = _clique_strat.FindClique(*G2);
					_color_strat.Color(*G2, ub2);
					{
						std::lock_guard<std::mutex> lock(queue_mutex);
						queue.push(Branch(std::move(G2), lb2, ub2, current.depth + 1));
					}
				}				

                lock_task.unlock();
				
				// Update local sbest_ub
				unsigned short previous_best_ub = best_ub.load();
				best_ub.store(std::min({previous_best_ub, ub1, ub2}));
				Log_par("[UPDATE] Updated best_ub: " + std::to_string(best_ub.load()), current.depth);

				// After first iteration change to efficient branching strategy (not random)
				first_iteration = false;
			}
		}
		}
		//printMessage("Rank: " + std::to_string(my_rank) + " Finalizing.");
		Log_par("[TERMINATION] Finalizing... ", 0);
		MPI_Barrier(MPI_COMM_WORLD);
		// End execution
		return best_ub;
	}
		