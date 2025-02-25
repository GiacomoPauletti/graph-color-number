
#include <iostream>

#include "branch_n_bound_par.hpp"
#include "branching_strategy.hpp"

#include "clique_strategy.hpp"
#include "fastwclq.hpp"
#include "color.hpp"
#include "dsatur_color.hpp"

#include "csr_graph.hpp"
#include "dimacs.hpp"
#include "test_common.hpp"

#include <mpi.h>

#include <cstdlib> // For std::stoi

int main(int argc, char** argv) {
    // Default values
    int timeout = 60;
    int N_trials = 1;
    std::string folder_name = "graphs_instances/";
    std::string file_name;

    // Check for required arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file_name> [timeout] [N_trials]\n";
        return 1;
    }

    file_name = folder_name + argv[1]; // Prefix folder to filename

    // Parse optional timeout argument
    if (argc > 2) {
        try {
            timeout = std::stoi(argv[2]); // Convert timeout to integer
            if (timeout <= 0) {
                std::cerr << "Error: Timeout must be a positive integer.\n";
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid timeout argument.\n";
            return 1;
        }
    }

    // Parse optional N_trials argument
    if (argc > 3) {
        try {
            N_trials = std::stoi(argv[3]); // Convert N_trials to integer
            if (N_trials <= 0) {
                std::cerr << "Error: N_trials must be a positive integer.\n";
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid N_trials argument.\n";
            return 1;
        }
    }


    Dimacs dimacs;
    CSRGraph* graph;
    NeighboursBranchingStrategy branching_strategy;
    FastCliqueStrategy clique_strategy;
    DSaturColorStrategy color_strategy;

	// Initialize MPI with multithreading enabled
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE) {
		std::cerr << "MPI does not support full multithreading!" << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if ( my_rank == 0 ) {
        std::cout << "Reading file: " << file_name << "\n";
        std::cout << "Using timeout: " << timeout << " seconds\n";
        std::cout << "Number of trials: " << N_trials << "\n";
    }


	double optimum_time;
	// Test the solver multiple times on the same graph (since its not deterministic)
	for (int i=0; i<N_trials; i++) {
		// Root process reads the graph
        if (!dimacs.load(file_name.c_str())) {
            std::cout << dimacs.getError() << std::endl;
            return 1;
        }

        graph = CSRGraph::LoadFromDimacs(file_name);

		if (my_rank == 0) {
			
            std::cout << "Succesfully read Graph." << std::endl;
		}
	
		BranchNBoundPar solver(branching_strategy, clique_strategy, color_strategy, "log_par.txt");

		if (my_rank == 0) {
		std::cout << "Starting trial " << i << "..." << std::endl;
		}
		// Start the timer.
		auto start_time = MPI_Wtime();
		int chromatic_number = solver.Solve(*graph, optimum_time, timeout);
		auto end_time = MPI_Wtime();
		auto time = end_time - start_time;
		if (my_rank == 0) {
			std::cout << "Trial " << i << " took " << time << " seconds." << std::endl;
			if (optimum_time == -1)
				std::cout << "It was a timeout." << std::endl;
			else 
				std::cout << "Finalized prematurely with self-measured " << optimum_time << " seconds. " << std::endl;
			std::cout << "Chromatic number: " << chromatic_number << std::endl;
		}
	}

	MPI_Finalize();
	return 0;
}
