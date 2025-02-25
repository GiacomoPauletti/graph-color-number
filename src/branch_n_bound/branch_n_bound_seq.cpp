#include "branch_n_bound_seq.hpp"

bool BranchNBoundSeq::CheckTimeout(
    const std::chrono::steady_clock::time_point& start_time,
    int timeout_seconds) {
	auto current_time = std::chrono::steady_clock::now();
	auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
				   current_time - start_time)
				   .count();
	return elapsed_seconds >= timeout_seconds;
}

void BranchNBoundSeq::Log(const std::string& message, int depth = 0,
			  bool is_branching = false) {
	if (_log_file.is_open()) {
		// Indentation based on depth
		std::string indentation(depth * 2, ' ');

		if (is_branching) {
			_log_file << indentation << "[Depth " << depth
				  << "] Branching on u = " << message
				  << std::endl;
		} else {
			_log_file << indentation << message << std::endl;
		}
	}
}

int BranchNBoundSeq::Solve(Graph& g, int timeout_seconds,
			   int iteration_threshold) {
	// Start the timeout timer
	auto start_time = std::chrono::steady_clock::now();

	// Initialize bounds
	int lb = _clique_strat.FindClique(g);
	unsigned short ub;
	_color_strat.Color(g, ub);
	unsigned short best_ub = ub;

	// Log initial bounds
	Log("[INIT] Initial bounds: lb = " + std::to_string(lb) +
	    ", ub = " + std::to_string(ub));

	using BranchQueue = std::priority_queue<Branch, std::vector<Branch>>;
	BranchQueue queue;
	queue.push(
	    Branch(g.GetHistory(), lb, ub, 1));  // Initial branch with depth 1

	// Counter for iterations where best_ub remains unchanged
	int best_ub_unchanged_count = 0;

	// Start computation tree
	Log("========== START ==========", 0);
	Log("Root: (lb = " + std::to_string(lb) +
		", ub = " + std::to_string(ub) + ")",
	    0);

	while (!queue.empty()) {
		// Check for timeout
		if (CheckTimeout(start_time, timeout_seconds)) {
			Log("[TIMEOUT] Timeout reached. Returning best "
			    "solution found so far: " +
				std::to_string(best_ub),
			    0);
			return best_ub;
		}

		// Dequeue the next branch
		Branch current = queue.top();
		queue.pop();
		auto current_G = g.Clone();
		current_G->AddHistory(current.history);
		int current_lb = current.lb;
		unsigned short current_ub = current.ub;

		// Log current node
		Log("Processing node: lb = " + std::to_string(current_lb) +
			", ub = " + std::to_string(current_ub),
		    current.depth);

		// If current_lb == current_ub, chromatic number found
		if (current_lb == current_ub) {
			/*
			Log("[FOUND] Chromatic number found: " +
				std::to_string(current_lb),
			    current.depth);
			Log("========== END ==========", 0);
			*/
			best_ub = std::min(current_ub, best_ub);
			continue;
		}

		// Prune if current_lb >= best_ub
		if (current_lb >= best_ub) {
			Log("[PRUNE] Branch pruned at depth " +
				std::to_string(current.depth) +
				": lb = " + std::to_string(current_lb) +
				" >= best_ub = " + std::to_string(best_ub),
			    current.depth);
			continue;
		}

		// Find two non-adjacent vertices u and v according to strategy
		auto [u, v] = _branching_strat.ChooseVertices(*current_G);
		Log("Branching on vertices: u = " + std::to_string(u) +
			", v = " + std::to_string(v),
		    current.depth, true);

		// If no such pair exists, the graph is complete (we are at a
		// leaf branch)
		if (u == -1 || v == -1) {
			/*
			Log("Graph is complete. Chromatic number = " +
				std::to_string(current_G->GetNumVertices()),
			    current.depth);
			Log("========== END ==========", 0);
			*/
			best_ub = std::min<unsigned short>(current_G->GetNumVertices(),best_ub); //check this
			continue;
		}

		// Branch 1 - Merge u and v (assign same color)
		auto G1 = current_G->Clone();  // Copy Graph
		G1->MergeVertices(u, v);
		int lb1 = _clique_strat.FindClique(*G1);
		unsigned short ub1;
		_color_strat.Color(*G1, ub1);
		Log("[Branch 1] (Merge u, v) lb = " + std::to_string(lb1) +
			", ub = " + std::to_string(ub1),
		    current.depth);
		if (lb1 < best_ub) {
			queue.push(
			    Branch(G1->GetHistory(), lb1, ub1, current.depth + 1));
		}

		// Branch 2 - Add edge between u and v (assign different colors)
		auto G2 = current_G->Clone();  // Copy Graph
		G2->AddEdge(u, v);
		int lb2 = _clique_strat.FindClique(*G2);
		unsigned short ub2;
		_color_strat.Color(*G2, ub2);
		Log("[Branch 2] (Add edge u-v) lb = " + std::to_string(lb2) +
			", ub = " + std::to_string(ub2),
		    current.depth);
		if ((lb2 < best_ub) && (lb2 < ub1)) {
			queue.push(
			    Branch(G1->GetHistory(), lb2, ub2, current.depth + 1));
		}

		// Update best_ub
		int previous_best_ub = best_ub;
		best_ub = std::min({best_ub, ub1, ub2});
		Log("[UPDATE] Updated best_ub: " + std::to_string(best_ub),
		    current.depth);

		// Check if best_ub has changed
		if (best_ub == previous_best_ub) {
			best_ub_unchanged_count++;
		} else {
			best_ub_unchanged_count = 0;  // Reset the counter
		}

		// Check if the iteration threshold is exceeded
		if (best_ub_unchanged_count >= iteration_threshold) {
			Log("[ITERATION THRESHOLD] Reached. Returning best "
			    "solution found so far: " +
				std::to_string(best_ub),
			    0);
			Log("========== END ==========", 0);
			return best_ub;
		}
	}

	Log("[RESULT] Final chromatic number: " + std::to_string(best_ub), 0);
	Log("========== END ==========", 0);
	return best_ub;
}

