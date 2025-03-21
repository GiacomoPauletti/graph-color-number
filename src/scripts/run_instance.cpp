#include <iostream>
#include <fstream>
#include <mpi.h>
#include <cstdlib> // For std::stoi
#include <unordered_map>
#include <filesystem>

#include "branch_n_bound_par.hpp"
#include "branching_strategy.hpp"
#include "clique_strategy.hpp"
#include "fastwclq.hpp"
#include "color.hpp"
#include "recolor.hpp"
#include "advanced_color.hpp"
#include "dsatur_color.hpp"
#include "csr_graph.hpp"
#include "dimacs.hpp"


/**
 * @brief checks if the coloring is valid (i.e. a vertex is not colored or two neighours have the same color)
 * 
 * @param graph graph to check if the coloring is valid
 * @return true if coloring is valid
 * @return false if coloring is not valid
 */
bool CheckColoring(const Graph& graph) {
    unsigned short current_color;
    std::vector<int> neighbours;
    for ( int vertex : graph.GetVertices() ) {
        current_color = graph.GetColor(vertex);
        graph.GetNeighbours(vertex, neighbours);

        if ( current_color == 0 ) {
            return false;
        }

        for ( int neighbour : neighbours ) {
            if ( graph.GetColor(neighbour) == current_color ) {
                return false;
            } 
        }
        
    }
    return true;
}

/**
 * @brief Main function to run the graph coloring solver using the branch and bound method.
 *
 * This function initializes MPI, reads the graph instance from a file, and runs the BranchNBoundPar solver.
 * It handles command-line arguments for the input file, timeout, and solution gathering period.
 * The function compares the computed chromatic number with the expected result and prints the outcome and
 * computation time.
 */
int main(int argc, char** argv) {
    // Default values
    int timeout = 60;
    int sol_gather_period = 10;
    int balanced = 1;
    int color_strategy = 0;
    int logging_flag = 0;
    std::string file_name;
    std::string output_file = "output.txt";

    // Check for required arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file_name> [--timeout=<timeout>] [--sol_gather_period=<period>] "
                  << "[--balanced=<0|1>] [--output=<output_file>] [--logging=<0|1>]\n";
        return 1;
    }

    file_name = argv[1]; // Get filename from arguments

    // Parse optional arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        std::istringstream iss(arg);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            try {
                if (key == "--timeout") {
                    timeout = std::stoi(value);
                    if (timeout <= 0) {
                        std::cerr << "Error: Timeout must be a positive integer.\n";
                        return 1;
                    }
                } else if (key == "--sol_gather_period") {
                    sol_gather_period = std::stoi(value);
                    if (sol_gather_period <= 0) {
                        std::cerr << "Error: Solution gathering period must be a positive integer.\n";
                        return 1;
                    }
                } else if (key == "--balanced") {
                    balanced = std::stoi(value);
                } else if (key == "--color_strategy") {
                    color_strategy = std::stoi(value);
                } else if (key == "--output") {
                    output_file = value;
                } else if (key == "--logging") {
                    logging_flag = std::stoi(value);
                } else {
                    std::cerr << "Error: Unknown argument " << arg << "\n";
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for argument " << key << ".\n";
                return 1;
            }
        } else {
            std::cerr << "Error: Invalid argument format " << arg << ".\n";
            return 1;
        }
    }


    // Load expected results from text file
    std::ifstream txt_file("expected_chi.txt");
    if (!txt_file.is_open()) {
        std::cerr << "Error: Could not open expected results text file.\n";
        return 1;
    }

    std::unordered_map<std::string, int> expected_results;
    std::string key;
    int value;
    while (txt_file >> key >> value) {
        expected_results[key] = value;
    }
    txt_file.close();

    // Check if the expected result for the given file is available
    std::string file_key = std::filesystem::path(file_name).filename().string();
    if (expected_results.find(file_key) == expected_results.end()) {
        std::cerr << "Error: No expected result found for the given file.\n";
        return 1;
    }

    int expected_chromatic_number = expected_results[file_key];

    Dimacs dimacs;
    CSRGraph* graph;
    NeighboursBranchingStrategy branching_strategy;
    FastCliqueStrategy clique_strategy;

    // Light color strategy
    GreedyColorStrategy greedy_color_strategy;
    // Mixed color strategy
    DSaturColorStrategy base_color_strategy;
    DSaturColorStrategy another_dsatur_strategy;
    GreedySwapRecolorStrategy recolor_strategy;
    ColorNRecolorStrategy advanced_color_strategy(base_color_strategy, recolor_strategy);
    InterleavedColorStrategy mixed_color_strategy(greedy_color_strategy, advanced_color_strategy, 5, 2);
    InterleavedColorStrategy another_mixed_color_strategy(another_dsatur_strategy, advanced_color_strategy, 5, 2);
    // Heavy color strategy


    ColorStrategy* color_strategy_obj;
    if (color_strategy == 0) {
        color_strategy_obj = &greedy_color_strategy;
    } else if (color_strategy == 1) {
        color_strategy_obj = &mixed_color_strategy;
    }
    else if (color_strategy == 2) {
        color_strategy_obj = &base_color_strategy;
    } else {
        color_strategy_obj = &another_mixed_color_strategy;
    }

    // Initialize MPI with multithreading enabled
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "MPI does not support full multithreading!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Output arguments
    if (my_rank == 0) {
        std::cout << "Reading file: " << file_name << "\n";
        std::cout << "Using timeout: " << timeout << " seconds\n";
        std::cout << "Using sol_gather_period: " << sol_gather_period << " seconds\n";
        std::cout << "Using balanced approach: " << balanced << "\n";
    }

    // Read the Graph
    std::string full_file_name = "graphs_instances/" + file_name; 
    // All processes read the graph, since they all start with it.
    if (!dimacs.load(full_file_name.c_str())) {
        std::cout << dimacs.getError() << std::endl;
        return 1;
    }
    graph = CSRGraph::LoadFromDimacs(full_file_name);
    std::cout << "Rank " << my_rank << ": Successfully read Graph " << file_name << std::endl;

    BranchNBoundPar solver(branching_strategy, clique_strategy, *color_strategy_obj, "logs/log_" + std::to_string(my_rank) + ".txt", logging_flag==1);
    BalancedBranchNBoundPar balanced_solver(branching_strategy, clique_strategy, *color_strategy_obj, "logs/log_" + std::to_string(my_rank) + ".txt", logging_flag==1);


    // Start the timer.
    auto start_time = MPI_Wtime();
    // Run.
    double optimum_time;    
    int chromatic_number;
    if (balanced) {
        chromatic_number = balanced_solver.Solve(*graph, optimum_time, timeout-0.05, sol_gather_period,  expected_chromatic_number);
    } else {
        chromatic_number = solver.Solve(*graph, optimum_time, timeout-0.05, sol_gather_period, expected_chromatic_number);
    }
    // Stop the timer.
    auto end_time = MPI_Wtime();
    auto time = end_time - start_time;

    // Output results
    if (my_rank == 0) {
        std::cout << "Execution took " << time << " seconds." << std::endl;
        if (optimum_time == -1)
            std::cout << "It was a timeout." << std::endl;
        else
            std::cout << "Solve() finished prematurely measuring " << optimum_time << " seconds. " << std::endl;
        
        if ( !CheckColoring(*graph) ) {
            std::cout << "Coloring is not valid!" << std::endl;
        }

        // Compare with expected chromatic number
        if (chromatic_number != expected_chromatic_number) 
            std::cout << "Failed: expected " << expected_chromatic_number << " but got " << chromatic_number << std::endl;
        else 
            std::cout << "Suceeded: Chromatic number: " << chromatic_number << std::endl;
        
            std::ofstream out(output_file);
            out << "problem_instance_file_name "    << file_name << std::endl;
            out << "cmd line "                      << std::endl;
            out << "solver version "                << std::endl;
            out << "number_of_vertices "            << graph->GetNumVertices() << std::endl;
            out << "number_of_edges: "              << graph->GetNumEdges() << std::endl;
            out << "time_limit_sec "                << timeout << std::endl;
            int n_proc;
            MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
            out << "number_of_worker_processes "    << n_proc << std::endl;
            out << "number_of_cores_per_worker "    << 4 << std::endl;
            if ( optimum_time == -1 ) {
                out << "wall_time_sec "             << "> 10000" << std::endl;
                out << "is_within_time_limit "      << false << std::endl;
            } else {
                out << "wall_time_sec "             << optimum_time << std::endl;
                out << "is_within_time_limit "      << true << std::endl;
            }

            std::vector<unsigned short> colors = graph->GetFullColoring();
            unsigned max_color = 0;
            for ( int color : colors ) {
                if ( color > max_color ) {
                    max_color = color;
                }
            }

            out << "number_of_colors "              << max_color << std::endl;
            std::vector<int> vertices = graph->GetVertices();
            for ( int vertex : vertices ) {
                out << vertex << " " << colors[vertex] << std::endl;
            }

    }


    MPI_Finalize();
    return 0;
}
