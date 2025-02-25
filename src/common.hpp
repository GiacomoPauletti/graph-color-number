#ifndef COMMON_HPP
#define COMMON_HPP

#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>
#include <cstring>
#include <iostream>

#include "graph.hpp"
#include "csr_graph.hpp"

template <class Value>
using VertexMap = std::map<unsigned int, Value, std::less<unsigned int>>;

using VertexSet = std::vector<unsigned int>;  // comparator needs to be added

template <class Value>
using VertexPairMap = std::map<std::pair<unsigned int, unsigned int>,
			       Value>;	// comparator needs to be added

template <class Value>
using VertexPairSet = std::set<std::pair<unsigned int, unsigned int>,
			       Value>;	// comparator needs to be added

template <class Value>
using EdgeMap = VertexPairMap<Value>;  // comparator needs to be added

using Edges = std::vector<std::vector<bool>>;

void GetNeighbours(const Edges& edges, const unsigned int vertex_index,
		   VertexSet& neighbours);

/**
 * @brief Structure to manage branches and priorities in parallel
 * branch-and-bound solver.
 */
using GraphPtr = std::unique_ptr<Graph>;
struct Branch {
	Graph::GraphHistory history;
	int lb;
	unsigned short ub;
	int depth;

	// Default constructor
	Branch() = default;

	// Parameterized constructor
	Branch(const Graph::GraphHistory& history, int lower, unsigned short upper, int dp)
	    : history(history), lb(lower), ub(upper), depth(dp) {}

	// Copy constructor
	Branch(const Branch& b)
	    : history(b.history), lb(b.lb), ub(b.ub), depth(b.depth) {}

	// Move constructor
	Branch(Branch&& b) noexcept
	    : history(b.history), lb(b.lb), ub(b.ub), depth(b.depth) {}

	// Copy assignment operator
	Branch& operator=(const Branch& other) {
		if (this != &other) {
			history = other.history;
			lb = other.lb;
			ub = other.ub;
			depth = other.depth;
		}
		return *this;
	}

	// Move assignment operator
	Branch& operator=(Branch&& other) noexcept {
		if (this != &other) {
			history = other.history;
			lb = other.lb;
			ub = other.ub;
			depth = other.depth;
		}
		return *this;
	}

	// Comparison operator for priority queue
	bool operator<(const Branch& other) const {
		return depth < other.depth;
	}

	// Method to serialize branch
	std::vector<char> serialize() const {
		std::vector<char> buffer;
		std::string historyData = history.Serialize();
		size_t historySize = historyData.size();
	
		buffer.resize(sizeof(lb) + sizeof(ub) + sizeof(depth) + historySize);
	
		char* ptr = buffer.data();
		std::memcpy(ptr, &lb, sizeof(lb));
		ptr += sizeof(lb);
		std::memcpy(ptr, &ub, sizeof(ub));
		ptr += sizeof(ub);
		std::memcpy(ptr, &depth, sizeof(depth));
		ptr += sizeof(depth);
		// serializing GraphHistory
		std::memcpy(ptr, historyData.data(), historySize);
	
		return buffer;
	}

	// Method to deserialize branch
	static Branch deserialize(const std::vector<char>& buffer) {
		Branch b;
		const char* ptr = buffer.data();
	
		std::memcpy(&b.lb, ptr, sizeof(b.lb));
		ptr += sizeof(b.lb);
		std::memcpy(&b.ub, ptr, sizeof(b.ub));
		ptr += sizeof(b.ub);
		std::memcpy(&b.depth, ptr, sizeof(b.depth));
		ptr += sizeof(b.depth);
	
		// deserializing HistoryData
		std::string historyData(buffer.begin() + sizeof(b.lb) + sizeof(b.ub) + sizeof(b.depth),
								buffer.end());
		b.history.Deserialize(historyData);
		return b;
	}
};

#endif	// COMMON_HPP
