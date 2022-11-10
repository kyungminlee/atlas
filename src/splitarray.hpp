#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>

#include "range.hpp"
#include "Util.h"


#if 0
template <size_t D, typename T>
struct RaggedArray {
	static constexpr size_t Dim = D;
	static constexpr size_t dim() { return Dim; }

	std::vector<size_t> _offset;
	std::vector<T> _data;
};
#endif


struct ProcessorID {
	int mpi_rank;
	int gpu_device;
	bool operator==(ProcessorID const & rhs) const {
		return mpi_rank == rhs.mpi_rank && gpu_device == rhs.gpu_device;
	}
	bool operator<(ProcessorID const & rhs) const {
		if (mpi_rank < rhs.mpi_rank) { return true; }
		if (mpi_rank > rhs.mpi_rank) { return false; }
		if (gpu_device < rhs.gpu_device) { return true; }
		if (gpu_device > rhs.gpu_device) { return false; }
		return false;
	}
};

namespace std {

std::ostream & operator<<(std::ostream & os, ProcessorID const & id) {
	return os << '(' << id.mpi_rank << ',' << id.gpu_device << ')';
}
// TODO implement hash for ProcessorID

template <>
struct hash<ProcessorID> {
	size_t operator()(ProcessorID const & id) const {
		hash<int> int_hash;
		hash<size_t> size_t_hash;
		return size_t_hash(int_hash(id.mpi_rank)) ^ int_hash(id.gpu_device);
	}
};

} // namespace std;


struct SplitManager {
	SplitManager(MPI_Comm comm): _mpi_comm(comm) {
		int mpi_ierr; 
		int mpi_size, mpi_rank;
		mpi_ierr = MPI_Comm_size(comm, &mpi_size);
		mpi_ierr = MPI_Comm_rank(comm, &mpi_rank);

		int gpu_size;
		cudaError_t cuda_stat;
		cuda_stat = cudaGetDeviceCount(&gpu_size);

		std::vector<int> gpu_size_list(mpi_size, -1);

		mpi_ierr = MPI_Allgather(&gpu_size, 1, MPI_INT, gpu_size_list.data(), 1, MPI_INT, comm);

		std::vector<int> gpu_offset_list;
		gpu_offset_list.reserve(mpi_size + 1);
		gpu_offset_list.push_back(0);
		for (int r = 0 ; r < mpi_size ; ++r) {
			gpu_offset_list.push_back(gpu_offset_list.back() + gpu_size_list[r]);
		}
		int total_gpu_size = gpu_offset_list.back();
		std::vector<ProcessorID> gpu_id_list;
		for (int r = 0 ; r < mpi_size ; ++r) {
			for (int g = 0 ; g < gpu_size_list[r] ; ++g) {
				gpu_id_list.push_back({r, g});
			}
		}
		_mpi_size = mpi_size;
		_gpu_size_list = std::move(gpu_size_list);
		_gpu_offset_list = std::move(gpu_offset_list);
		
		for (size_t i = 0 ; i < gpu_id_list.size() ; ++i) {
			_gpu_id_lookup.insert(std::make_pair(gpu_id_list[i], i));
		}
		_gpu_id_list = std::move(gpu_id_list);
	}

	auto begin() const { return _gpu_id_list.begin(); }
	auto end() const { return _gpu_id_list.end(); }
	auto begin(int rank) const { return std::next(_gpu_id_list.begin(), _gpu_offset_list.at(rank)); }
	auto end(int rank) const { return std::next(_gpu_id_list.begin(), _gpu_offset_list.at(rank + 1)); }

	int mpi_size() const { return _mpi_size; }
	int gpu_size(int rank) const { return _gpu_size_list.at(rank); }
	int total_gpu_size() const { return _gpu_offset_list.back(); }

	size_t lookup(ProcessorID const & id) const {
		return _gpu_id_lookup.at(id);
	}

	void dump(std::ostream & os = std::cout) const {
		int mpi_ierr; 
		int mpi_rank;
		mpi_ierr = MPI_Comm_rank(_mpi_comm, &mpi_rank);
		for (int r = 0 ; r < _mpi_size ; ++r) {
			MPI_Barrier(_mpi_comm);
			if (r == mpi_rank) {
				std::cout << _gpu_id_list << std::endl;
			}
		}
	}

private:
	MPI_Comm _mpi_comm;
	int _mpi_size;
	std::vector<int> _gpu_size_list; // size is MPI_Comm_size
	std::vector<int> _gpu_offset_list; // size is MPI_Comm_size + 1. gpu_id_list[i] for i in [gpu_offset_list[j], gpu_offset_list[j]) 
	std::vector<ProcessorID> _gpu_id_list;
	std::unordered_map<ProcessorID, size_t> _gpu_id_lookup;
};

