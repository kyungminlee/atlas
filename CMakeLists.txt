cmake_minimum_required(VERSION 3.10)

include_directories("${CMAKE_SOURCE_DIR}/src")
include_directories("${CMAKE_SOURCE_DIR}/dep/spdlog-1.11.0/include")
# include_directories("/opt/homebrew/opt/spdlog/include")
# include_directories("/opt/homebrew/opt/fmt/include")
link_directories("/opt/homebrew/opt/spdlog/lib")

project(atlas)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)


# include(CheckLanguage)

# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# 	message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting to OFF")
# 	set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "")
# endif()

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# check_language(CUDA)

# find_package(MPI REQUIRED)
# # find_package(CUDA REQUIRED)
# find_package(CUDAToolkit REQUIRED)


# if (MPI_FOUND)
# 	message(STATUS "MPI_VERSION:                 ${MPI_VERSION}")
# 	message(STATUS "MPI_CXX_COMPILER:            ${MPI_CXX_COMPILER}")
# 	message(STATUS "MPI_CXX_COMPILE_OPTIONS:     ${MPI_CXX_COMPILE_OPTIONS}")
# 	message(STATUS "MPI_CXX_COMPILE_DEFINITIONS: ${MPI_CXX_COMPILE_DEFINITIONS}")
# 	message(STATUS "MPI_CXX_INCLUDE_DIRS:        ${MPI_CXX_INCLUDE_DIRS}")
# 	message(STATUS "MPI_CXX_LIBRARIES:           ${MPI_CXX_LIBRARIES}")
# 	message(STATUS "MPI_CXX_LINK_FLAGS:          ${MPI_CXX_LINK_FLAGS}")
# 	message(STATUS "MPIEXEC_EXECUTABLE:          ${MPIEXEC_EXECUTABLE}")
# 	message(STATUS "MPIEXEC_NUMPROC_FLAG:        ${MPIEXEC_NUMPROC_FLAG}")
# 	message(STATUS "MPIEXEC_MAX_NUMPROCS:        ${MPIEXEC_MAX_NUMPROCS}")
# 	message(STATUS "MPIEXEC_PREFLAGS:            ${MPIEXEC_PREFLAGS}")
# 	message(STATUS "MPIEXEC_POSTFLAGS:           ${MPIEXEC_POSTFLAGS}")
# endif()


function(set_cxx_options target)
	target_compile_options(${target} PRIVATE
		$<$<COMPILE_LANGUAGE:CXX>:
		-Wall
		-Wextra
		-Wcast-align;
		-Wdisabled-optimization;
		-Wformat=2;
		-Winit-self;
		-Wlogical-op;
		-Wmissing-include-dirs;
		-Woverloaded-virtual;
		-Wpointer-arith;
		-Wshadow;
		-Wstrict-aliasing;
		-Wswitch-enum;
		-Wvla;
		>
		)
endfunction()

function(set_cuda_options target)
	target_compile_options(${target} PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:
		--Wno-deprecated-gpu-targets;
		--expt-extended-lambda;
		-Xcompiler=-Wall;
		-Xcompiler=-Wextra;
		-Xcompiler=-Wcast-align;
		-Xcompiler=-Wdisabled-optimization;
		-Xcompiler=-Wformat=2;
		-Xcompiler=-Winit-self;
		-Xcompiler=-Wlogical-op;
		-Xcompiler=-Wmissing-include-dirs;
		-Xcompiler=-Woverloaded-virtual;
		-Xcompiler=-Wpointer-arith;
		-Xcompiler=-Wshadow;
		-Xcompiler=-Wstrict-aliasing;
		-Xcompiler=-Wswitch-enum;
		-Xcompiler=-Wvla;
		-Xptxas=-v;
		>
		)
endfunction()


function(set_cxx_standard target)
	set_property(TARGET ${target} PROPERTY CXX_STANDARD 14)
	set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)
	set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
	# set_property(TARGET ${target} PROPERTY CUDA_STANDARD 11)
	# set_property(TARGET ${target} PROPERTY CUDA_STANDARD_REQUIRED ON)
endfunction()



# add_executable(splitarray src/splitarray.cpp)
# target_link_libraries(splitarray MPI::MPI_CXX CUDA::cudart)
add_executable(check-concept test/check-concept.cpp)
set_property(TARGET check-concept PROPERTY CXX_STANDARD 20)
set_property(TARGET check-concept PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET check-concept PROPERTY CXX_STANDARD_REQUIRED ON)
# target_link_libraries(test MPI::MPI_CXX CUDA::cudart)
# set_cxx_standard(test)

add_subdirectory("${CMAKE_SOURCE_DIR}/dep/Catch2-3.1.1")
# find_package(Catch2 3 REQUIRED)
# find_package(Catch2 3 REQUIRED)
# # These tests can use the Catch2-provided main
# add_executable(tests test.cpp)
# target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

# # These tests need their own main
# add_executable(custom-main-tests test.cpp test-main.cpp)
# target_link_libraries(custom-main-tests PRIVATE Catch2::Catch2)


# Include(FetchContent)

# FetchContent_Declare(
#   Catch2
#   GIT_REPOSITORY https://github.com/catchorg/Catch2.git
#   GIT_TAG        v3.0.1 # or a later release
# )

# FetchContent_MakeAvailable(Catch2)

# These tests can use the Catch2-provided main
add_executable(tests test/test-indexer.cpp test/test-array.cpp)
add_executable(test-array test/test-array.cpp)
add_executable(test-split test/test-split.cpp)
# add_executable(tests test/test-indexer.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test-array PRIVATE Catch2::Catch2WithMain)

# add_executable(test_cuda_mem test/test_cuda_mem.cpp)

# These tests need their own main
# add_executable(custom-main-tests test.cpp test-main.cpp)
# target_link_libraries(custom-main-tests PRIVATE Catch2::Catch2)


include(CTest)
include(Catch)
catch_discover_tests(tests)

# Reference https://github.com/cwpearson/mpi_test/blob/master/CMakeLists.txt
