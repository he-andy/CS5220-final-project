cmake_minimum_required(VERSION 3.11)
project(final_project)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)
# netID
set(NET_ID xxx CACHE STRING "Your netID")

# Packaging system
if (NOT ${NET_ID} STREQUAL xxx)
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "CS5220${NET_ID}_final")
    install(FILES gpu.cu DESTINATION .)
    install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
    include(CPack)
endif ()



set(Prg-Clang "PrgEnv-nvidia")

add_executable(gpu gpu.cu)
target_compile_features(gpu PRIVATE cxx_std_11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -arch=sm_60 --extended-lambda")
find_package(CUDA REQUIRED)
link_directories()
target_link_options(gpu PRIVATE -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64 -lcusolver)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()


target_compile_options(gpu PRIVATE -Wall)

configure_file(job-gpu job-gpu COPYONLY)


add_executable(omp main.cpp omp.cpp common.h)
target_compile_features(omp PRIVATE cxx_std_11)
target_link_libraries(omp PRIVATE OpenMP::OpenMP_CXX -Wl,-ydgelsd_)
target_compile_options(omp PRIVATE -Wall -pedantic -fopenmp -march=znver3)


add_executable(serial main.cpp serial.cpp common.h)
target_compile_features(serial PRIVATE cxx_std_11)
target_link_libraries(serial ${BLAS_LIBRARIES})
target_compile_options(serial PRIVATE -Wall -pedantic -march=znver3)
