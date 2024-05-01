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

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

#add_executable(gpu main.cpp gpu.cu common.h)
#target_compile_features(gpu PRIVATE cxx_std_11)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")
#target_compile_options(gpu PRIVATE -Wall -pedantic)

add_executable(omp main.cpp omp.cpp common.h)
target_compile_features(omp PRIVATE cxx_std_11)
target_link_libraries(omp PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(omp PRIVATE -Wall -pedantic -fopenmp)


add_executable(serial main.cpp serial.cpp common.h)
target_compile_features(serial PRIVATE cxx_std_11)
target_link_libraries(serial ${BLAS_LIBRARIES})
target_compile_options(serial PRIVATE -Wall -pedantic)
