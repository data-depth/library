#pragma once
#include <cuda_runtime.h>
#include <math.h>

// Extern C for python usage
extern "C" {
    // Declaring cuda kernel before 
    __global__ void abs_kernel(float* matrix, int rows, int cols);
    // Declare the host function - important for windows
    #ifdef _WIN32
    __declspec(dllexport)
    #endif
    void abs_matrix(float* d_matrix, int rows, int cols); // declare function 
    } 