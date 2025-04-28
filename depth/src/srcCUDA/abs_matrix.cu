/******************************************************************************/
/* File:             abs_matrix.cu                                            */
/* Created by:       Leonardo Leone                                           */
/* Last revised:     28.04.2025                                               */
/*                                                                            */
/* Contains a cuda function to compute the abs value of a function in         */
/* parallel                                                                   */
/*                                                                            */
/******************************************************************************/

#include <cuda_runtime.h>
#include <math.h>
#include "abs_matrix.cuh"


// Define cuda kernel
__global__ void abs_kernel(float* matrix, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < rows && y < cols) { // check bounds
        int idx = x * cols + y;
        matrix[idx] = fabsf(matrix[idx]);
    }
}
// Define the host function
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
void abs_matrix(float* d_matrix, int rows, int cols) {
    
    /****************************************************************************/
    /* abs_matrix computes the absolute value of a matrix.                      */
    /*                                                                          */
    /* Args:                                                                    */
    /*   d_matrix - the GPU pointer for the matrix                              */
    /*   rows - number of rows in the matrix.                                   */
    /*   cols - number of columns in the matrix.                                */
    /*                                                                          */
    /****************************************************************************/
    
    // optimize block size
    int blockDimX=32;
    int blockDimY=32;
    if(rows<32){blockDimX=rows;};
    if(cols<32){blockDimY=cols;};

    dim3 threadsPerBlock(blockDimX,blockDimY);// define TPB
    dim3 blocksPerGrid((rows + blockDimX-1)/blockDimX, (cols + blockDimY-1) / blockDimY);
    abs_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, rows, cols);
    cudaDeviceSynchronize(); // sync devices -- threads
}