/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */




// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>


// includes, kernels
#include "pde_kernel.cu"

#include "pde.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void gpu_pde( int N, int M, float *par_output); 


extern "C"
void seq_pde(int , int , float *); 

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{

    int N = 100; 
    int M = 1000; 
  
    float *seq_output, *par_output; 

    if (argc > 1) { 
    	N = atoi(argv[1]); 
  	if (argc > 2 ) { 
  	    M = atoi(argv[2]); 
	    }
    }

    if (M > 10000) printf(" Warning: M is too big! \n");

     seq_output = (float *) malloc( sizeof(float ) * (N+1) *(N+1) ); 	
     par_output = (float *) malloc( sizeof(float ) * (N+1) *(N+1) ); 	

	for (int ii = 0; ii <= N; ii++) 
		for (int jj = 0; jj <= N; jj++) {
	        seq_output[ii+jj*(N+1)] = 0.0;
		par_output[ii+jj*(N+1)] = 0.0;

	}
	clock_t start_cpu = clock();
    seq_pde (N, M, seq_output); 
    printf("cpu time: %d\n", clock()-start_cpu);


    // float seq_time =  cutGetTimerValue( timer);
    // printf( "SEQ-Processing time: %f (ms)\n", seq_time);
    // CUT_SAFE_CALL( cutCreateTimer( &timer));	
    // CUT_SAFE_CALL( cutStartTimer( timer));
    clock_t start_gpu = clock();
    gpu_pde(N, M, par_output); 	 
    printf("gpu time: %d\n", clock()-start_gpu);
//     CUT_SAFE_CALL( cutStopTimer( timer));

//     float par_time = cutGetTimerValue( timer);
//     printf( "PAR-Processing time: %f (ms)\n", par_time);
//     printf("Speedup: %f\n", (par_time >0)  ? seq_time/par_time : 0); 

//     CUT_SAFE_CALL( cutDeleteTimer( timer));

    for(int i = 0;i<(N+1)*(N+1);i++) {
        if(abs((seq_output[i]-par_output[i])/seq_output[i])>0.1f) {
            printf("error: base: %f cuda: %f\n",seq_output[i],par_output[i]);
        }
        //printf("base: %f cuda: %f\n",seq_output[i],par_output[i]);
    }
//    CUTBoolean res = cutComparefe( seq_output, par_output, (N+1)*(N+1), 0.1f); 

    //printf( "Test %s\n", (1 == res) ? "PASSED" : "FAILED");

}

////////////////////////////////////////////////////////////////////////////////
//! Run the code at GPU 
////////////////////////////////////////////////////////////////////////////////
void
gpu_pde( int N, int M, float *par_output) 
{
    float *d_res;
    cudaMalloc(reinterpret_cast<void **>(&d_res), sizeof(float ) * (N+1) *(N+1));
    // first calculate how many blocks we need
    int area_per_block = THREAD_PER_BLOCK * THREAD_SIZE;
    int tmp = (N+area_per_block-1)/area_per_block;
    int block_cnt = (1+tmp)*tmp/2;
    dim3 threads(THREAD_PER_BLOCK, THREAD_PER_BLOCK);
    dim3 grid(block_cnt);
    pdeKernel<<<grid, threads>>>(N, M, d_res);
    cudaMemcpy(par_output, d_res, sizeof(float ) * (N+1) *(N+1), 
        cudaMemcpyDeviceToHost);
}
