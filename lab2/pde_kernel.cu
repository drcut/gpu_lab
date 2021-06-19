#ifndef _PDE_KERNEL_H_
#define _PDE_KERNEL_H_

#include <stdio.h>
#include "pde.h"

////////////////////////////////////////////////////////////////////////////////
//! GPU version of pde_parallel function 
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__device__ float calculate_a(int i, int j,int M) {
    float res = 0;
    for (int k = 1; k <= M; k++)
            res += (float)(i * j) / (float)(i + j + k);
    return res;
}
__global__ void
pdeKernel(int N, int M, float* g_odata) 
{
    int area_per_block = THREAD_PER_BLOCK * THREAD_SIZE;
    int tmp = (N+area_per_block-1)/area_per_block;
    // fold 1-D grid into 2-D grid
    int bx = blockIdx.x;
    int block_x, block_y;
    int t = 1;
    for(int i = 1;i<=tmp;i++) {
        if(bx<i) {
            block_x = i-1;
            block_y = bx;
            break;
        }
        bx-=i;
    }
    // calculate the region this thread need to calculate
    int left = max(2, block_x*THREAD_PER_BLOCK*THREAD_SIZE + THREAD_SIZE * threadIdx.x);
    int right = min(left + THREAD_SIZE,N-1);
    int top = max(2, block_y*THREAD_PER_BLOCK*THREAD_SIZE + THREAD_SIZE*threadIdx.y);
    int down = min(top+THREAD_SIZE,N-1);

    // calculate (1+1/2+1/3+...+1/M) and (1+1/(2*2)+...+1/(M*M))
    float factor1 = 0.0;
    float factor2 = 0.0;
    for(int i = 1;i<=M;i++) {
        factor1+=1.0/i;
        factor2+=1.0/(i*i);
    }
    // calculate each point
    for(int x = left;x<=right;x++)
        for(int y = top; y<=down;y++) {
            if(y<=x) {
                float sum1 = calculate_a(x-1,y,M)+
                            calculate_a(x+1,y,M)+
                            calculate_a(x,y-1,M)+
                            calculate_a(x,y+1,M);
                sum1*=factor1;

                float sum2 = calculate_a(x-1,y-1,M)+
                            calculate_a(x+1,y+1,M)+
                            calculate_a(x+1,y-1,M)+
                            calculate_a(x-1,y+1,M);
                sum2*=factor2;
                g_odata[x + y * (N + 1)]=sum1+sum2;
            }
        }
}

#endif // #ifndef _PDE_KERNEL_H_
