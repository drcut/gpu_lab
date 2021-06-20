__global__ void gemm(float *__restrict__ feature,
                         float *__restrict__ kernel, float *__restrict__ gemm,
                         int M, int K, int N) {
  float gemm_local[4];
  float gemm_local_rest[4];
  __shared__ float feature_shared[128];
  __shared__ float kernel_shared[128];
  float feature_shared_local[4];
  float kernel_shared_local[4];
  float gemm_local1[4];
  float gemm_local1_rest[4];
  float feature_shared_local1[4];
  float kernel_shared_local1[4];
  if (((int)blockIdx.x) < (M / 16)) {
    for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
      for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
        gemm_local[((i_c_init * 2) + j_c_init)] = 0.000000e+00f;
        gemm_local_rest[((i_c_init * 2) + j_c_init)] = 0.000000e+00f;
      }
    }
    for (int rx_outer = 0; rx_outer < ((K + 7) / 8); ++rx_outer) {
      __syncthreads();
      for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
        if ((rx_outer * 8) < (K - ((int)threadIdx.x))) {
          feature_shared[(((((int)threadIdx.y) * 16) + (ax0_inner * 8)) +
                          ((int)threadIdx.x))] =
              feature[(((rx_outer * 8) + ((((((int)blockIdx.x) * 16) +
                                            (((int)threadIdx.y) * 2)) +
                                           ax0_inner) *
                                          K)) +
                       ((int)threadIdx.x))];
        }
      }
      for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
        if ((rx_outer * 8) < (K - ((int)threadIdx.x))) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - ax1_inner)) {
            kernel_shared[(
                ((((int)threadIdx.x) * 16) + (((int)threadIdx.y) * 2)) +
                ax1_inner)] =
                kernel[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                         (((rx_outer * 8) + ((int)threadIdx.x)) * N)) +
                        ax1_inner)];
          }
        }
      }
      __syncthreads();
      for (int rx_inner_outer = 0; rx_inner_outer < 4; ++rx_inner_outer) {
        for (int ax0 = 0; ax0 < 2; ++ax0) {
          for (int ax1 = 0; ax1 < 2; ++ax1) {
            if (((rx_outer * 8) + (rx_inner_outer * 2)) < (K - ax1)) {
              feature_shared_local[((ax0 * 2) + ax1)] =
                  feature_shared[((((((int)threadIdx.x) * 16) + (ax0 * 8)) +
                                   (rx_inner_outer * 2)) +
                                  ax1)];
            }
          }
        }
        for (int ax01 = 0; ax01 < 2; ++ax01) {
          for (int ax11 = 0; ax11 < 2; ++ax11) {
            if (((rx_outer * 8) + (rx_inner_outer * 2)) < (K - ax01)) {
              if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                  (N - ax11)) {
                kernel_shared_local[((ax01 * 2) + ax11)] =
                    kernel_shared[((((rx_inner_outer * 32) + (ax01 * 16)) +
                                    (((int)threadIdx.y) * 2)) +
                                   ax11)];
              }
            }
          }
        }
        for (int i_c = 0; i_c < 2; ++i_c) {
          for (int j_c = 0; j_c < 2; ++j_c) {
            for (int rx_inner_inner = 0; rx_inner_inner < 2; ++rx_inner_inner) {
              if (((rx_outer * 8) + (rx_inner_outer * 2)) <
                  (K - rx_inner_inner)) {
                if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                    (N - j_c)) {
                  gemm_local[((i_c * 2) + j_c)] += 
                  feature_shared_local[((i_c * 2) + rx_inner_inner)] * 
                  kernel_shared_local[((rx_inner_inner * 2) + j_c)];
                }
              }
            }
          }
        }
      }
    }
    for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
      for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
        if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
            (N - j_inner_inner)) {
          gemm[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                 ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) +
                   i_inner_inner) *
                  N)) +
                j_inner_inner)] =
              gemm_local[((i_inner_inner * 2) + j_inner_inner)];
        }
      }
    }
  } else {
    for (int i_c_init1 = 0; i_c_init1 < 2; ++i_c_init1) {
      for (int j_c_init1 = 0; j_c_init1 < 2; ++j_c_init1) {
        gemm_local1[((i_c_init1 * 2) + j_c_init1)] = 0.000000e+00f;
      }
    }
    for (int rx_outer1 = 0; rx_outer1 < ((K + 7) / 8); ++rx_outer1) {
      for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
        if (((((int)blockIdx.x) * 16) + (((int)threadIdx.y) * 2)) <
            (M - ax0_inner1)) {
          if ((rx_outer1 * 8) < (K - ((int)threadIdx.x))) {
            feature_shared[(((((int)threadIdx.y) * 16) + (ax0_inner1 * 8)) +
                            ((int)threadIdx.x))] =
                feature[(((rx_outer1 * 8) + ((((((int)blockIdx.x) * 16) +
                                               (((int)threadIdx.y) * 2)) +
                                              ax0_inner1) *
                                             K)) +
                         ((int)threadIdx.x))];
          }
        }
      }
      for (int ax1_inner1 = 0; ax1_inner1 < 2; ++ax1_inner1) {
        if ((rx_outer1 * 8) < (K - ((int)threadIdx.x))) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - ax1_inner1)) {
            kernel_shared[(
                ((((int)threadIdx.x) * 16) + (((int)threadIdx.y) * 2)) +
                ax1_inner1)] =
                kernel[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                         (((rx_outer1 * 8) + ((int)threadIdx.x)) * N)) +
                        ax1_inner1)];
          }
        }
      }
      for (int rx_inner_outer1 = 0; rx_inner_outer1 < 4; ++rx_inner_outer1) {
        for (int ax02 = 0; ax02 < 2; ++ax02) {
          for (int ax12 = 0; ax12 < 2; ++ax12) {
            if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
                (M - ax02)) {
              if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) < (K - ax12)) {
                feature_shared_local1[((ax02 * 2) + ax12)] =
                    feature_shared[((((((int)threadIdx.x) * 16) + (ax02 * 8)) +
                                     (rx_inner_outer1 * 2)) +
                                    ax12)];
              }
            }
          }
        }
        for (int ax03 = 0; ax03 < 2; ++ax03) {
          for (int ax13 = 0; ax13 < 2; ++ax13) {
            if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) < (K - ax03)) {
              if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                  (N - ax13)) {
                kernel_shared_local1[((ax03 * 2) + ax13)] =
                    kernel_shared[((((rx_inner_outer1 * 32) + (ax03 * 16)) +
                                    (((int)threadIdx.y) * 2)) +
                                   ax13)];
              }
            }
          }
        }
        for (int i_c1 = 0; i_c1 < 2; ++i_c1) {
          for (int j_c1 = 0; j_c1 < 2; ++j_c1) {
            for (int rx_inner_inner1 = 0; rx_inner_inner1 < 2;
                 ++rx_inner_inner1) {
              if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) <
                  (K - rx_inner_inner1)) {
                if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
                    (M - i_c1)) {
                  if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                      (N - j_c1)) {
                    gemm_local1[((i_c1 * 2) + j_c1)] += 
                    feature_shared_local1[((i_c1 * 2) + rx_inner_inner1)] *
                    kernel_shared_local1[((rx_inner_inner1 * 2) + j_c1)];
                  }
                }
              }
            }
          }
        }
      }
    }
    for (int i_inner_inner1 = 0; i_inner_inner1 < 2; ++i_inner_inner1) {
      for (int j_inner_inner1 = 0; j_inner_inner1 < 2; ++j_inner_inner1) {
        if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
            (M - i_inner_inner1)) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - j_inner_inner1)) {
            gemm[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                   ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) +
                     i_inner_inner1) *
                    N)) +
                  j_inner_inner1)] =
                gemm_local1[((i_inner_inner1 * 2) + j_inner_inner1)];
          }
        }
      }
    }
  }
}
