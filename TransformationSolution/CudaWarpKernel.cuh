#pragma once

#include <cuda_runtime.h>

void launchWarpKernelMultiView(
    uchar3* d_rgb,
    float* d_disp,
    uchar3* d_warped,
    float* d_zBuffer,
    unsigned char* d_holeMask,
    int width,
    int height,
    int numViews,
    float disparityGain);
