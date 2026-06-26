#pragma once

#include <cuda_runtime.h>

void launchWarpKernel(
    uchar3* d_rgb,
    float* d_disp,
    uchar3* d_warped,
    float* d_zBuffer,
    unsigned char* d_holeMask,
    int width,
    int height,
    int viewOffset,
    float disparityGain);
