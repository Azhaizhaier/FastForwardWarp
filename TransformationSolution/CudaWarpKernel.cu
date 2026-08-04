#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
* Single-view warp kernel with atomic z-buffer.
* disparity = signed (for shift direction)
* origDisp  = original non-negative disparity (for z-buffer occlusion)
*/
__global__
void warpKernel(
    uchar3* rgb,
    float* disparity,
    float* origDisp,
    uchar3* warped,
    float* zBuffer,
    unsigned char* holeMask,
    int width,
    int height,
    int viewOffset,
    float disparityGain)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    float signedDisp = disparity[idx];
    float origDispVal = origDisp[idx];

    if (isinf(origDispVal))
        return;

    int shift =
        static_cast<int>(signedDisp * disparityGain * viewOffset);
    int newX = x + shift;

    if (newX < 0 || newX >= width)
        return;

    int newIdx = y * width + newX;

    // Atomic z-buffer using ORIGINAL disparity (non-negative)
    float oldVal = zBuffer[newIdx];
    while (origDispVal > oldVal)
    {
        int assumed = __float_as_int(oldVal);
        int desired = __float_as_int(origDispVal);
        int result = atomicCAS((int*)&zBuffer[newIdx], assumed, desired);
        if (result == assumed)
        {
            warped[newIdx] = rgb[idx];
            holeMask[newIdx] = 0;
            break;
        }
        oldVal = __int_as_float(result);
    }
}

void launchWarpKernel(
    uchar3* d_rgb,
    float* d_disp,
    float* d_origDisp,
    uchar3* d_warped,
    float* d_zBuffer,
    unsigned char* d_holeMask,
    int width,
    int height,
    int viewOffset,
    float disparityGain)
{
    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    warpKernel <<< grid, block >>> (
        d_rgb, d_disp, d_origDisp, d_warped, d_zBuffer, d_holeMask,
        width, height, viewOffset, disparityGain);
}
