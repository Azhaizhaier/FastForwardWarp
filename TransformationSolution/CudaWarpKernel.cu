#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
* Single-view warp kernel with atomic z-buffer for correct occlusion.
*/
__global__
void warpKernel(
    uchar3* rgb,
    float* disparity,
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
    float disp = disparity[idx];
    if (isinf(disp))
        return;

    int shift =
        static_cast<int>(disp * disparityGain * viewOffset);
    int newX = x + shift;

    if (newX < 0 || newX >= width)
        return;

    int newIdx = y * width + newX;

    // Atomic z-buffer: only the thread with the largest disparity
    // at the destination position wins and writes its pixel.
    float oldVal = zBuffer[newIdx];
    while (disp > oldVal)
    {
        int assumed = __float_as_int(oldVal);
        int desired = __float_as_int(disp);
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
        d_rgb, d_disp, d_warped, d_zBuffer, d_holeMask,
        width, height, viewOffset, disparityGain);
}
