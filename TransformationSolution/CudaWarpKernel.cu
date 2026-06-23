#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
* Warp all views in a single kernel launch using a 3D grid.
* blockIdx.z = view index.
* Output buffers are indexed by viewBase = v * width * height.
*/
__global__
void warpKernelMultiView(
    uchar3* rgb,
    float* disparity,
    uchar3* warped,
    float* zBuffer,
    unsigned char* holeMask,
    int width,
    int height,
    int numViews,
    float disparityGain)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int v = blockIdx.z;

    if (x >= width || y >= height || v >= numViews)
        return;

    int idx = y * width + x;

    float disp = disparity[idx];
    if (isinf(disp))
        return;

    float viewOffset = static_cast<float>(v - numViews / 2);

    int shift =
        static_cast<int>(
            disp * disparityGain * viewOffset);

    int newX = x + shift;
    if (newX < 0 || newX >= width)
        return;

    // Per-view slice offset
    size_t viewBase = static_cast<size_t>(v) * width * height;

    int newIdx = static_cast<int>(viewBase + y * width + newX);
    int zIdx  = static_cast<int>(viewBase + idx);

    // z-buffer: larger disparity (closer) occludes smaller
    if (disp > zBuffer[zIdx])
    {
        zBuffer[zIdx] = disp;
        warped[newIdx] = rgb[idx];
        holeMask[newIdx] = 0;
    }
}

void launchWarpKernelMultiView(
    uchar3* d_rgb,
    float* d_disp,
    uchar3* d_warped,
    float* d_zBuffer,
    unsigned char* d_holeMask,
    int width,
    int height,
    int numViews,
    float disparityGain)
{
    dim3 block(16, 16);

    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        numViews);

    warpKernelMultiView <<< grid, block >>> (
        d_rgb,
        d_disp,
        d_warped,
        d_zBuffer,
        d_holeMask,
        width,
        height,
        numViews,
        disparityGain);

    cudaDeviceSynchronize();
}
