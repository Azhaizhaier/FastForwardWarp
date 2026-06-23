#include "CudaWarpEngine.h"

#include "CudaWarpKernel.cuh"

#include <cuda_runtime.h>
#include <vector>

void CudaWarpEngine::process(
    Frame& frame)
{
    int width  = frame.rgb.cols;
    int height = frame.rgb.rows;
    int numViews = this->m_numViews;

    // Per-view byte counts
    size_t rgbBytes   = static_cast<size_t>(width) * height * sizeof(uchar3);
    size_t dispBytes  = static_cast<size_t>(width) * height * sizeof(float);
    size_t maskBytes  = static_cast<size_t>(width) * height * sizeof(unsigned char);

    // All-view byte counts (contiguous slices)
    size_t totalRgbBytes  = rgbBytes  * numViews;
    size_t totalDispBytes = dispBytes * numViews;
    size_t totalMaskBytes = maskBytes * numViews;

    frame.warpedViews.resize(numViews);
    frame.holeMasks.resize(numViews);

    // --- Device allocations ---
    uchar3*         d_rgb     = nullptr;
    float*          d_disp    = nullptr;
    uchar3*         d_warped  = nullptr;
    float*          d_zBuffer = nullptr;
    unsigned char*  d_holeMask = nullptr;

    cudaMalloc(&d_rgb,     rgbBytes);
    cudaMalloc(&d_disp,    dispBytes);
    cudaMalloc(&d_warped,  totalRgbBytes);
    cudaMalloc(&d_zBuffer, totalDispBytes);
    cudaMalloc(&d_holeMask, totalMaskBytes);

    // Upload inputs (once)
    cudaMemcpy(d_rgb,  frame.rgb.ptr<uchar3>(),    rgbBytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_disp, frame.depth.ptr<float>(),   dispBytes, cudaMemcpyHostToDevice);

    // Initialise all output slices at once
    cudaMemset(d_warped,  0,   totalRgbBytes);
    cudaMemset(d_zBuffer, 0,   totalDispBytes);
    cudaMemset(d_holeMask, 255, totalMaskBytes);

    // --- Single kernel launch: all views in parallel ---
    launchWarpKernelMultiView(
        d_rgb, d_disp,
        d_warped, d_zBuffer, d_holeMask,
        width, height, numViews,
        DISPARITY_GAIN);

    // --- Single D2H copy per buffer ---
    std::vector<uchar3>        hostWarped(numViews * width * height);
    std::vector<unsigned char> hostMask  (numViews * width * height);

    cudaMemcpy(hostWarped.data(), d_warped,   totalRgbBytes,  cudaMemcpyDeviceToHost);
    cudaMemcpy(hostMask.data(),   d_holeMask, totalMaskBytes, cudaMemcpyDeviceToHost);

    // Split contiguous results into per-view cv::Mat
    for (int v = 0; v < numViews; v++)
    {
        size_t offset = static_cast<size_t>(v) * width * height;

        cv::Mat warped(height, width, CV_8UC3, hostWarped.data() + offset);
        cv::Mat mask(height, width, CV_8UC1, hostMask.data() + offset);

        frame.warpedViews[v] = warped.clone();
        frame.holeMasks[v]   = mask.clone();
    }

    // --- Cleanup ---
    cudaFree(d_rgb);
    cudaFree(d_disp);
    cudaFree(d_warped);
    cudaFree(d_zBuffer);
    cudaFree(d_holeMask);
}
