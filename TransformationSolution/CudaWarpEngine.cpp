#include "CudaWarpEngine.h"

#include "CudaWarpKernel.cuh"

#include <cuda_runtime.h>

void CudaWarpEngine::process(
    Frame& frame)
{
    int width     = frame.rgb.cols;
    int height    = frame.rgb.rows;
    int numViews  = this->m_numViews;

    size_t rgbBytes  = static_cast<size_t>(width) * height * sizeof(uchar3);
    size_t dispBytes = static_cast<size_t>(width) * height * sizeof(float);
    size_t maskBytes = static_cast<size_t>(width) * height * sizeof(unsigned char);

    frame.warpedViews.resize(numViews);
    frame.holeMasks.resize(numViews);

    // Device allocations (single-view, reused per view)
    uchar3*        d_rgb      = nullptr;
    float*         d_disp     = nullptr;
    float*         d_origDisp = nullptr;
    uchar3*        d_warped   = nullptr;
    float*         d_zBuffer  = nullptr;
    unsigned char* d_holeMask = nullptr;

    cudaMalloc(&d_rgb,      rgbBytes);
    cudaMalloc(&d_disp,     dispBytes);
    cudaMalloc(&d_origDisp, dispBytes);
    cudaMalloc(&d_warped,   rgbBytes);
    cudaMalloc(&d_zBuffer,  dispBytes);
    cudaMalloc(&d_holeMask, maskBytes);

    // Upload signed disparity (for shift) and original disparity (for z-buffer)
    cudaMemcpy(d_rgb,      frame.rgb.ptr<uchar3>(),       rgbBytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_disp,     frame.depth.ptr<float>(),      dispBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_origDisp, frame.depthOrig.ptr<float>(),  dispBytes, cudaMemcpyHostToDevice);

    // Serial for(v) loop
    for (int v = 0; v < numViews; v++)
    {
        int viewOffset = v - numViews / 2;

        cv::Mat warped(height, width, CV_8UC3, cv::Scalar(0));
        cv::Mat mask(height, width, CV_8UC1, cv::Scalar(255));

        cudaMemset(d_warped,  0,   rgbBytes);
        cudaMemset(d_zBuffer, 0,   dispBytes);
        cudaMemset(d_holeMask, 255, maskBytes);

        launchWarpKernel(
            d_rgb, d_disp, d_origDisp, d_warped, d_zBuffer, d_holeMask,
            width, height, viewOffset, DISPARITY_GAIN);

        cudaMemcpy(warped.ptr<uchar3>(),
                   d_warped,   rgbBytes,  cudaMemcpyDeviceToHost);
        cudaMemcpy(mask.ptr<unsigned char>(),
                   d_holeMask, maskBytes, cudaMemcpyDeviceToHost);

        frame.warpedViews[v] = warped;
        frame.holeMasks[v]   = mask;
    }

    cudaFree(d_rgb);
    cudaFree(d_disp);
    cudaFree(d_origDisp);
    cudaFree(d_warped);
    cudaFree(d_zBuffer);
    cudaFree(d_holeMask);
}
