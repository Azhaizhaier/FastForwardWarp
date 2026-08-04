#include "MultiViewDirectWarpEngine.h"

#include <vector>

void MultiViewDirectWarpEngine::process(Frame& frame)
{
    const int width = frame.rgb.cols;
    const int height = frame.rgb.rows;
    const int numViews = this->m_numViews;

    // -------------------------------------------------------
    //  1.  Allocate per-view output buffers
    // -------------------------------------------------------
    std::vector<cv::Mat> warpedVec(numViews);
    std::vector<cv::Mat> maskVec(numViews);
    std::vector<cv::Mat> zBufferVec(numViews);

    for (int v = 0; v < numViews; v++)
    {
        warpedVec[v] = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        maskVec[v] = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
        zBufferVec[v] = cv::Mat(height, width, CV_32FC1, cv::Scalar(0.0f));
    }

    // -------------------------------------------------------
    //  2.  Pre-compute view offsets   (cache inner-loop invariant)
    // -------------------------------------------------------
    std::vector<int> viewOffsets(numViews);
    const int centerView = numViews / 2;
    for (int v = 0; v < numViews; v++)
        viewOffsets[v] = v - centerView;

    // -------------------------------------------------------
    //  3.  Pre-allocate row-pointer tables (reused per row)
    // -------------------------------------------------------
    std::vector<cv::Vec3b*> warpedRows(numViews);
    std::vector<float*>       zRows(numViews);
    std::vector<uchar*>      maskRows(numViews);

    // -------------------------------------------------------
    //  4.  Pixel-major loop:  for(y) { for(x) { for(v) } }
    // -------------------------------------------------------
    for (int y = 0; y < height; y++)
    {
        // Read input rows once per scan-line
        const float* signedRow = frame.depth.ptr<float>(y);     // signed for shift
        const float* origRow = frame.depthOrig.ptr<float>(y);    // original for z-buffer
        const cv::Vec3b* rgbRow = frame.rgb.ptr<cv::Vec3b>(y);

        // Refresh per-view output row pointers for this y
        for (int v = 0; v < numViews; v++)
        {
            warpedRows[v] = warpedVec[v].ptr<cv::Vec3b>(y);
            zRows[v] = zBufferVec[v].ptr<float>(y);
            maskRows[v] = maskVec[v].ptr<uchar>(y);
        }

        for (int x = 0; x < width; x++)
        {
            const float signedDisp = signedRow[x];
            const float origDisp = origRow[x];
            if (std::isinf(origDisp))
                continue;

            const cv::Vec3b color = rgbRow[x];

            // Generate all views for this single source pixel
            for (int v = 0; v < numViews; v++)
            {
                const int shift =
                    static_cast<int>(signedDisp * DISPARITY_GAIN * viewOffsets[v]);
                const int newX = x + shift;

                if (newX < 0 || newX >= width)
                    continue;

                // Z-buffer: closer (larger disparity) occludes farther
                if (origDisp > zRows[v][newX])
                {
                    zRows[v][newX] = origDisp;
                    warpedRows[v][newX] = color;
                    maskRows[v][newX] = 0;   // 0 = valid (non-hole)
                }
            }
        }
    }

    // -------------------------------------------------------
    //  5.  Store results
    // -------------------------------------------------------
    frame.warpedViews.resize(numViews);
    frame.holeMasks.resize(numViews);
    for (int v = 0; v < numViews; v++)
    {
        frame.warpedViews[v] = warpedVec[v];
        frame.holeMasks[v] = maskVec[v];
    }
}
