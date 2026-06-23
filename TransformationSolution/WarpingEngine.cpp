#include "WarpingEngine.h"
#include <iostream>

static cv::Mat normalizeDepth(const cv::Mat& depth)
{
    if (depth.type() == CV_8UC1)
        return depth;

    double minVal, maxVal;
    cv::minMaxLoc(depth, &minVal, &maxVal);
    float range = static_cast<float>(maxVal - minVal);
    if (range < 1e-6f) range = 1e-6f;

    cv::Mat out;
    depth.convertTo(out, CV_8UC1, 255.0 / range, -minVal);
    return out;
}

void WarpingEngine::processOptimizeByPtr(Frame& frame)
{
    const int width = frame.rgb.cols;
    const int height = frame.rgb.rows;
    cv::Mat depth = normalizeDepth(frame.depth);

    frame.warpedViews.resize(this->m_numViews);
    frame.holeMasks.resize(this->m_numViews);

    for (int v = 0; v < this->m_numViews; v++)
    {
        cv::Mat warped(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat mask(height, width, CV_8UC1, cv::Scalar(255));
        cv::Mat zBuffer(height, width, CV_8UC1, cv::Scalar(MAX_DEPTH));

        int viewOffset = v - this->m_numViews / 2;

        for (int y = 0; y < height; y++)
        {
            const uchar* depthRow = depth.ptr<uchar>(y);
            const cv::Vec3b* rgbRow = frame.rgb.ptr<cv::Vec3b>(y);
            cv::Vec3b* warpedRow = warped.ptr<cv::Vec3b>(y);
            uchar* maskRow = mask.ptr<uchar>(y);
            uchar* zRow = zBuffer.ptr<uchar>(y);

            for (int x = 0; x < width; x++)
            {
                uchar depthValue = depthRow[x];
                if (depthValue == 0) continue;

                int disparity = static_cast<int>(
                    (MAX_DEPTH - depthValue) * DISPARITY_SCALE * viewOffset);
                int newX = x - disparity;
                if (newX < 0 || newX >= width) continue;

                if (depthValue < zRow[newX])
                {
                    warpedRow[newX] = rgbRow[x];
                    zRow[newX] = depthValue;
                    maskRow[newX] = 0;
                }
            }
        }

        frame.warpedViews[v] = warped;
        frame.holeMasks[v] = mask;
    }
}

void WarpingEngine::process(Frame& frame)
{
    int width = frame.rgb.cols;
    int height = frame.rgb.rows;
    cv::Mat depth = normalizeDepth(frame.depth);

    frame.warpedViews.clear();
    frame.holeMasks.clear();

    for (int v = 0; v < this->m_numViews; v++)
    {
        cv::Mat warped = cv::Mat::zeros(height, width, CV_8UC3);
        cv::Mat mask = cv::Mat::ones(height, width, CV_8UC1) * 255;
        cv::Mat zBuffer = cv::Mat::ones(height, width, CV_8UC1) * MAX_DEPTH;

        int viewOffset = v - this->m_numViews / 2;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                uchar depthValue = depth.at<uchar>(y, x);
                if (depthValue == 0) continue;

                int disparity = static_cast<int>(
                    (MAX_DEPTH - depthValue) * DISPARITY_SCALE * viewOffset);
                int newX = x - disparity;
                if (newX < 0 || newX >= width) continue;

                uchar storedDepth = zBuffer.at<uchar>(y, newX);
                if (depthValue < storedDepth)
                {
                    warped.at<cv::Vec3b>(y, newX) = frame.rgb.at<cv::Vec3b>(y, x);
                    zBuffer.at<uchar>(y, newX) = depthValue;
                    mask.at<uchar>(y, newX) = 0;
                }
            }
        }

        frame.warpedViews.push_back(warped);
        frame.holeMasks.push_back(mask);
    }
}
