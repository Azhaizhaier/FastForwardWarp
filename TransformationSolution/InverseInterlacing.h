#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

// -----------------------------------------------------------------------
//  InverseInterlacing
//
//  Recovers per-view images from an interlaced image by reversing the
//  lenticular mapping.  The forward mapping (LenticularInterlacing) is a
//  pure index permutation — it copies one subpixel from one view without
//  blending, interpolation or averaging — therefore the inverse is exact.
//
//  The pattern mask is generated once in the constructor using the same
//  mathematical formula as the forward pass, so the recovered views are
//  bit-identical to the originals for any pixel that was not a hole.
//
//  Complexity
//    Constructor   : O(displayWidth * displayHeight)
//    recoverViews  : O(displayWidth * displayHeight)  (one pass)
//    recoverMasks  : O(displayWidth * displayHeight)  (one pass)
// -----------------------------------------------------------------------
class InverseInterlacing
{
public:
    InverseInterlacing(
        int displayWidth,
        int displayHeight,
        int viewWidth,
        int viewHeight,
        int viewNumber,
        float theta,
        float subpixel,
        float koff);

    // Recover per-view colour images from the interlaced frame.
    // Input:  CV_8UC3 interlaced RGB image  (displayWidth x displayHeight)
    // Output: vector of CV_8UC3 views       (displayWidth x displayHeight each)
    std::vector<cv::Mat> recoverViews(
        const cv::Mat& interlacedImage);

    // Recover per-view hole masks.
    // Input:  CV_8UC3 or CV_8UC1 interlaced mask  (displayWidth x displayHeight)
    //         For CV_8UC1 the same value is applied to all three subpixels.
    // Output: vector of CV_8UC1 masks per view.
    std::vector<cv::Mat> recoverMasks(
        const cv::Mat& interlacedMask);

private:
    void generatePatternMask();

    int m_displayWidth;
    int m_displayHeight;
    int m_viewWidth;
    int m_viewHeight;
    int m_viewNumber;
    float m_theta;
    float m_subpixel;
    float m_koff;

    // CV_8UC3, each channel stores the view index for that subpixel.
    //   channel 0 (B) = blue subpixel view
    //   channel 1 (G) = green subpixel view
    //   channel 2 (R) = red subpixel view
    cv::Mat m_patternMask;
};
