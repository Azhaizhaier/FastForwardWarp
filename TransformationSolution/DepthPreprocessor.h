#pragma once

#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------------
// DisparityConfig --?controls how raw depth is converted to signed disparity.
// ---------------------------------------------------------------------------
struct DisparityConfig
{
    bool enableFocusPlane = true;

    enum class FocusMode
    {
        Fixed,
        Mean,
        Median,
        Percentile
    };

    FocusMode focusMode = FocusMode::Median;

    /// Fixed focus depth
    float fixedDepth = 0.5f;  // Ensure this member is present

    // Percentile
    float percentile = 50.0f;

    // Dead Zone
    float deadZone = 0.03f;

    // Max disparity
    float maxDisparity = 1.0f;

    // Enable non-linear compression
    bool enableCompression = false;
    float gamma = 0.8f;

    // Temporal smoothing coefficient (0.0 = full update, 1.0 = frozen).
    float temporalAlpha = 0.9f;
};
  

// ---------------------------------------------------------------------------
// SignedDisparityResult --?output of the full pipeline.
// ---------------------------------------------------------------------------
struct SignedDisparityResult
{
    cv::Mat disparity;          // CV_32FC1,  signed disparity map
    float   focusDepth = 0.0f;  // estimated / fixed focus-plane depth
    float   minDepth   = 0.0f;  // minimum disparity value
    float   maxDepth   = 0.0f;  // maximum disparity value
};

// ---------------------------------------------------------------------------
// DepthPreprocessor --?converts a raw depth map into a signed disparity map
// with a focus plane, dead zone, and optional non-linear compression.
// ---------------------------------------------------------------------------
class DepthPreprocessor
{
public:
    explicit DepthPreprocessor(const DisparityConfig& config);

    SignedDisparityResult process(const cv::Mat& depth);

private:
    cv::Mat normalizeDepth(const cv::Mat& depth);
    float   estimateFocusDepth(const cv::Mat& normalizedDepth);
    cv::Mat convertToSignedDisparity(const cv::Mat& normalizedDepth, float focusDepth);
    cv::Mat applyDeadZone(const cv::Mat& disparity);
    cv::Mat compressDisparity(const cv::Mat& disparity);

private:
    DisparityConfig m_config;
    float m_previousFocusDepth = -1.0f;
};
