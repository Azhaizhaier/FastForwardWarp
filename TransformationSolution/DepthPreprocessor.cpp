#include "DepthPreprocessor.h"

#include <algorithm>
#include <numeric>
#include <cmath>
//#include <iostream>

// =========================================================================
//  Constructor
// =========================================================================
DepthPreprocessor::DepthPreprocessor(const DisparityConfig& config)
    : m_config(config)
{
}

// =========================================================================
//  normalizeDepth
//
//  Input   : CV_32FC1 depth image, arbitrary positive range.
//  Output  : CV_32FC1 depth linearly mapped to [0, 1].
//  Algorithm: (d - minVal) / (maxVal - minVal).
//             If the range is below 1e-6 the output is filled with 0.5.
// =========================================================================
cv::Mat DepthPreprocessor::normalizeDepth(const cv::Mat& depth)
{
    CV_Assert(!depth.empty() && depth.type() == CV_32FC1);

    // Strip Inf/NaN before computing stats
    cv::Mat clean = depth.clone();
    cv::patchNaNs(clean, 0.0f);
    cv::Mat infMask;
    cv::Mat absC;
    cv::absdiff(clean, cv::Scalar(0.0f), absC);
    cv::compare(absC, cv::Scalar(1e20), infMask, cv::CMP_GE);
    clean.setTo(0.0f, infMask);

    double minVal, maxVal;
    cv::minMaxLoc(clean, &minVal, &maxVal);

    const double range = maxVal - minVal;
    if (range < 1e-6)
    {
        return cv::Mat(depth.size(), CV_32FC1, cv::Scalar(0.5f));
    }

    cv::Mat normalized;
    clean.convertTo(normalized, CV_32FC1, 1.0 / range, -minVal / range);
    return normalized;
}

// =========================================================================
//  estimateFocusDepth
//
//  Determines the focus-plane depth from the normalized [0,1] image.
//
//  Modes
//    Fixed       return config.fixedDepth
//    Mean        cv::mean()
//    Median      std::nth_element() over flattened non-NaN values
//    Percentile  std::nth_element() at the requested percentile
//
//  Temporal smoothing is applied unless previousFocusDepth < 0.
// =========================================================================
float DepthPreprocessor::estimateFocusDepth(const cv::Mat& normalizedDepth)
{
    // --- 1.  Raw focus estimate ---
    float currentFocus = 0.0f;

    switch (m_config.focusMode)
    {
    case DisparityConfig::FocusMode::Fixed:
        currentFocus = m_config.fixedDepth;
        break;

    case DisparityConfig::FocusMode::Mean:
        currentFocus = static_cast<float>(cv::mean(normalizedDepth)[0]);
        break;

    case DisparityConfig::FocusMode::Median:
    case DisparityConfig::FocusMode::Percentile:
    {
        // Flatten, ignoring NaN values
        const size_t totalPx = static_cast<size_t>(normalizedDepth.total());
        const float* src     = normalizedDepth.ptr<float>();

        std::vector<float> values;
        values.reserve(totalPx);

        for (size_t i = 0; i < totalPx; ++i)
        {
            const float v = src[i];
            if (!std::isnan(v) && !std::isinf(v))
                values.push_back(v);
        }

        if (values.empty())
        {
            currentFocus = 0.5f;
            break;
        }

        // Determine the rank index
        size_t k;
        if (m_config.focusMode == DisparityConfig::FocusMode::Median)
        {
            k = values.size() / 2;
        }
        else // Percentile
        {
            const float p = std::min(std::max(m_config.percentile, 0.0f), 100.0f);
            k = static_cast<size_t>(p / 100.0f * (values.size() - 1));
        }

        std::nth_element(values.begin(),
                         values.begin() + static_cast<ptrdiff_t>(k),
                         values.end());
        currentFocus = values[k];
        break;
    }
    }

    // --- 2.  Temporal smoothing ---
    if (m_config.enableFocusPlane)
    {
        if (m_previousFocusDepth < 0.0f)
        {
            // First frame 鈥?initialise
            m_previousFocusDepth = currentFocus;
        }
        else
        {
            // Exponential moving average
            const float alpha = std::min(std::max(m_config.temporalAlpha, 0.0f), 1.0f);
            currentFocus = alpha * m_previousFocusDepth
                         + (1.0f - alpha) * currentFocus;
            m_previousFocusDepth = currentFocus;
        }
    }
    else
    {
        m_previousFocusDepth = -1.0f;
    }

    return currentFocus;
}

// =========================================================================
//  convertToSignedDisparity
//
//  disparity = normalizedDepth - focusDepth
//  clamped to [-maxDisparity, +maxDisparity].
// =========================================================================
cv::Mat DepthPreprocessor::convertToSignedDisparity(
    const cv::Mat& normalizedDepth,
    float focusDepth)
{
    const float md = std::abs(m_config.maxDisparity);
    cv::Mat disparity;
    cv::subtract(normalizedDepth, cv::Scalar(focusDepth), disparity);
    // Clamp to  [-md, +md]
    cv::max(disparity, -md, disparity);
    cv::min(disparity,  md, disparity);
    return disparity;
}

// =========================================================================
//  applyDeadZone
//
//  Values where |d| < deadZone are set to zero.
// =========================================================================
cv::Mat DepthPreprocessor::applyDeadZone(const cv::Mat& disparity)
{
    const float dz = std::abs(m_config.deadZone);
    if (dz <= 0.0f)
        return disparity.clone();

    cv::Mat result = disparity.clone();

    // Mask of pixels inside the dead zone
    cv::Mat insideDead;
    cv::threshold(cv::abs(disparity), insideDead, dz, 1.0, cv::THRESH_BINARY_INV);

    result.setTo(0.0f, insideDead > 0);

    return result;
}

// =========================================================================
//  compressDisparity
//
//  If enableCompression:
//      out = sign(d) * |d|^gamma
//  Otherwise identity.
// =========================================================================
cv::Mat DepthPreprocessor::compressDisparity(const cv::Mat& disparity)
{
    if (!m_config.enableCompression)
        return disparity.clone();

    const float g = std::min(std::max(m_config.gamma, 0.1f), 3.0f);
    cv::Mat result;
    cv::pow(cv::abs(disparity), g, result);


    // Build a sign mask: 1.0 where d >= 0, -1.0 where d < 0
    cv::Mat signMask(disparity.size(), CV_32F, cv::Scalar(1.0f));
    cv::Mat neg;
    cv::compare(disparity, 0, neg, cv::CMP_LT);
    signMask.setTo(-1.0f, neg);

    cv::multiply(result, signMask, result);
    return result;
}

// =========================================================================
//  process   full pipeline
//
//  Depth  Normalize  Estimate Focus  Signed Disparity  Dead Zone
//        (optional Compression)  SignedDisparityResult
// =========================================================================
SignedDisparityResult DepthPreprocessor::process(const cv::Mat& depth)
{
    // 1. Normalise to [0, 1]
    double rawMin, rawMax;
    cv::Mat cleanDepth = depth.clone();
    cv::patchNaNs(cleanDepth, 0.0f);
    cv::Mat infD;
    cv::Mat absCD;
    cv::absdiff(cleanDepth, cv::Scalar(0.0f), absCD);
    cv::compare(absCD, cv::Scalar(1e20), infD, cv::CMP_GE);
    cleanDepth.setTo(0.0f, infD);
    //double rawMin, rawMax;
    cv::minMaxLoc(cleanDepth, &rawMin, &rawMax);
    const float range = static_cast<float>(rawMax - rawMin);

    const cv::Mat norm = normalizeDepth(depth);

    // 2. Estimate focus-plane depth
    const float focusDepth = estimateFocusDepth(norm);

    // 3. Convert to signed disparity
    cv::Mat disp = convertToSignedDisparity(norm, focusDepth);

    // 4. Apply dead zone
    disp = applyDeadZone(disp);

    // 5. Optional compression
    disp = compressDisparity(disp);

    // 5.5 Rescale to original depth range
    if (range > 1e-6f)
        disp *= range;

    // 6. Gather result
    SignedDisparityResult result;
    result.disparity  = disp;
    result.focusDepth = focusDepth;

    double minD, maxD;
    cv::minMaxLoc(disp, &minD, &maxD);
    result.minDepth = static_cast<float>(minD);
    result.maxDepth = static_cast<float>(maxD);

    return result;
}
