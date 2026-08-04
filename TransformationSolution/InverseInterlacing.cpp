#include "InverseInterlacing.h"

#include <cmath>


// ======================================================================
//  Constructor  — store parameters and pre-compute the pattern mask.
// ======================================================================
InverseInterlacing::InverseInterlacing(
    int displayWidth,
    int displayHeight,
    int viewWidth,
    int viewHeight,
    int viewNumber,
    float theta,
    float subpixel,
    float koff)
    : m_displayWidth(displayWidth)
    , m_displayHeight(displayHeight)
    , m_viewWidth(viewWidth)
    , m_viewHeight(viewHeight)
    , m_viewNumber(viewNumber)
    , m_theta(theta)
    , m_subpixel(subpixel)
    , m_koff(koff)
{
    generatePatternMask();
}

// ======================================================================
//  generatePatternMask
//
//  Builds a CV_8UC3 lookup image where each channel stores the view
//  index for that subpixel position.  The mathematical formula is
//  identical to LenticularInterlacing::buildLUT() so the mapping is
//  guaranteed to be the exact inverse of the forward pass.
//
//  Formula (per subpixel, loop c = 0,1,2 for R,G,B):
//    subpixelX  = x * 3 + (2 - c)
//    r = fmod(subpixelX + koff - 3 * y * theta, subpixel)
//    if r < 0: r += subpixel
//    view = floor(r * viewNumber / subpixel)
//    clamp(view, 0, viewNumber - 1)
//    pattern[y][x][2 - c] = view     // 2-c:  B←c2, G←c1, R←c0
//
//  Complexity: O(displayWidth * displayHeight) — computed once.
// ======================================================================
void InverseInterlacing::generatePatternMask()
{
    m_patternMask = cv::Mat(
        m_displayHeight, m_displayWidth, CV_8UC3);

    for (int y = 0; y < m_displayHeight; y++)
    {
        auto* row = m_patternMask.ptr<cv::Vec3b>(y);

        for (int x = 0; x < m_displayWidth; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                const int subpixelX = x * 3 + (2 - c);

                double r = std::fmod(
                    static_cast<double>(subpixelX)
                    + m_koff
                    - 3.0 * y * m_theta,
                    m_subpixel);

                if (r < 0.0)
                    r += m_subpixel;

                int view = static_cast<int>(
                    r * m_viewNumber / m_subpixel);

                if (view >= m_viewNumber)
                    view = m_viewNumber - 1;

                // c=0 → R (channel 2), c=1 → G (1), c=2 → B (0)
                row[x][2 - c] = static_cast<uchar>(view);
            }
        }
    }
}

// ======================================================================
//  recoverViews
//
//  For every display pixel and each colour channel, determine which view
//  owns that subpixel and copy the value back.
//
//  Assumptions
//    - The interlacing is a pure index permutation (no blending).
//    - Every subpixel in the interlaced image originated from exactly one
//      view, so the inverse recovers the original per-view data.
//
//  Complexity: O(displayWidth * displayHeight) per call.
// ======================================================================
std::vector<cv::Mat> InverseInterlacing::recoverViews(
    const cv::Mat& interlacedImage)
{
    using Pixel = cv::Vec3b;

    // Pre-allocate recovered views (display resolution)
    std::vector<cv::Mat> views(m_viewNumber);
    for (int v = 0; v < m_viewNumber; v++)
        views[v] = cv::Mat(
            m_displayHeight, m_displayWidth,
            CV_8UC3, cv::Scalar(0, 0, 0));

    // Pointer cache: one row-ptr per view, refreshed each scan-line
    std::vector<Pixel*> dstRows(m_viewNumber);

    for (int y = 0; y < m_displayHeight; y++)
    {
        const auto* srcRow = interlacedImage.ptr<Pixel>(y);
        const auto* patRow = m_patternMask.ptr<Pixel>(y);

        for (int v = 0; v < m_viewNumber; v++)
            dstRows[v] = views[v].ptr<Pixel>(y);

        for (int x = 0; x < m_displayWidth; x++)
        {
            const auto src = srcRow[x];
            const auto pat = patRow[x];

            // B subpixel → view pat[0]
            dstRows[pat[0]][x][0] = src[0];
            // G subpixel → view pat[1]
            dstRows[pat[1]][x][1] = src[1];
            // R subpixel → view pat[2]
            dstRows[pat[2]][x][2] = src[2];
        }
    }


    // Resize back to original view resolution if different from display
    if (m_displayWidth != m_viewWidth || m_displayHeight != m_viewHeight)
    {
        const cv::Size viewSize(m_viewWidth, m_viewHeight);
        for (auto& view : views)
        {
            cv::Mat resized;
            cv::resize(view, resized, viewSize, 0, 0, cv::INTER_CUBIC);
            view = std::move(resized);
        }
    }

    return views;
}

// ======================================================================
//  recoverMasks
//
//  Same index-reversal logic as recoverViews, but for hole masks.
//
//  Input may be CV_8UC3 (per-subpixel hole flags) or CV_8UC1 (single
//  flag replicated to all subpixels of the pixel).
//
//  Complexity: O(displayWidth * displayHeight) per call.
// ======================================================================
std::vector<cv::Mat> InverseInterlacing::recoverMasks(
    const cv::Mat& interlacedMask)
{
    const bool singleChannel = (interlacedMask.type() == CV_8UC1);

    // Pre-allocate recovered masks (initialised to 255 = hole)
    std::vector<cv::Mat> masks(m_viewNumber);
    for (int v = 0; v < m_viewNumber; v++)
        masks[v] = cv::Mat(
            m_displayHeight, m_displayWidth,
            CV_8UC1, cv::Scalar(0));

    std::vector<uchar*> dstRows(m_viewNumber);

    for (int y = 0; y < m_displayHeight; y++)
    {
        const auto* patRow = m_patternMask.ptr<cv::Vec3b>(y);

        for (int v = 0; v < m_viewNumber; v++)
            dstRows[v] = masks[v].ptr<uchar>(y);

        if (singleChannel)
        {
            const auto* srcRow = interlacedMask.ptr<uchar>(y);
            for (int x = 0; x < m_displayWidth; x++)
            {
                const uchar val = srcRow[x];
                dstRows[patRow[x][0]][x] = dstRows[patRow[x][0]][x] | val;
                dstRows[patRow[x][1]][x] = dstRows[patRow[x][1]][x] | val;
                dstRows[patRow[x][2]][x] = dstRows[patRow[x][2]][x] | val;
            }
        }
        else
        {
            using Pixel = cv::Vec3b;
            const auto* srcRow = interlacedMask.ptr<Pixel>(y);

            for (int x = 0; x < m_displayWidth; x++)
            {
                const auto src = srcRow[x];
                const auto pat = patRow[x];

                dstRows[pat[0]][x] = dstRows[pat[0]][x] | src[0];
                dstRows[pat[1]][x] = dstRows[pat[1]][x] | src[1];
                dstRows[pat[2]][x] = dstRows[pat[2]][x] | src[2];
            }
        }
    }

    // Resize back to original view resolution
    if (m_displayWidth != m_viewWidth || m_displayHeight != m_viewHeight)
    {
        const cv::Size viewSize(m_viewWidth, m_viewHeight);
        for (auto& m : masks)
        {
            cv::Mat resized;
            cv::resize(m, resized, viewSize, 0, 0, cv::INTER_NEAREST);
            m = std::move(resized);
        }
    }

    return masks;
}
