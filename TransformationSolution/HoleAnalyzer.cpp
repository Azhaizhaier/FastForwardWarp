#include "HoleAnalyzer.h"
#include <numeric>
#include <fstream>
#include <cstdio>
#include <iomanip>

// -----------------------------------------------------------------------
//  Internal helper 鈥?classify a hole size into one of five histogram bins
// -----------------------------------------------------------------------
namespace {

void addToHistogram(HoleStatistics& s, int area)
{
    if      (area <= 10)   ++s.hole_1_10;
    else if (area <= 50)   ++s.hole_11_50;
    else if (area <= 100)  ++s.hole_51_100;
    else if (area <= 500)  ++s.hole_101_500;
    else                   ++s.hole_501_plus;
}

}   // anonymous namespace

// =======================================================================
//  HoleAnalyzer::analyze          (single mask)
// =======================================================================
HoleStatistics HoleAnalyzer::analyze(
    const cv::Mat& holeMask)
{
    if (holeMask.empty())
        return HoleStatistics{};

    HoleStatistics stat;

    // ---- Basic ----
    stat.width       = holeMask.cols;
    stat.height      = holeMask.rows;
    stat.totalPixels = static_cast<size_t>(holeMask.total());
    stat.holePixels  = static_cast<size_t>(cv::countNonZero(holeMask));
    stat.holeRatio   = 100.0 * static_cast<double>(stat.holePixels)
                               / static_cast<double>(stat.totalPixels);

    // ---- Connected components ----
    cv::Mat labels, statsMat, centroids;
    const int regions = cv::connectedComponentsWithStats(
        holeMask, labels, statsMat, centroids, 8);

    stat.connectedRegions = regions - 1;               // exclude label 0

    int64_t sumArea    = 0;
    int64_t sumWidth   = 0;
    int64_t sumHeight  = 0;

    for (int i = 1; i < regions; ++i)
    {
        const int area   = statsMat.at<int>(i, cv::CC_STAT_AREA);
        const int w      = statsMat.at<int>(i, cv::CC_STAT_WIDTH);
        const int h      = statsMat.at<int>(i, cv::CC_STAT_HEIGHT);

        stat.regionAreas.push_back(area);
        sumArea   += area;
        sumWidth  += w;
        sumHeight += h;

        if (area > stat.largestRegion) stat.largestRegion = area;
        if (w    > stat.maxWidth)      stat.maxWidth      = w;
        if (h    > stat.maxHeight)     stat.maxHeight     = h;

        addToHistogram(stat, area);
    }

    if (stat.connectedRegions > 0)
    {
        stat.averageRegion  = static_cast<double>(sumArea)
                            / stat.connectedRegions;
        stat.averageWidth   = static_cast<double>(sumWidth)
                            / stat.connectedRegions;
        stat.averageHeight  = static_cast<double>(sumHeight)
                            / stat.connectedRegions;
    }

    return stat;
}

// =======================================================================
//  HoleAnalyzer::analyzeViews     (per-view results)
// =======================================================================
std::vector<HoleStatistics> HoleAnalyzer::analyzeViews(
    const std::vector<cv::Mat>& holeMasks)
{
    std::vector<HoleStatistics> results;
    results.reserve(holeMasks.size());

    for (const auto& mask : holeMasks)
        results.push_back(analyze(mask));

    return results;
}

// =======================================================================
//  HoleAnalyzer::analyzeOverall   (accumulated across all masks)
// =======================================================================
HoleStatistics HoleAnalyzer::analyzeOverall(
    const std::vector<cv::Mat>& holeMasks)
{
    HoleStatistics overall;

    if (holeMasks.empty())
        return overall;

    // Accumulate per-view metrics, maintaining raw regionAreas.
    int64_t sumArea   = 0;
    int64_t sumWidth  = 0;
    int64_t sumHeight = 0;

    for (const auto& mask : holeMasks)
    {
        const HoleStatistics v = analyze(mask);

        if (overall.width  == 0) overall.width  = v.width;
        if (overall.height == 0) overall.height = v.height;

        overall.holePixels       += v.holePixels;
        overall.totalPixels      += v.totalPixels;
        overall.connectedRegions += v.connectedRegions;

        if (v.largestRegion > overall.largestRegion)
            overall.largestRegion = v.largestRegion;
        if (v.maxWidth      > overall.maxWidth)
            overall.maxWidth      = v.maxWidth;
        if (v.maxHeight     > overall.maxHeight)
            overall.maxHeight     = v.maxHeight;

        // Histogram  鈥?sum bins across views
        overall.hole_1_10    += v.hole_1_10;
        overall.hole_11_50   += v.hole_11_50;
        overall.hole_51_100  += v.hole_51_100;
        overall.hole_101_500 += v.hole_101_500;
        overall.hole_501_plus+= v.hole_501_plus;

        // Raw region areas 鈥?keep per-component data for re-analysis
        sumArea   += std::accumulate(v.regionAreas.begin(),
                                     v.regionAreas.end(), int64_t{0});
        sumWidth  += static_cast<int64_t>(v.averageWidth
                                        * v.connectedRegions);
        sumHeight += static_cast<int64_t>(v.averageHeight
                                        * v.connectedRegions);

        overall.regionAreas.insert(overall.regionAreas.end(),
                                   v.regionAreas.begin(),
                                   v.regionAreas.end());
    }

    // ---- Overall ratio ----
    if (overall.totalPixels > 0)
        overall.holeRatio = 100.0
                          * static_cast<double>(overall.holePixels)
                          / static_cast<double>(overall.totalPixels);

    // ---- Overall averages ----
    if (overall.connectedRegions > 0)
    {
        overall.averageRegion = static_cast<double>(sumArea)
                              / overall.connectedRegions;
        overall.averageWidth  = static_cast<double>(sumWidth)
                              / overall.connectedRegions;
        overall.averageHeight = static_cast<double>(sumHeight)
                              / overall.connectedRegions;
    }

    return overall;
}

// =======================================================================
//  HoleAnalyzer::print            (single)
// =======================================================================
void HoleAnalyzer::print(const HoleStatistics& s)
{
    auto P = [&](const std::string& label, const std::string& value) {
        std::cout << "  " << std::left << std::setw(20) << label
            << " : " << value << "\n";
        };

    std::cout << "\n=============================="
        << "\n  Hole Statistics"
        << "\n==============================\n";

    P("Resolution", std::to_string(s.width) + " x " + std::to_string(s.height));
    P("Total Pixels", std::to_string(s.totalPixels));
    P("Hole Pixels", std::to_string(s.holePixels));

    char ratio[32];
    std::snprintf(ratio, sizeof(ratio), "%.2f%%", s.holeRatio);
    P("Hole Ratio", ratio);

    P("Connected Regions", std::to_string(s.connectedRegions));
    P("Largest Region", std::to_string(s.largestRegion));
    P("Average Region", std::to_string(s.averageRegion));
    P("Average Width", std::to_string(s.averageWidth));
    P("Average Height", std::to_string(s.averageHeight));
    P("Max Width", std::to_string(s.maxWidth));
    P("Max Height", std::to_string(s.maxHeight));

    std::cout << "  ----------------------------\n"
        << "  Histogram\n"
        << "  ----------------------------\n";
    P("1 ~ 10 px", std::to_string(s.hole_1_10));
    P("11 ~ 50 px", std::to_string(s.hole_11_50));
    P("51 ~ 100 px", std::to_string(s.hole_51_100));
    P("101 ~ 500 px", std::to_string(s.hole_101_500));
    P("> 500 px", std::to_string(s.hole_501_plus));

    std::cout << "==============================\n";
}
// =======================================================================
void HoleAnalyzer::print(const std::vector<HoleStatistics>& stats)
{
    for (size_t i = 0; i < stats.size(); ++i)
    {
        std::cout << "\n========== View " << i << " ==========";
        print(stats[i]);
    }
}

// =======================================================================
//  HoleAnalyzer::saveCSV
// =======================================================================
void HoleAnalyzer::saveCSV(
    const HoleStatistics& s,
    const std::string& filename)
{
    std::ofstream out(filename);
    if (!out.is_open())
    {
        std::cerr << "[HoleAnalyzer] Failed to open " << filename << "\n";
        return;
    }

    out << "Metric,Value\n";

    auto row = [&](const char* metric, const std::string& val) {
        out << metric << ',' << val << '\n';
    };

    row("Resolution",
        std::to_string(s.width) + "x" + std::to_string(s.height));
    row("Total Pixels", std::to_string(s.totalPixels));
    row("Hole Pixels",  std::to_string(s.holePixels));

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f%%", s.holeRatio);
    row("Hole Ratio", buf);

    row("Connected Regions", std::to_string(s.connectedRegions));
    row("Largest Region",    std::to_string(s.largestRegion));
    row("Average Region",    std::to_string(s.averageRegion));
    row("Average Width",     std::to_string(s.averageWidth));
    row("Average Height",    std::to_string(s.averageHeight));
    row("Max Width",         std::to_string(s.maxWidth));
    row("Max Height",        std::to_string(s.maxHeight));

    row("Histogram 1~10",    std::to_string(s.hole_1_10));
    row("Histogram 11~50",   std::to_string(s.hole_11_50));
    row("Histogram 51~100",  std::to_string(s.hole_51_100));
    row("Histogram 101~500", std::to_string(s.hole_101_500));
    row("Histogram >500",    std::to_string(s.hole_501_plus));

    out.close();
}
