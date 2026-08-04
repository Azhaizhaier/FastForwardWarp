#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>

// ---------------------------------------------------------------------------
// HoleStatistics — aggregated metrics for one or more hole masks.
// All fields are zero-initialised by default.
// ---------------------------------------------------------------------------
struct HoleStatistics
{
    // ---- Basic ----
    int width = 0;
    int height = 0;

    size_t totalPixels = 0;
    size_t holePixels = 0;
    double holeRatio = 0.0;

    // ---- Connected-component metrics ----
    int connectedRegions = 0;
    int largestRegion = 0;
    double averageRegion = 0.0;

    // ---- Bounding-box metrics (from CC_STAT_WIDTH / HEIGHT) ----
    double averageWidth = 0.0;
    double averageHeight = 0.0;
    int maxWidth = 0;
    int maxHeight = 0;

    // ---- Hole-size histogram bins ----
    int hole_1_10 = 0;
    int hole_11_50 = 0;
    int hole_51_100 = 0;
    int hole_101_500 = 0;
    int hole_501_plus = 0;

    // ---- Raw data for downstream analysis ----
    std::vector<int> regionAreas;
};

// ---------------------------------------------------------------------------
// HoleAnalyzer — static utility that computes HoleStatistics for single or
// multiple hole masks.
// ---------------------------------------------------------------------------
class HoleAnalyzer
{
public:
    // Analyse a single 8-bit hole mask (255 = hole, 0 = valid).
    static HoleStatistics analyze(
        const cv::Mat& holeMask);

    // Analyse every mask in the vector, returning per-view results.
    static std::vector<HoleStatistics> analyzeViews(
        const std::vector<cv::Mat>& holeMasks);

    // Accumulate all masks into one set of statistics.  The result
    // represents the multi-view rendering as a whole.
    static HoleStatistics analyzeOverall(
        const std::vector<cv::Mat>& holeMasks);

    // Formatted console output for a single statistics object.
    static void print(
        const HoleStatistics& stat);

    // Formatted console output for a vector of per-view statistics.
    static void print(
        const std::vector<HoleStatistics>& stats);

    // Export the statistics to a CSV file (Metric,Value).
    static void saveCSV(
        const HoleStatistics& statistics,
        const std::string& filename);
};
