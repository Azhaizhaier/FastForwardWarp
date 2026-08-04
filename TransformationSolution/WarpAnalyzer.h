#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#include "WarpingEngine.h"


struct ViewScaleResult
{
    int numViews;
    double avgTimeMs;
    double fps;
};

class WarpBenchmark
{
public:
    static void testWarp(
        WarpingEngine& warper,
        Frame& frame,
        int iterations = 50);

    static std::vector<ViewScaleResult> testViewScale(
        WarpingEngine& warper,
        Frame& frame,
        const std::vector<int>& viewCounts,
        int iterations = 20);

    static void exportViewScaleCSV(
        const std::string& filename,
        const std::vector<std::string>& engineNames,
        const std::vector<std::vector<ViewScaleResult>>& allResults);
};


//
class WarpAnalyzer
{
public:
	// Print a summary table of benchmark results for different warping engines and view counts.
    static void printSummaryTable(
        const std::vector<std::string>& engineNames,
        const std::vector<std::vector<ViewScaleResult>>& allResults);
    //判断两张图片是不是完全相等
	static bool testImagesIfEqual(const cv::Mat& img1, const cv::Mat& img2);
};

