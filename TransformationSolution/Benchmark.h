#pragma once

#include "Frame.h"
#include "WarpingEngine.h"
#include <vector>
#include <string>

struct ViewScaleResult
{
    int numViews;
    double avgTimeMs;
    double fps;
};

class Benchmark
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
