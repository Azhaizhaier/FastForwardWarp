#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "ImageLoader.h"
#include "WarpingEngine.h"
#include "OpenMPDisparityWarpEngine.h"
#include "CudaWarpEngine.h"
#include "Benchmark.h"

static void printSummaryTable(
    const std::vector<std::string>& engineNames,
    const std::vector<std::vector<ViewScaleResult>>& allResults)
{
    constexpr int wEng = 18;
    constexpr int wView = 8;
    constexpr int wAvg = 14;
    constexpr int wFps = 12;
    constexpr int wTotal = wEng + wView + wAvg + wFps + 1;

    std::cout << "\n" << std::string(wTotal, '=') << "\n";
    std::cout << std::left << std::setw(wEng) << "Engine"
              << std::right << std::setw(wView) << "Views"
              << std::setw(wAvg) << "Avg(ms)"
              << std::setw(wFps) << "FPS" << "\n";
    std::cout << std::string(wTotal, '-') << "\n";

    for (size_t e = 0; e < engineNames.size(); ++e)
    {
        for (const auto& r : allResults[e])
        {
            std::cout << std::left << std::setw(wEng) << engineNames[e]
                      << std::right << std::setw(wView) << r.numViews
                      << std::setw(wAvg) << std::fixed << std::setprecision(2) << r.avgTimeMs
                      << std::setw(wFps) << std::setprecision(1) << r.fps << "\n";
        }
        if (e + 1 < engineNames.size())
            std::cout << std::string(wTotal, '-') << "\n";
    }
    std::cout << std::string(wTotal, '=') << "\n";
}

int main()
{
    ImageLoader loader;
    std::string rgb_path = "C:\\Users\\admin\\Desktop\\data\\img\\img0.png";
    std::string depth_path = "C:\\Users\\admin\\Desktop\\data\\depth\\disp0.pfm";

    Frame frame = loader.loaderFrame(rgb_path, depth_path);
    if (frame.rgb.empty() || frame.depth.empty())
    {
        std::cerr << "Error loading images." << std::endl;
        return -1;
    }

    std::cout << "Image size: "
              << frame.rgb.cols << " x " << frame.rgb.rows << "\n"
              << "Depth type: "
              << (frame.depth.type() == CV_32FC1 ? "CV_32FC1 (float)" :
                  frame.depth.type() == CV_8UC1  ? "CV_8UC1" : "other")
              << std::endl;

    const std::vector<int> viewCounts = {5, 10, 15, 20, 25, 30};
    const int iterations = 10;

    std::vector<std::string> engineNames;
    std::vector<std::vector<ViewScaleResult>> allResults;

    // 1. WarpingEngine (base class, handles both 8U and 32F depth)
    {
        WarpingEngine engine;
        std::cout << "\n--- WarpingEngine (Base, 1 thread) ---" << std::endl;
        auto results = Benchmark::testViewScale(engine, frame, viewCounts, iterations);
        engineNames.push_back("WarpingEngine");
        allResults.push_back(results);
    }

    // 2. OpenMPDisparityWarpEngine
    {
        OpenMPDisparityWarpEngine engine;
        std::cout << "\n--- OpenMPDisparityWarpEngine ---" << std::endl;
        auto results = Benchmark::testViewScale(engine, frame, viewCounts, iterations);
        engineNames.push_back("OpenMP");
        allResults.push_back(results);
    }

    // 3. CudaWarpEngine
    {
        CudaWarpEngine engine;
        std::cout << "\n--- CudaWarpEngine ---" << std::endl;
        auto results = Benchmark::testViewScale(engine, frame, viewCounts, iterations);
        engineNames.push_back("CUDA");
        allResults.push_back(results);
    }

    // Print summary table
    printSummaryTable(engineNames, allResults);

    // Export CSV
    Benchmark::exportViewScaleCSV("benchmark_viewscale.csv", engineNames, allResults);

    // Show a sample result image
    std::cout << "\nPress any key to close the preview window." << std::endl;
    cv::imshow("CUDA Warp - Center View", frame.warpedViews[CENTER_VIEW]);
    cv::waitKey(0);

    return 0;
}
