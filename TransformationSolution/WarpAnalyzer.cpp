//#include <vector>
#include <fstream>
//#include <iomanip>

#include "Timer.h"
#include "WarpAnalyzer.h"


void WarpBenchmark::testWarp(
    WarpingEngine& warper,
    Frame& frame,
    int iterations)
{
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        warper.process(frame);
    }
    timer.stop();
    double averageTime = timer.elapsed() / iterations;
    double fps = 1000.0 / averageTime;

    std::cout << "\n========== Warp Benchmark ==========\n";
    std::cout
        << "Resolution : "
        << frame.rgb.cols
        << " x "
        << frame.rgb.rows
        << std::endl;

    std::cout
        << "Views      : "
        << warper.getNumViews()
        << std::endl;

    std::cout
        << "Iterations : "
        << iterations
        << std::endl;

    std::cout
        << "Average    : "
        << averageTime
        << " ms"
        << std::endl;

    std::cout
        << "FPS        : "
        << fps
        << std::endl;
}

std::vector<ViewScaleResult> WarpBenchmark::testViewScale(
    WarpingEngine& warper,
    Frame& frame,
    const std::vector<int>& viewCounts,
    int iterations)
{
    std::vector<ViewScaleResult> results;
    results.reserve(viewCounts.size());

    for (int nv : viewCounts)
    {
        warper.setNumViews(nv);

        Timer timer;
        timer.start();
        for (int i = 0; i < iterations; ++i)
        {
            warper.process(frame);
        }
        timer.stop();

        double avgTime = timer.elapsed() / iterations;
        double fps = 1000.0 / avgTime;

        results.push_back({ nv, avgTime, fps });

        std::cout << "  Views=" << nv
            << "  Avg=" << std::fixed << std::setprecision(2) << avgTime << " ms"
            << "  FPS=" << std::setprecision(1) << fps
            << std::endl;
    }

    return results;
}

void WarpBenchmark::exportViewScaleCSV(
    const std::string& filename,
    const std::vector<std::string>& engineNames,
    const std::vector<std::vector<ViewScaleResult>>& allResults)
{
    std::ofstream out(filename);
    if (!out.is_open())
    {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }

    out << "Engine,Views,AvgTimeMs,FPS\n";

    for (size_t e = 0; e < engineNames.size(); ++e)
    {
        for (const auto& r : allResults[e])
        {
            out << engineNames[e] << ","
                << r.numViews << ","
                << r.avgTimeMs << ","
                << r.fps << "\n";
        }
    }

    std::cout << "\nResults saved to: " << filename << std::endl;
}


bool WarpAnalyzer::testImagesIfEqual(const cv::Mat& img1, const cv::Mat& img2)
{
	if (img1.empty() || img2.empty())
	{
		std::cerr << "One or both images are empty." << std::endl;
		return false;
	}
	if (img1.size() != img2.size() || img1.type() != img2.type())
	{
		std::cerr << "Images have different sizes or types." << std::endl;
		return false;
	}
	cv::Mat diff;
	cv::absdiff(img1, img2, diff);
	return cv::countNonZero(diff.reshape(1)) == 0; // Reshape to single channel for countNonZero
}

void WarpAnalyzer::printSummaryTable(
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