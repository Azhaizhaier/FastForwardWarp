#include "Benchmark.h"
#include "Timer.h"
#include <iostream>
#include <fstream>
#include <iomanip>

void Benchmark::testWarp(
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

std::vector<ViewScaleResult> Benchmark::testViewScale(
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

void Benchmark::exportViewScaleCSV(
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
