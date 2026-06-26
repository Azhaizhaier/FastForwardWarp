#include <vector>
#include <string>

#include "Benchmark.h"
#include "SummaryTools.h"

bool SummaryTools::testImagesIfEqual(const cv::Mat& img1, const cv::Mat& img2)
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

void SummaryTools::printSummaryTable(
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