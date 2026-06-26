#pragma once
#include <vector>
#include<string>

#include "Benchmark.h"


class SummaryTools
{
public:
    static void printSummaryTable(
        const std::vector<std::string>& engineNames,
        const std::vector<std::vector<ViewScaleResult>>& allResults);
    //判断两张图片是不是完全相等
	static bool testImagesIfEqual(const cv::Mat& img1, const cv::Mat& img2);
};
