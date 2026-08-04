#pragma once
#include <opencv2/opencv.hpp>

struct Frame
{
	cv::Mat rgb; //RGB
	cv::Mat depth; // signed disparity (after DepthPreprocessor)
	cv::Mat depthOrig; // original non-negative disparity (for z-buffer)
	//cv::Mat disparity;//视差

	std::vector<cv::Mat> warpedViews; //Warped RGBs
	std::vector<cv::Mat> holeMasks; //Hole Mask

	cv::Mat interlacedImage;   // 显示
    cv::Mat interlacedHoleMask; 
};

constexpr auto MAX_DEPTH = 255; // 最大深度值，单位灰度级
constexpr auto MAX_OFFSET = 10; // 最大视差偏移，单位像素
constexpr auto VIEWS_NUM = 5; // 生成的视图数量
constexpr auto CENTER_VIEW = VIEWS_NUM / 2; // 中间视图索引
constexpr auto DISPARITY_SCALE = MAX_OFFSET / static_cast<float>(MAX_DEPTH); // 深度到视差的缩放因子


constexpr auto DISPARITY_GAIN = 0.3f; // 视图缩放因子，保持原始分辨率

constexpr int MAX_VIEWS = 32; //最多视图数
