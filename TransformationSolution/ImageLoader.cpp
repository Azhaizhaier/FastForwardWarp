#include"ImageLoader.h"
#include<iostream>
#include<fstream>
#include<vector>

Frame ImageLoader::loaderFrame(const std::string& rgb_path, const std::string& depth_path)
{
	Frame frame;
	frame.rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);

	std::string extension = depth_path.substr(depth_path.find_last_of(".") + 1);

	if (extension == "pfm") {
		frame.depth = loadPFM(depth_path);
		frame.depthOrig = frame.depth.clone();
	}
	else {
		frame.depth = cv::imread(depth_path, cv::IMREAD_GRAYSCALE);
		frame.depthOrig = frame.depth.clone();
	}

	return frame;
}
cv::Mat ImageLoader::loadPFM(const std::string& filename)
{
	// 这里可以实现PFM文件的加载逻辑
	// PFM是一种高动态范围图像格式，通常用于存储深度图或浮点图像
	// 你可以参考相关文档或库来实现这个功能
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open PFM file: " << filename << std::endl;
		return cv::Mat();
	}

	std::string format;

	file >> format;

	if (format != "Pf") {
		std::cerr << "Unsupported PFM format: " << format << std::endl;
		return cv::Mat();
	}

	int width;
	int height;

	file >> width >> height;

	float scale; // 这个值可以用来判断图像的字节序，正数表示大端序，负数表示小端序
	file >> scale;

	file.ignore(1); // 跳过一个换行符	
	
	cv::Mat disparity(height, width, CV_32FC1);

	file.read(reinterpret_cast<char*>(disparity.data), width * height * sizeof(float));

	cv::flip(disparity, disparity, 0);

	return disparity;
}

//显示视差图，OpenCV
cv::Mat ImageLoader::disparityToDisplay(const cv::Mat& disparity)
{
	cv::Mat display;

	double minVal, maxVal;

	cv::minMaxLoc(disparity, &minVal, &maxVal);

	disparity.convertTo(display, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
	
	return display;
}
