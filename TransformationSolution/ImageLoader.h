#pragma once
#include<iostream>
#include<string>
#include "frame.h"
class ImageLoader
{
public:
	Frame loaderFrame(const std::string& rgb_path, const std::string& depth_path);

private:
	cv::Mat loadPFM(const std::string& filename);
	cv::Mat disparityToDisplay(const cv::Mat& disprity);

};
