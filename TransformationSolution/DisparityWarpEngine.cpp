#include "DisparityWarpEngine.h"

void DisparityWarpEngine::process(Frame& frame)
{	
	const int width = frame.rgb.cols;
	const int height = frame.rgb.rows;

	frame.warpedViews.resize(this->m_numViews);
	
	frame.holeMasks.resize(this->m_numViews);

	for (int v = 0; v < this->m_numViews; v++) {
		cv::Mat warped(
			height,
			width,
			CV_8UC3,
			cv::Scalar(0, 0, 0)); // 初始化为全黑

		cv::Mat mask(
			height,
			width,
			CV_8UC1,
			cv::Scalar(255)); // 初始化为全白，表示没有hole

		cv::Mat zBuffer(
			height,
			width,
			CV_32FC1,
			cv::Scalar(0)); // 初始化为全0，表示最远的深度
		const int viewOffset = v - this->m_numViews / 2;


		for (int y = 0; y < height; y++) {

			const cv::Vec3b* rgbRow = frame.rgb.ptr<cv::Vec3b>(y);

			const float* dispRow = frame.depth.ptr<float>(y);

			cv::Vec3b* warpedRow = warped.ptr<cv::Vec3b>(y);

			float* zRow = zBuffer.ptr<float>(y);

			for (int x = 0; x < width; x++) {

				float disparity = dispRow[x];

				if (std::isinf(disparity) ) {
					continue; // 跳过无效的视差值
				}

				int shift = static_cast<int>(disparity * DISPARITY_GAIN * viewOffset);

				int newX = x - shift; //???为啥是减去shift？因为视差越大，物体越近，应该向左移动（对于右视图）或者向右移动（对于左视图）

				if (newX < 0 || newX >= width) {
					continue; // 跳出边界
				}

				if (disparity > zRow[newX]) { // 因为视差值越大，物体越近，所以应该使用较小的视差值来更新zBuffer

					warpedRow[newX] = rgbRow[x]; // 将RGB值写入新的位置

					zRow[newX] = disparity; // 更新zBuffer

					mask.at<uchar>(y, newX) = 0; // 标记为非hole
				}
			}
		}

		frame.warpedViews[v] = warped;
		frame.holeMasks[v] = mask;
	}
	std::cout << "Multi-view warping completed - herited override" << std::endl;
}	
