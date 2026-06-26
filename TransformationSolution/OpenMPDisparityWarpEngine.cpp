#include "OpenMPDisparityWarpEngine.h"

#include <omp.h>


void OpenMPDisparityWarpEngine::process(Frame& frame)
{
	const int width = frame.rgb.cols;

	const int height = frame.rgb.rows;

	frame.warpedViews.resize(this->m_numViews); // 预先分配空间，避免在并行循环中动态扩展vector导致的线程安全问题!!!!!!!!!!!!!!!

	frame.holeMasks.resize(this->m_numViews);

#pragma omp parallel for
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
			cv::Scalar(255)); // 初始化为全白，全为空洞

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

			uchar* maskRow = mask.ptr<uchar>(y);

			for (int x = 0; x < width; x++) {

				float disparity = dispRow[x];

				if (std::isinf(disparity)) {
					continue; // 跳过无效的视差值
				}

				int shift = static_cast<int>(disparity * DISPARITY_GAIN * viewOffset);


				int newX = x + shift;

				if (newX < 0 || newX >= width) {
					continue; // 跳出边界
				}

				if (disparity > zRow[newX]) { // 因为视差值越大，物体越近，所以应该使用较小的视差值来更新zBuffer
					zRow[newX] = disparity;
					warpedRow[newX] = rgbRow[x];
					maskRow[newX] = 0; // 不是空洞
				}
			}
		}
		frame.warpedViews[v] = warped;
		frame.holeMasks[v] = mask;

	}

}
