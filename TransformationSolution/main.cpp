#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "ImageLoader.h"
#include "WarpingEngine.h"
#include "DisparityWarpEngine.h"
#include "OpenMPDisparityWarpEngine.h"
#include "CudaWarpEngine.h"
#include "Benchmark.h"
#include "SummaryTools.h"


int main()
{

    //读入两张图片看看是否完全相等
	cv::Mat img1 = cv::imread("C:\\Users\\admin\\Desktop\\data\\CPU vs OpenMP vs GPU for view\\CPU\\hole_mask_4.png", cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread("C:\\Users\\admin\\Desktop\\data\\CPU vs OpenMP vs GPU for view\\OpenMP\\hole_mask_4.png", cv::IMREAD_COLOR);

	SummaryTools::testImagesIfEqual(img1, img2) ? std::cout << "Images are equal." << std::endl : std::cout << "Images are NOT equal." << std::endl;

  /* ImageLoader loader;
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
    std::vector<std::vector<ViewScaleResult>> allResults;*/





    // 1. DispairtyEngine (base class, handles both 8U and 32F depth)
    /*{
        DisparityWarpEngine engine;
        std::cout << "\n--- WarpingEngine (Base, 1 thread) ---" << std::endl;
        auto results = Benchmark::testViewScale(engine, frame, viewCounts, iterations);
        engineNames.push_back("WarpingEngine");
        allResults.push_back(results);
    }*/

    // 2. OpenMPDisparityWarpEngine
   /* {
        OpenMPDisparityWarpEngine engine;
        std::cout << "\n--- OpenMPDisparityWarpEngine ---" << std::endl;
        auto results = Benchmark::testViewScale(engine, frame, viewCounts, iterations);
        engineNames.push_back("OpenMP");
        allResults.push_back(results);
    }*/

    // 3. CudaWarpEngine
    /*{
        CudaWarpEngine engine;
        std::cout << "\n--- CudaWarpEngine ---" << std::endl;
        auto results = Benchmark::testViewScale(engine, frame, viewCounts, iterations);
        engineNames.push_back("CUDA");
        allResults.push_back(results);
    }*/

    // Print summary table
    //SummaryTools::printSummaryTable(engineNames, allResults);

    // Export CSV
    //Benchmark::exportViewScaleCSV("benchmark_viewscale.csv", engineNames, allResults);

 //   WarpingEngine engine;
	//engine.processOptimizeByPtr(frame);

	////将warp后的图像&空洞掩码保存到本地
	//for (int i = 0; i < frame.warpedViews.size(); i++)
	//{
	//	std::string filename = "warped_view_" + std::to_string(i) + ".png";
	//	std::string filename_mask = "hole_mask_" + std::to_string(i) + ".png";
	//	cv::imwrite(filename, frame.warpedViews[i]);
	//	cv::imwrite(filename_mask, frame.holeMasks[i]);
	//}
    // Show a sample result image
    /*std::cout << "\nPress any key to close the preview window." << std::endl;
    cv::imshow("CUDA Warp - Center View", frame.warpedViews[29]);
    cv::waitKey(0);*/

    return 0;
}
