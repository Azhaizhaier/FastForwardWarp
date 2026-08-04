#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "ImageLoader.h"
#include "WarpingEngine.h"
#include "DisparityWarpEngine.h"
#include "OpenMPDisparityWarpEngine.h"
#include "MultiViewDirectWarpEngine.h"
#include "CudaWarpEngine.h"
#include "LenticularInterlacing.h"
#include "WarpAnalyzer.h"
#include "HoleAnalyzer.h"
#include "DepthPreprocessor.h"
#include "InverseInterlacing.h"
#include "DisplayConfig.h"

int main()
{

   

    //ImageLoader loader;
    //std::string rgb_path = "C:\\Users\\admin\\Desktop\\data\\img\\img0.png";
    //std::string depth_path = "C:\\Users\\admin\\Desktop\\data\\depth\\disp0.pfm";

    //Frame frame = loader.loaderFrame(rgb_path, depth_path);
    //if (frame.rgb.empty() || frame.depth.empty())
    //{
    //    std::cerr << "Error loading images." << std::endl;
    //    return -1;
    //}

    ////将深度改为有符号深度，并定义焦平面
    //DisparityConfig disparityConfig;

    //DepthPreprocessor dptPreprocessor(disparityConfig);
    //auto signed_depth = dptPreprocessor.process(frame.depth);

    //frame.depth = signed_depth.disparity;


    //std::cout << "Image size: "
    //          << frame.rgb.cols << " x " << frame.rgb.rows << "\n"
    //          << "Depth type: "
    //          << (frame.depth.type() == CV_32FC1 ? "CV_32FC1 (float)" :
    //              frame.depth.type() == CV_8UC1  ? "CV_8UC1" : "other")
    //          << std::endl;

    /*const std::vector<int> viewCounts = {2};
    const int iterations = 10;

    std::vector<std::string> engineNames;
    std::vector<std::vector<ViewScaleResult>> allResults;*/



    //// 1. DispairtyEngine (base class, handles both 8U and 32F depth)
    //{
    //    DisparityWarpEngine engine;
    //    std::cout << "\n--- WarpingEngine (Base, 1 thread) ---" << std::endl;
    //    auto results = WarpBenchmark::testViewScale(engine, frame, viewCounts, iterations);
    //    engineNames.push_back("WarpingEngine");
    //    allResults.push_back(results);
    //}

    // 2. OpenMPDisparityWarpEngine
    /*{
        OpenMPDisparityWarpEngine engine;
        std::cout << "\n--- OpenMPDisparityWarpEngine ---" << std::endl;
        auto results = WarpBenchmark::testViewScale(engine, frame, viewCounts, iterations);
        engineNames.push_back("OpenMP");
        allResults.push_back(results);
    }*/

    //// 3. CudaWarpEngine
    //{
    //    CudaWarpEngine engine;
    //    std::cout << "\n--- CudaWarpEngine ---" << std::endl;
    //    auto results = WarpBenchmark::testViewScale(engine, frame, viewCounts, iterations);
    //    engineNames.push_back("CUDA");
    //    allResults.push_back(results);
    //}

    //// 4.  MultiViewDirectWarpEngine 
    //{
    //    MultiViewDirectWarpEngine engine;
    //    std::cout << "\n--- MultiViewDirectWarpEngine ---" << std::endl;
    //    auto results = WarpBenchmark::testViewScale(engine, frame, viewCounts, iterations);
    //    engineNames.push_back("CUDA");
    //    allResults.push_back(results);
    //}

    // Print summary table
    /*WarpAnalyzer::printSummaryTable(engineNames, allResults);*/

    // Export CSV
    //WarpBenchmark::exportViewScaleCSV("benchmark_viewscale.csv", engineNames, allResults);
    

    //OpenMPDisparityWarpEngine engine;
   // engine.setNumViews(30);

	//engine.process(frame);

   

	//LenticularInterlacing interlacer(VIEWS_NUM);

    // Run interlacing first so interlacedImage + interlacedHoleMask are filled
    //interlacer.process(frame);


    //HoleStatistics interlaced_hole_result; //单交织图分析
    //HoleStatistics warped_hole_one_result; //总warp分析
    //std::vector < HoleStatistics > warped_hole_results; //warped holes分析vector

    //interlaced_hole_result =  HoleAnalyzer::analyze(frame.interlacedHoleMask);

    //warped_hole_results = HoleAnalyzer::analyzeViews(frame.holeMasks);

    //warped_hole_one_result = HoleAnalyzer::analyzeOverall(frame.holeMasks);


    ////HoleAnalyzer::print(warped_hole_results);

    //std::cout << std::endl;
    //std::cout << "=================总计分析所有hole masks==================" << std::endl;
    //HoleAnalyzer::print(warped_hole_one_result);

    //std::cout << std::endl;
    //std::cout << "=================interlace hole masks==================" << std::endl;
    //HoleAnalyzer::print(interlaced_hole_result);



	/*interlacer.process(frame);
    cv::imshow("interlaced_image_0", frame.interlacedImage);
    cv::waitKey(0);*/

    /*std::string filename = "scence2_interlaced_image_5_focus_median.png";
    std::string holemask_filename = "scence2_interlaced_image_hole_mask_5_foucus_median.png";
    cv::imwrite(filename, frame.interlacedImage);
    cv::imwrite(holemask_filename, frame.interlacedHoleMask);*/
	//将warp后的图像&空洞掩码保存到本地
	/*for (int i = 0; i < frame.warpedViews.size(); i++)
	{
		std::string filename = "scence2_warped_view_" + std::to_string(i) + ".png";
		std::string filename_mask = "scence2_hole_mask_" + std::to_string(i) + ".png";
		cv::imwrite(filename, frame.warpedViews[i]);
		cv::imwrite(filename_mask, frame.holeMasks[i]);
	}*/

 //   InverseInterlacing inv(
 //       DISPLAY_WIDTH_4K, DISPLAY_HEIGHT_4K,
 //       DISPLAY_WIDTH_4K, DISPLAY_HEIGHT_4K,                  // viewWidth, viewHeight (仅存储，恢复用 4K)
 //       VIEWS_NUM,
 //       THETA, SUBPIXEL, KOFF);

 //   auto views = inv.recoverViews(frame.interlacedImage);

 //   for (int i = 0; i < views.size(); i++)
 //   {
 //       std::string filename = "inverse_view_" + std::to_string(i) + ".png";
 //       //std::string filename_mask = "hole_mask_" + std::to_string(i) + ".png";
 //       cv::imwrite(filename, views[i]);
 //       //cv::imwrite(filename_mask, frame.holeMasks[i]);
 //   }


 //   for (int i = 0; i < views.size(); i++)
 //   {
 //       cv::Mat img1 = frame.warpedViews[i];
 //       cv::Mat img2 = views[i];
 //       WarpAnalyzer::testImagesIfEqual(img1, img2) ? std::cout << "Images are equal." << std::endl : std::cout << "Images are NOT equal." << std::endl;
 //   }

   


    // views[v] = 3840x2160 CV_8UC3, 第 v 个视图
 //   //读入两张图片看看是否完全相等
 //  cv::Mat img1 = cv::imread("C:\\Users\\admin\\Desktop\\data\\CPU vs OpenMP vs GPU for view\\CPU\\hole_mask_0.png", cv::IMREAD_COLOR);
 //  cv::Mat img2 = cv::imread("D:\\source_code\\TransformationSolution\\TransformationSolution\\TransformationSolution\\hole_mask_0.png", cv::IMREAD_COLOR);

 //  WarpAnalyzer::testImagesIfEqual(img1, img2) ? std::cout << "Images are equal." << std::endl : std::cout << "Images are NOT equal." << std::endl;

    // Show a sample result image
    /*std::cout << "\nPress any key to close the preview window." << std::endl;
    cv::imshow("CUDA Warp - Center View", frame.warpedViews[29]);
    cv::waitKey(0);*/


    Frame f;
    f.warpedViews.resize(5);
    std::vector<cv::Mat> views(7);

    for (int i = 0; i < 5; i++) {
        //f.warpedViews[i] = cv::imread("C:\\Users\\admin\\Desktop\\data\\Art-7views\\Art\\view"+ std::to_string(i) +".png", cv::IMREAD_COLOR);
        //f.warpedViews[i] = cv::imread("D:\\source_code\\TransformationSolution\\TransformationSolution\\TransformationSolution\\warped_view_" + std::to_string(i) + ".png", cv::IMREAD_COLOR);
        f.warpedViews[i] = cv::imread("D:\\source_code\\simpleLama\\src\\view" + std::to_string(i)+"_filled_Lama.png", cv::IMREAD_COLOR);
    }
   
    LenticularInterlacing interlacer(5);
    interlacer.process(f);

    std::string filename = "view_filled_interlaced_Lama.png";
   
    cv::imwrite(filename, f.interlacedImage);
  
   // 
   //    InverseInterlacing inv(
   //    DISPLAY_WIDTH_4K, DISPLAY_HEIGHT_4K,
   //    DISPLAY_WIDTH_4K, DISPLAY_HEIGHT_4K,                  // viewWidth, viewHeight (仅存储，恢复用 4K)
   //    5,
   //    THETA, SUBPIXEL, KOFF);

   // auto views = inv.recoverViews(f.interlacedImage);

   //    for (int i = 0; i < views.size(); i++)
   // {
   //     std::string filename = "3_inverse_view_" + std::to_string(i) + ".png";
   //     //std::string filename_mask = "hole_mask_" + std::to_string(i) + ".png";
   //     cv::imwrite(filename, views[i]);
   //     //cv::imwrite(filename_mask, frame.holeMasks[i]);
   // }

   //    for (int i = 0; i < views.size(); i++)
   //{
   //    cv::Mat img1 = f.warpedViews[i];
   //    cv::Mat img2 = views[i];
   //    WarpAnalyzer::testImagesIfEqual(img1, img2) ? std::cout << "Images are equal." << std::endl : std::cout << "Images are NOT equal." << std::endl;
   //}
    return 0;
}

