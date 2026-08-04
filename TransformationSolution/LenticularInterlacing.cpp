#include "LenticularInterlacing.h"
#include "DisplayConfig.h"

#include <cmath>

//Lenticular Mask是固定的，将mask生成函数放到构造函数当中
//整个工程生命周期只生成一次
void LenticularInterlacing::
buildLUT()
{
    lut_.resize(width_ * height_);

    for (int y = 0; y < height_; y++)
    {
        for (int x = 0; x < width_; x++)
        {
            PixelViewMap& m =
                lut_[y * width_ + x];

            for (int c = 0; c < 3; c++)
            {
                int subpixelX =
                    x * 3 + (2 - c);

                float r =
                    std::fmod(
                        subpixelX
                        + KOFF
                        - 3.0f * y * THETA,
                        SUBPIXEL);

                if (r < 0)
                    r += SUBPIXEL;

                int view =
                    static_cast<int>(
                        r *
                        viewNum_
                        / SUBPIXEL);

                if (view >= viewNum_)
                    view =
                    viewNum_ - 1;

                switch (c)
                {
                case 0:

                    m.rView = view;

                    break;

                case 1:

                    m.gView = view;

                    break;

                case 2:

                    m.bView = view;

                    break;
                }
            }
        }
    }
}
//构造函数
LenticularInterlacing::
LenticularInterlacing(
    int viewNum)
    :
    width_(DISPLAY_WIDTH_4K),
    height_(DISPLAY_HEIGHT_4K),
    viewNum_(viewNum)
{
    buildLUT();
}
//缩放视图适配与显示分辨率
void LenticularInterlacing::resizeViews(Frame& frame)
{
    cv::Size displaySize(
        DISPLAY_WIDTH_4K,
        DISPLAY_HEIGHT_4K);

    for (auto& view : frame.warpedViews)
    {
        if (view.size() == displaySize)
            continue;

        cv::Mat resized;

        cv::resize(
            view,
            resized,
            displaySize,
            0,
            0,
            cv::INTER_CUBIC);

        view = std::move(resized);
    }

    // 如果后续 Hole Mask 也需要参与显示，
    // 建议一起缩放
    for (auto& mask : frame.holeMasks)
    {
        if (mask.empty())
            continue;

        if (mask.size() == displaySize)
            continue;

        cv::Mat resized;

        cv::resize(
            mask,
            resized,
            displaySize,
            0,
            0,
            cv::INTER_NEAREST);

        mask = std::move(resized);
    }
}

void LenticularInterlacing::process(Frame& frame)
{
    //--------------------------------------------
    // Step 1. Resize
    //--------------------------------------------
    bool needResize =
        frame.warpedViews[0].cols != DISPLAY_WIDTH_4K ||
        frame.warpedViews[0].rows != DISPLAY_HEIGHT_4K;

    if (needResize)
    {
        resizeViews(frame);
    }

    //--------------------------------------------
    // Step 2. Allocate output
    //--------------------------------------------
    frame.interlacedImage =
        cv::Mat::zeros(
            DISPLAY_HEIGHT_4K,
            DISPLAY_WIDTH_4K,
            CV_8UC3);

    frame.interlacedHoleMask =
        cv::Mat::zeros(
            DISPLAY_HEIGHT_4K,
            DISPLAY_WIDTH_4K,
            CV_8UC1);

    //--------------------------------------------
    // Cache row pointers
    //--------------------------------------------


    const cv::Vec3b* rgbRows[MAX_VIEWS];
    //const uchar* maskRows[MAX_VIEWS];

    //--------------------------------------------
    // Step 3. Interlace
    //--------------------------------------------
    for (int y = 0; y < DISPLAY_HEIGHT_4K; y++)
    {
        //-----------------------------
        // Cache every view row
        //-----------------------------
        for (int v = 0; v < viewNum_; v++)
        {
            rgbRows[v] =
                frame.warpedViews[v].ptr<cv::Vec3b>(y);

            //maskRows[v] =
                //frame.holeMasks[v].ptr<uchar>(y);
        }

        //-----------------------------
        // Destination
        //-----------------------------
        cv::Vec3b* dstRGB =
            frame.interlacedImage.ptr<cv::Vec3b>(y);

        //uchar* dstMask =
            //frame.interlacedHoleMask.ptr<uchar>(y);

        //-----------------------------
        // Current LUT Row
        //-----------------------------
        const PixelViewMap* lutRow =
            &lut_[y * DISPLAY_WIDTH_4K];

        //-----------------------------
        // Pixel loop
        //-----------------------------
        for (int x = 0; x < DISPLAY_WIDTH_4K; x++)
        {
            const PixelViewMap& m =
                lutRow[x];

            //---------------- RGB ----------------
            cv::Vec3b pixel;

            pixel[0] =
                rgbRows[m.bView][x][0];

            pixel[1] =
                rgbRows[m.gView][x][1];

            pixel[2] =
                rgbRows[m.rView][x][2];

            dstRGB[x] = pixel;

            //---------------- Hole Mask ----------------
            /*三个Subpixel里面：

                有一个Hole。

                整个Pixel就是Hole*/
            /*dstMask[x] =
                (maskRows[m.bView][x] |
                    maskRows[m.gView][x] |
                    maskRows[m.rView][x])
                ? 255
                : 0;*/
        }
    }
}