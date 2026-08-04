#pragma once

#include <vector>

#include "InterlacingFilter.h"

struct PixelViewMap
{
    uint8_t bView;

    uint8_t gView;

    uint8_t rView;
};

class LenticularInterlacing :
    public InterlacingFilter
{

public:

    LenticularInterlacing(
        int viewNum);

    void process(Frame& frame) override;

private:

    void buildLUT();

    void resizeViews(Frame& frame);

private:

    int width_;

    int height_;

    int viewNum_;

    std::vector<PixelViewMap> lut_;
};