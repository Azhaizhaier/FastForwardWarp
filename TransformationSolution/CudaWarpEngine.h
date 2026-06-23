#pragma once

#include "WarpingEngine.h"

class CudaWarpEngine :
    public WarpingEngine
{
public:

    void process(
        Frame& frame) override;
};
