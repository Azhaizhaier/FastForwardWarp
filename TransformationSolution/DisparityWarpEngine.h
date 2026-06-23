#pragma once
#include "Frame.h"
#include "WarpingEngine.h"

class DisparityWarpEngine : public WarpingEngine
{
public:
	void process(Frame& frame) override;
};