#pragma once
#include "WarpingEngine.h"

class OpenMPDisparityWarpEngine : public WarpingEngine
{
public:
	void process(Frame& frame) override;
};