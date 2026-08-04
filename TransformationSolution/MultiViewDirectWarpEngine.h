#pragma once

#include "WarpingEngine.h"
class MultiViewDirectWarpEngine
	: public WarpingEngine
{
public:
	void process(Frame& frame) override;
};
