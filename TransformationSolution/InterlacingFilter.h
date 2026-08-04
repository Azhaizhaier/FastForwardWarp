#pragma once
#include "Frame.h"
class InterlacingFilter
{
public:
	virtual ~InterlacingFilter() = default;
	virtual void process(Frame& frame) = 0;
};
