#pragma once
#include "Frame.h"

class WarpingEngine
{
protected:
    int m_numViews = VIEWS_NUM;

public:
    virtual void process(Frame& frame);
    void processOptimizeByPtr(Frame& frame);

    void setNumViews(int n) { m_numViews = n; }
    int getNumViews() const { return m_numViews; }
};
