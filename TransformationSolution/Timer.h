#pragma once

#include <chrono>

class Timer
{
private:
	std::chrono::high_resolution_clock::time_point m_start;
	std::chrono::high_resolution_clock::time_point m_end;
public:
	void start();
	void stop();
	//常函数（const member function）是指在函数声明后面加上const修饰符的成员函数。
	//这种函数承诺不会修改对象的状态，即不会修改成员变量的值。
	//常函数只能调用其他常函数，不能调用非常函数，因为非常函数可能会修改对象的状态。
	double elapsed()const;
};