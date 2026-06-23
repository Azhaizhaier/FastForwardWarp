#include "Timer.h"
void Timer::start() {
	m_start = std::chrono::high_resolution_clock::now();
}
void Timer::stop() {
	m_end = std::chrono::high_resolution_clock::now();
}
double Timer::elapsed() const {
	return std::chrono::duration<double, std::milli>(m_end - m_start).count();
}