#pragma once
#include <vector>

//meant to benchmark performance of a single function over the span of the program
class Benchmarker
{
public:
	static Benchmarker* GetInstance();

	void AddDuration(float duration);

	void Destroy();

private:
	Benchmarker();

	static Benchmarker* m_Instance;
	std::vector<float> m_Durations{};
};