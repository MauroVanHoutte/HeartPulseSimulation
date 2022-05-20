#pragma once
#include <vector>
#include <fstream>

//meant to benchmark performance of a single function over the span of the program
class Benchmarker
{
public:
	static Benchmarker* GetInstance();
	
	void StartBenchmark(float pulseRate, const std::string& name);
	void EndBenchmark();

	void AddDuration(float duration);

	void Destroy();

private:
	Benchmarker();

	static Benchmarker* m_Instance;
	std::vector<float> m_Durations{};
	std::ofstream m_File{};

	int m_ThousandIterations = 1;
};