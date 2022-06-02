#include "BenchMarker.h"
#include <iomanip>
#include <iostream>
#include <numeric>
#include <time.h>
#include <sstream>

Benchmarker* Benchmarker::m_Instance = nullptr;

Benchmarker* Benchmarker::GetInstance()
{
	if (m_Instance == nullptr)
		m_Instance = new Benchmarker();
	return m_Instance;
}

void Benchmarker::StartBenchmark(float pulseRate, const std::string& name)
{
	//https://stackoverflow.com/questions/16357999/current-date-and-time-as-string/16358264
	time_t t = std::time(nullptr);
	tm tm{};
	localtime_s(&tm, &t);

	std::ostringstream oss;
	oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
	auto date = oss.str();
	std::string path{ "Resources/Output/" };
	m_File.open(path + "Benchmark-" + name + "-" + date + ".txt");
	m_File << "Pulse rate: " << pulseRate << "Hz\n";
}

void Benchmarker::EndBenchmark()
{
	float totalTime{ std::accumulate(m_Durations.begin(), m_Durations.end(), 0.f) };
	m_File << "Total iterations: " << m_Durations.size() << std::endl;
	m_File << "Taking a total of: " << totalTime << " Seconds" << std::endl;
	m_File << "Average time per execution: " << totalTime / m_Durations.size() << std::endl;
	m_File.close();
	m_Durations.clear();
	m_ThousansIterations = 1;
}

Benchmarker::Benchmarker()
{

}

void Benchmarker::AddDuration(float duration)
{
	m_Durations.push_back(duration);

	if (m_Durations.size() > 100 * m_ThousansIterations)
	{
		float total = std::accumulate(m_Durations.begin(), m_Durations.end(), 0.f);
		std::cout << 100 * m_ThousansIterations << " iterations with a total time of : " << total << std::endl;
		std::cout << "Average time per iteration: " << total / m_Durations.size() << std::endl;
		m_ThousansIterations++;
	}
}

void Benchmarker::Destroy()
{
	delete m_Instance;
}
