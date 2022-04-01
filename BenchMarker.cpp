#include "BenchMarker.h"
#include <iomanip>
#include <iostream>
#include <numeric>
#include <time.h>

Benchmarker* Benchmarker::m_Instance = nullptr;

Benchmarker* Benchmarker::GetInstance()
{
	if (m_Instance == nullptr)
		m_Instance = new Benchmarker();
	return m_Instance;
}

Benchmarker::Benchmarker()
{

}

void Benchmarker::AddDuration(float duration)
{
	m_Durations.push_back(duration);

	if (m_Durations.size() > 1000)
	{
		float total = std::accumulate(m_Durations.begin(), m_Durations.end(), 0.f);
		std::cout << "10000 iterations with a total time of: " << total << std::endl;
		std::cout << "Average time per iteration: " << total / m_Durations.size() << std::endl;
		m_Durations.clear();
	}
}

void Benchmarker::Destroy()
{
	delete m_Instance;
}
