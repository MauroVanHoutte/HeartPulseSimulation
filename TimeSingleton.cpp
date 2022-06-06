#include "TimeSingleton.h"

TimeSingleton* TimeSingleton::m_Instance = nullptr;

TimeSingleton* TimeSingleton::GetInstance()
{
	if (m_Instance == nullptr)
		m_Instance = new TimeSingleton();
	return m_Instance;
}

TimeSingleton::TimeSingleton()
{

}

float TimeSingleton::DeltaTime()
{
	return m_DeltaTime;
}

float TimeSingleton::DeltaTimeInMs()
{
	return m_DeltaTimeInMs;
}

void TimeSingleton::Update(float deltaTime)
{
	m_DeltaTime = deltaTime;
	m_DeltaTimeInMs = deltaTime * 1000;
}

void TimeSingleton::Destroy()
{
	delete m_Instance;
}
