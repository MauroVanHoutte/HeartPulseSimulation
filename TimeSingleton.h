#pragma once

class TimeSingleton
{
public:
	static TimeSingleton* GetInstance();

	float DeltaTime();
	float DeltaTimeInMs();

	void Update(float deltaTime);

	void Destroy();

private:
	TimeSingleton();
	static TimeSingleton* m_Instance;

	float m_DeltaTime{0};
	float m_DeltaTimeInMs{0};
};