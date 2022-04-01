#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>

//Singleton
class ThreadManager
{
public:
	static ThreadManager* GetInstance();

	void AddJobFunction(const std::function<void()>& job);

	void Destroy();

	size_t GetNrThreads();

private:

	ThreadManager();

	void ThreadLoop();

	static ThreadManager* m_Instance;
	std::vector<std::thread> m_Threads;
	std::queue<std::function<void()>> m_JobQueue;

	std::mutex m_QueueMutex;
	std::condition_variable m_ConditionVariable;

	std::atomic<bool> m_Quit = false;
};