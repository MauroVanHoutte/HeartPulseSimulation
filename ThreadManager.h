#pragma once
#include <vector>
#include <thread>
#include <deque>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>

//Singleton
class ThreadManager
{
public:
	static ThreadManager* GetInstance();

	template<class F, class R = std::result_of_t<F& ()>>
	std::future<R> AddJobFunction(F&& f)
	{
		std::packaged_task<R()> pt(std::forward<F>(f)); //allows returning a future
		auto r = pt.get_future(); //return value

		{ //scope to destroy lock
			std::unique_lock<std::mutex> lock(m_QueueMutex);
			m_JobQueue.emplace_back(std::move(pt));
		}

		m_ConditionVariable.notify_one(); //notify thread that a task is added

		return r; // future of the task is returned and can be used to wait for completion of the task
	}

	void Destroy();

	size_t GetNrThreads();

private:

	ThreadManager();

	void ThreadLoop();

	static ThreadManager* m_Instance;
	std::vector<std::thread> m_Threads;
	std::deque<std::packaged_task<void()>> m_JobQueue;

	std::mutex m_QueueMutex;
	std::condition_variable m_ConditionVariable;

	std::atomic<bool> m_Quit = false;
};