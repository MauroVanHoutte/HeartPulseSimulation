#include "ThreadManager.h"

ThreadManager* ThreadManager::m_Instance = nullptr;

void ThreadManager::ThreadLoop()
{
    while (!m_Quit)
    {
        std::packaged_task<void()> job;
        {
            std::unique_lock<std::mutex> lock(m_QueueMutex); // queue is locked, other threads have to wait before accesing 

            m_ConditionVariable.wait(lock, [this]() { return !m_JobQueue.empty() || m_Quit; }); // if queue is empty and app is not quitting, wait for notify

            if (m_Quit)
                return;

            if (m_JobQueue.empty())
                continue;

            job = std::move(m_JobQueue.front()); // job is moved from the queue to local variable
            m_JobQueue.pop_front();
        }

        if (!job.valid())
            return;

        job(); // job is executed
    }
}

ThreadManager* ThreadManager::GetInstance() // singleton
{
    if (m_Instance == nullptr)
        m_Instance = new ThreadManager;
    return m_Instance;
}

void ThreadManager::Destroy()
{
    m_Quit = true;

    m_ConditionVariable.notify_all(); // notify threads of quitting

    for (std::thread& thread : m_Threads)
    {
        thread.join();
    }

    m_Threads.clear();

    delete m_Instance;
}

size_t ThreadManager::GetNrThreads()
{
    return m_Threads.size();
}

ThreadManager::ThreadManager()
{
    int nrThreads = std::thread::hardware_concurrency(); //number of threads available in system
    m_Threads.reserve(nrThreads);
    for (size_t i = 0; i < nrThreads; i++)
    {
        m_Threads.push_back(std::thread(std::bind(&ThreadManager::ThreadLoop, this))); //initiliasi threads with threadloop function
    }
}

