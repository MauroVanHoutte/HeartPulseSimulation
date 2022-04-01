#pragma once

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>

#include <intrin.h>

//Session and Timer class provided by Tom Tesch
//drag the json file to chrome://tracing to visualize
#define STR(s) #s
#define XSTR(s) STR(s)

#define MEASURE 1
#if MEASURE
#define TIME() Timer Time{__FUNCTION__ "-" XSTR(__LINE__)}
#else
#define TIME()
#endif

class Session
{
public:
    void BeginSession(const std::string& filepath = "measurements.json");
    void EndSession();
    void SaveMeasurement(const std::string& Name, int64_t Start, int64_t End, const std::thread::id& ThreadID);
    static Session& Get() {
        static Session instance;
        return instance;
    }

private:
    std::ofstream m_OutputStream;
    bool m_first;
    Session() : m_first(true) {}

};

class Timer
{
public:
    Timer(const std::string& name);
    ~Timer();
private:
    std::string m_Name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
};

class Logger
{
public:
    template <typename T>
    void LogBuffer(const std::vector<T>& buffer, const std::string& bufferName)
    {
	    if (m_OutputStream.is_open())
	    {
            uint64_t memoryUsage = buffer.size() * sizeof(T);
            m_OutputStream << "Buffer: " << bufferName << "\n";
            m_OutputStream << "Buffer Size: " << buffer.size() << " Objects\n";
            m_OutputStream << "Buffer Object Size: " << sizeof(T) << " Bytes\n";
            m_OutputStream << "Memory used: " <<  memoryUsage << " Bytes\n\n";
	    }
    }

    void LogCPUData();

    void BeginSession(const std::string& filepath = "measurements.txt");
    void EndSession();
    static Logger& Get()
	{
        static Logger instance;
        return instance;
    }
private:
    Logger();
    std::ofstream m_OutputStream;
};
