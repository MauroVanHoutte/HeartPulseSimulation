#pragma once
#include "Time.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <windows.h>

#include <psapi.h>
#include "WbemCli.h"
#include <comutil.h>

#define _WIN32_DCOM
#include <comdef.h>
#include <Wbemidl.h>
# pragma comment(lib, "wbemuuid.lib")

void Session::BeginSession(const std::string& filepath) {
    m_OutputStream.open(filepath);
    m_OutputStream << "{\"traceEvents\":[";
    m_OutputStream.flush();
}

void Session::EndSession() {
    m_OutputStream << "]}";
    m_OutputStream.flush();
    m_OutputStream.close();
    m_first = true;
}

void Session::SaveMeasurement(const std::string& Name, int64_t Start, int64_t End, const std::thread::id& ThreadID) {
    if (m_first)
        m_first = false;
    else
        m_OutputStream << ",";

    m_OutputStream << "{" << "\"cat\":\"function\",\"dur\":" << (End - Start) << ",\"name\":\"" << Name << "\",";
    m_OutputStream << "\"ph\":\"X\",\"pid\":0," << "\"tid\":" << ThreadID << "," << "\"ts\":" << Start << "}";

    m_OutputStream.flush();
}

Timer::Timer(const std::string& name)
	: m_Name(name)
{
    m_StartTimepoint = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    auto endTimepoint = std::chrono::high_resolution_clock::now();

    int64_t begin = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
    int64_t end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

    Session::Get().SaveMeasurement(m_Name, begin, end, std::this_thread::get_id());
}

void Logger::LogCPUData()
{
    if (m_OutputStream.is_open())
    {
        int CPUInfo[4] = { -1 };
        unsigned   nExIds, i = 0;
        char CPUBrandString[0x40];
        // Get the information associated with each extended ID.
        __cpuid(CPUInfo, 0x80000000);
        nExIds = CPUInfo[0];
        for (i = 0x80000000; i <= nExIds; ++i)
        {
            __cpuid(CPUInfo, i);
            // Interpret CPU brand string
            if (i == 0x80000002)
                memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
            else if (i == 0x80000003)
                memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
            else if (i == 0x80000004)
                memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
        }
        //string includes manufacturer, model and clockspeed
        m_OutputStream << "CPU Type: " << CPUBrandString << "\n";
    }
}

//void Logger::LogProgramMemory()
//{
//    if (m_OutputStream.is_open())
//    {
//        ////https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c
//        //double vm_usage = 0.0;
//        //double resident_set = 0.0;
//
//        //std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);
//
//        //std::string pid, comm, state, ppid, pgrp, session, tty_nr;
//        //std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
//        //std::string utime, stime, cutime, cstime, priority, nice;
//        //std::string O, itrealvalue, starttime;
//
//        //unsigned long vsize;
//        //long rss;
//
//        //stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
//        //    >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
//        //    >> utime >> stime >> cutime >> cstime >> priority >> nice
//        //    >> O >> itrealvalue >> starttime >> vsize >> rss;
//
//        //stat_stream.close();
//
//        //long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
//        //vm_usage = vsize / 1024.0;
//        //resident_set = rss * page_size_kb;
//    }
//}

void Logger::BeginSession(const std::string& filepath)
{
    m_OutputStream.open(filepath);
}

void Logger::EndSession()
{
    m_OutputStream.close();
}

Logger::Logger()
{
}
