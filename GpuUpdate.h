#pragma once
#include <vector>
#include "VertexInput.h"

class CudaUpdate
{
public:
	CudaUpdate() = default;
	~CudaUpdate();

	void Setup(const std::vector<VertexData>& vertices, std::vector<float>& apPlot);

	void PulseVertex(int index, float conductionVelocity, bool useFibres);

	void Update(float apMinValue, float apd, float diastolicInterval, float deltaTimeInMs, float deltaTime, float conductionVelocity, bool useFibres);

	void SetAp(std::vector<VertexData>& vertices);

private:
	VertexData* m_DeviceVerts = nullptr;
	PulseData* m_DevicePulseData = nullptr;
	uint32_t** m_DeviceNeighbors = nullptr;
	uint32_t* m_DeviceNeighborCount = nullptr;
	float* m_DeviceAp = nullptr;
	uint32_t m_VertexCount;
	float* m_DeviceApPlot = nullptr;
	uint32_t m_ApPlotSize;
};