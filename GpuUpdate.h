#pragma once
#include <vector>
#include "VertexInput.h"

class CudaUpdate
{
public:
	CudaUpdate() = default;
	~CudaUpdate();

	void Update(std::vector<VertexInput>& vertices, std::vector<float>& apPlot, float apMinValue, float apd, float diastolicInterval, float deltaTimeInMs, float deltaTime, float dist);

private:
	VertexInput* m_DeviceVerts = nullptr;
	float* m_DeviceApPlot = nullptr;
};