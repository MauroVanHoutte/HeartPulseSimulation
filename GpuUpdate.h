#pragma once
#include <vector>
#include "VertexInput.h"

class CudaUpdate
{
public:
	CudaUpdate() = default;
	~CudaUpdate();

	void Update(std::vector<VertexData>& vertices, std::vector<float>& apPlot, float apMinValue, float apd, float diastolicInterval, float deltaTimeInMs, float deltaTime, float dist);

private:
	VertexData* m_DeviceVerts = nullptr;
	float* m_DeviceApPlot = nullptr;
};