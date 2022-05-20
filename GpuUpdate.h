#pragma once
#include <vector>
#include "VertexInput.h"

class CudaUpdate
{
public:
	CudaUpdate() = default;
	~CudaUpdate();

	void Update(std::vector<VertexInput>& vertices, float apMinValue, float apMaxValue, float deltaTimeInMs, float deltaTime, float dist);

private:
	VertexInput* m_DeviceVerts = nullptr;
	float* m_DeviceApPlot = nullptr;
};