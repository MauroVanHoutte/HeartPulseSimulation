#include "VertexInput.h"
#include <vector>

#include "GpuUpdate.h"

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void UpdateNodes(VertexInput* vertices, size_t nrOfVerts, float* apPlot, size_t apPlotSize, float apMinValue, float apd, float diastolicInterval, float deltaTimeInMs, float deltaTime, float dist)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nrOfVerts)
		return;

	VertexInput& vertex = vertices[i];

	switch (vertex.state)
	{
	case VertexInput::State::APD:
		vertex.timePassed += deltaTimeInMs;

		int idx = int(vertex.timePassed);

		if (idx > 0 && idx < apPlotSize && (size_t(idx) + size_t(1)) < apPlotSize)
		{
			float value1 = apPlot[idx];
			float value2 = apPlot[(size_t(idx) + size_t(1))];
			float t = vertex.timePassed - idx;

			float lerpedValue = value1 + t * (value2 - value1);

			float valueRange01 = (lerpedValue - apMinValue) / dist;

			vertex.actionPotential = lerpedValue;
			vertex.apVisualization = valueRange01;
		}

		if (vertex.timePassed >= apd)
		{
			vertex.timePassed = 0.f;
			vertex.state = VertexInput::State::DI;
			vertex.apVisualization = 0.f;
		}

		break;
	case VertexInput::State::DI:
		vertex.timePassed += deltaTimeInMs;

		if (vertex.timePassed >= diastolicInterval)
		{
			vertex.timePassed = 0.f;
			vertex.state = VertexInput::State::Waiting;
		}
		break;
	}

}


CudaUpdate::~CudaUpdate()
{
	if (m_DeviceVerts != nullptr)
		cudaFree(m_DeviceVerts);
	if (m_DeviceApPlot != nullptr)
		cudaFree(m_DeviceApPlot);

}

void CudaUpdate::Update(std::vector<VertexInput>& vertices, std::vector<float>& apPlot, float apMinValue, float apd, float diastolicInterval, float deltaTimeInMs, float deltaTime, float dist)
{
	cudaError_t err = cudaSuccess;
	if (m_DeviceVerts == nullptr)
	{
		err = cudaMalloc((void**)&m_DeviceVerts, vertices.size() * sizeof(VertexInput));
	}
	cudaMemcpy(m_DeviceVerts, vertices.data(), vertices.size() * sizeof(VertexInput), cudaMemcpyHostToDevice);
	if (m_DeviceApPlot == nullptr)
	{
		err = cudaMalloc((void**)&m_DeviceApPlot, apPlot.size() * sizeof(float));
	}
	cudaMemcpy(m_DeviceApPlot, apPlot.data(), apPlot.size() * sizeof(float), cudaMemcpyHostToDevice);


	int threadsPerBlock{256};
	int numBlocks{ (int(vertices.size()) + threadsPerBlock - 1) / threadsPerBlock };


	UpdateNodes <<<numBlocks, threadsPerBlock>>>(m_DeviceVerts, vertices.size(), m_DeviceApPlot, apPlot.size(), apMinValue, apd, diastolicInterval, deltaTimeInMs, deltaTime, dist);

	cudaMemcpy(vertices.data(), m_DeviceVerts, vertices.size() * sizeof(VertexInput), cudaMemcpyDeviceToHost);

}
