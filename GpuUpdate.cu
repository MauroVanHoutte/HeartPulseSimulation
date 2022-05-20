#include "VertexInput.h"
#include <vector>

#include "GpuUpdate.h"

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void UpdateNodes(VertexInput* vertices, size_t nrOfVerts, float apMinValue, float apMaxValue, float deltaTimeInMs, float deltaTime, float dist)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nrOfVerts)
		return;

	VertexInput& vertex = vertices[i];

	switch (vertex.state)
	{
	case VertexInput::State::APD:
	{
		vertex.timePassed += deltaTimeInMs;

		float timeRatio = vertex.timePassed / vertex.apd;

		float lerpedValue = (apMaxValue * (1 - timeRatio)) + (apMinValue * timeRatio);

		vertex.actionPotential = lerpedValue;
		vertex.apVisualization = 1 - timeRatio;

		if (vertex.timePassed >= vertex.apd)
		{
			vertex.timePassed = 0.f;
			vertex.state = VertexInput::State::DI;
			vertex.apVisualization = 0.f;
		}

		break;
	}
	case VertexInput::State::DI:
		vertex.timePassed += deltaTimeInMs;
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

void CudaUpdate::Update(std::vector<VertexInput>& vertices, float apMinValue, float apMaxValue, float deltaTimeInMs, float deltaTime, float dist)
{
	cudaError_t err = cudaSuccess;
	if (m_DeviceVerts == nullptr)
	{
		err = cudaMalloc((void**)&m_DeviceVerts, vertices.size() * sizeof(VertexInput));
	}
	cudaMemcpy(m_DeviceVerts, vertices.data(), vertices.size() * sizeof(VertexInput), cudaMemcpyHostToDevice);


	int threadsPerBlock{256};
	int numBlocks{ (int(vertices.size()) + threadsPerBlock - 1) / threadsPerBlock };


	UpdateNodes <<<numBlocks, threadsPerBlock>>>(m_DeviceVerts, vertices.size(), apMinValue, apMaxValue, deltaTimeInMs, deltaTime, dist);

	cudaMemcpy(vertices.data(), m_DeviceVerts, vertices.size() * sizeof(VertexInput), cudaMemcpyDeviceToHost);

}
