#include "VertexInput.h"
#include <vector>

#include "GpuUpdate.h"

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <glm.hpp>
#include <gtx/norm.hpp>

__device__ float dot(const glm::fvec3& a, const glm::fvec3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float powD(float base, int exp)
{
	if (exp == 0)
		return 0;

	float result = base;
	for (size_t i = 1; i < exp; i++)
	{
		result *= base;
	}
}

__device__ float root(float n)
{
	//https://stackoverflow.com/questions/3581528/how-is-the-square-root-function-implemented

	// Max and min are used to take into account numbers less than 1
	double lo = 1, hi = n, mid;   
	if (n < 1) lo = n, hi = 1;
	// Update the bounds to be off the target by a factor of 10
	while (100 * lo * lo < n) lo *= 10;
	while (0.01 * hi * hi > n) hi *= 0.1;

	for (int i = 0; i < 100; i++) {
		mid = (lo + hi) / 2;
		if (mid * mid == n) return mid;
		if (mid * mid > n) hi = mid;
		else lo = mid;
	}
	return mid;
}

__device__ float sqrDistance(const glm::fvec3& a, const glm::fvec3& b)
{
	return powD(a.x - b.x, 2) + powD(a.y - b.y, 2) + powD(a.z - b.z, 2);
}

__device__ void PulseVertex(VertexData& vertex, VertexData* vertices, uint32_t nrOfVerts, PulseData* pulseData, uint32_t** neighbors, uint32_t* neighborCount, float* apPlot, float conductionVelocity, bool useFibres, int index)
{
	if (vertex.state == State::Waiting || /* (vertex.actionPotential < m_APThreshold && */ vertex.state == State::DI)
	{
		vertex.actionPotential = apPlot[0];
		vertex.state = State::APD;

		for (size_t i = 0; i < neighborCount[index]; i++)
		{
			VertexData& neighbourVertex = vertices[neighbors[index][i]];

			//Potential problem with fibres. c0 is in m/s while the distance is most likely not in meters.
				//This is likely the cause of it.
			float distanceSqrd = sqrDistance(pulseData[index].position, pulseData[neighbors[index][i]].position);

			if (useFibres)
			{
				float d1 = 1; // parallel with fibre
				float d2 = d1 / 5; // perpendiculat with fibre
				float c0 = 0.6f; // m/s

				glm::fvec3 pulseDirection = pulseData[neighbors[index][i]].position - pulseData[index].position;
				float cosAngle = dot(pulseData[index].fibreDirection, pulseDirection);

				float c = c0 * root(d2 + (d1 - d2) * powD(cosAngle, 2));
				conductionVelocity = c * 100;
				//std::cout << c << "\n";
			}

			float travelTime = distanceSqrd / conductionVelocity;

			if (neighbourVertex.state == State::Waiting)
			{
				neighbourVertex.timeToTravel = travelTime;
				neighbourVertex.state = State::Receiving;
			}
		}
	}
}

__global__ void UpdateNodes(VertexData* vertices, uint32_t nrOfVerts, PulseData* pulseData, uint32_t** neighbors, uint32_t* neighborCount, float* ap , float* apPlot, uint32_t apPlotSize, float apMinValue, float apd, float diastolicInterval, float deltaTimeInMs, float deltaTime, float dist, float conductionVelocity, bool useFibres)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nrOfVerts)
		return;

	VertexData& vertex = vertices[i];

	int state = (int)vertex.state;

	switch (state)
	{
	case 2:
		vertex.timePassed += deltaTimeInMs;

		int idx = int(vertex.timePassed);

		if (idx > 0 && idx < apPlotSize && (size_t(idx) + size_t(1)) < apPlotSize)
		{
			float value1 = apPlot[idx];
			float value2 = apPlot[(size_t(idx) + size_t(1))];
			float t = vertex.timePassed - idx;

			float lerpedValue = value1 + t * (value2 - value1);

			float valueRange01 = (lerpedValue - apMinValue) / dist;

			ap[i] = lerpedValue;
		}

		if (vertex.timePassed >= apd)
		{
			vertex.timePassed = 0.f;
			vertex.actionPotential = apMinValue;
			vertex.state = State::DI;
		}

		break;
	case 3:
		vertex.timePassed += deltaTimeInMs;

		if (vertex.timePassed >= diastolicInterval)
		{
			vertex.timePassed = 0.f;
			vertex.state = State::Waiting;
		}
		break;
	case 1:
		vertex.timeToTravel -= deltaTime;
		if (vertex.timeToTravel <= 0.f)
		{
			vertex.state = (State)0;
			PulseVertex(vertex, vertices, nrOfVerts, pulseData, neighbors, neighborCount, apPlot, conductionVelocity, useFibres, i);
		}
	}

}



CudaUpdate::~CudaUpdate()
{
	if (m_DeviceVerts != nullptr)
		cudaFree(m_DeviceVerts);
	if (m_DeviceApPlot != nullptr)
		cudaFree(m_DeviceApPlot);
	if (m_DevicePulseData != nullptr)
		cudaFree(m_DevicePulseData);
	
	cudaDeviceSynchronize();

	cudaFree(m_DeviceNeighbors);

	cudaFree(m_DeviceNeighborCount);
}

void CudaUpdate::Setup(const std::vector<VertexData>& vertices, std::vector<float>& apPlot) 
{

	m_VertexCount = (uint32_t)vertices.size();

	cudaError_t err = cudaSuccess;

	if (m_DeviceVerts == nullptr)
	{
		err = cudaMalloc((void**)&m_DeviceVerts, vertices.size() * sizeof(VertexData));
	}
	cudaMemcpy(m_DeviceVerts, vertices.data(), vertices.size() * sizeof(VertexData), cudaMemcpyHostToDevice);

	if (m_DeviceApPlot == nullptr)
	{
		err = cudaMalloc((void**)&m_DeviceApPlot, apPlot.size() * sizeof(float));
	}
	cudaMemcpy(m_DeviceApPlot, apPlot.data(), apPlot.size() * sizeof(float), cudaMemcpyHostToDevice);
	m_ApPlotSize = apPlot.size();


	std::vector<PulseData> pulseData{};
	pulseData.reserve(vertices.size());
	for (auto& vertex : vertices)
	{
		pulseData.push_back(*vertex.pPulseData);
	}

	if (m_DevicePulseData == nullptr)
	{
		err = cudaMalloc((void**)&m_DevicePulseData, vertices.size() * sizeof(PulseData));
	}
	err = cudaMemcpy(m_DevicePulseData, pulseData.data(), pulseData.size() * sizeof(PulseData), cudaMemcpyHostToDevice);


	std::vector<uint32_t*> neighbors{};
	neighbors.reserve(vertices.size());
	for (auto& vertex : vertices)
	{
		neighbors.push_back(nullptr);
		err = cudaMalloc((void**)&neighbors[neighbors.size() - 1], vertex.pPulseData->pNeighborIndices.size() * sizeof(uint32_t));
		err = cudaMemcpy(neighbors[neighbors.size() - 1], vertex.pPulseData->pNeighborIndices.data(), vertex.pPulseData->pNeighborIndices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
	}
	err = cudaMalloc((void**)&m_DeviceNeighbors, vertices.size() * sizeof(uint32_t*));
	err = cudaMemcpy(m_DeviceNeighbors, neighbors.data(), vertices.size() * sizeof(uint32_t*), cudaMemcpyHostToDevice);

	std::vector<uint32_t> numberOfNeighbors{};
	numberOfNeighbors.reserve(vertices.size());
	for (auto& vertex : vertices)
	{
		numberOfNeighbors.push_back(vertex.pPulseData->neighborIndicesSize);
	}
	err = cudaMalloc((void**)&m_DeviceNeighborCount, vertices.size() * sizeof(uint32_t));
	err = cudaMemcpy(m_DeviceNeighborCount, numberOfNeighbors.data(), vertices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

	err = cudaMalloc((void**)&m_DeviceAp, vertices.size() * sizeof(float));
	err = cudaMemset(m_DeviceAp, 0, vertices.size() * sizeof(float));
}

__global__ void PulseGPUVertex(VertexData* vertices, uint32_t vertexCount, PulseData* pulseData, uint32_t** neighbors, uint32_t* neighborCount, float* apPlot, float conductionVelocity, bool useFibres, int index)
{
	PulseVertex(vertices[index], vertices, vertexCount, pulseData, neighbors, neighborCount, apPlot, conductionVelocity, useFibres, index);
}

void CudaUpdate::PulseVertex(int index, float conductionVelocity, bool useFibres)
{
	PulseGPUVertex <<<1, 1>>>(m_DeviceVerts, m_VertexCount, m_DevicePulseData, m_DeviceNeighbors, m_DeviceNeighborCount, m_DeviceApPlot, conductionVelocity, useFibres, index);
}

void CudaUpdate::Update(float apMinValue, float apd, float diastolicInterval, float deltaTimeInMs, float deltaTime, float dist, float conductionVelocity, bool useFibres)
{

	int threadsPerBlock{1024};
	int numBlocks{ (int(m_VertexCount) + threadsPerBlock - 1) / threadsPerBlock };


	UpdateNodes <<<numBlocks, threadsPerBlock>>>(m_DeviceVerts, m_VertexCount, m_DevicePulseData, m_DeviceNeighbors, m_DeviceNeighborCount, m_DeviceAp, m_DeviceApPlot, m_ApPlotSize, apMinValue, apd, diastolicInterval, deltaTimeInMs, deltaTime, dist, conductionVelocity, useFibres);

}

void CudaUpdate::SetAp(std::vector<VertexData>& vertices)
{
	std::vector<float> ap{};
	ap.resize(m_VertexCount, 0);
	cudaMemcpy(ap.data(), m_DeviceAp, m_VertexCount * sizeof(float), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < m_VertexCount; i++)
	{
		vertices[i].actionPotential = ap[i];
	}
}
