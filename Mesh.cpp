#include "Mesh.h"

#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>

#include "Time.h"
#include <unordered_map>
//glm/gtc/epsilon.hpp

#include "ThreadManager.h"
#include "BenchMarker.h"
#include "GpuUpdate.h"
#include "TimeSingleton.h"

//External Headers
#pragma warning(push)
#pragma warning(disable:4244)
#pragma warning(disable:4701)
#pragma warning(disable:4003)
//#define OBJL_CONSOLE_OUTPUT
#undef min
#undef max
#include "3rdParty/glm/gtx/norm.hpp"
#include "OBJ_Loader.h"
#include <cuda_runtime.h>
#undef OBJL_CONSOLE_OUTPUT
#pragma warning(pop)

using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

Mesh::Mesh()
	: m_pEffect{}
	, m_pOptimizerEffect{}
	, m_pVertexLayout{ nullptr }
	, m_pVertexBuffer{ nullptr }
	, m_pIndexBuffer{ nullptr }
	, m_pRasterizerStateWireframe{ nullptr }
	, m_pRasterizerStateSolid{ nullptr }
	, m_WireFrameEnabled{false}
	, m_DrawVertex(false)
	, m_AmountIndices{}
	, m_FibresLoaded{false}
	, m_WorldMatrix{ glm::mat4{1.f} }
	, m_SkipOptimization{false}
	//Data
	, m_DiastolicInterval{200}
	, m_APThreshold{0}
	, m_APMinValue(0)
	, m_APMaxValue(0)
	, m_APD(0)
	, m_PathName{}
{
	LoadPlotData(int(m_DiastolicInterval.count()) + 2);
	m_TasksFinished.reserve(ThreadManager::GetInstance()->GetNrThreads());
}

Mesh::Mesh(ID3D11Device* pDevice, const std::vector<VertexInput>& vertices, const std::vector<uint32_t>& indices)
	: Mesh()
{
	CreateEffect(pDevice);
	CreateDirectXResources(pDevice, vertices, indices);
}

Mesh::Mesh(ID3D11Device* pDevice, const std::string& filepath, bool skipOptimization, FileType fileType, int nrOfThreads)
	: Mesh()
{
	m_PathName = filepath;
	m_SkipOptimization = skipOptimization;
	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		CreateEffect(pDevice);
	}

	switch (fileType)
	{
	case FileType::OBJ:
		LoadMeshFromOBJ(nrOfThreads);
		break;
	case FileType::VTK: //not functional
		LoadMeshFromVTK();
		break;
	case FileType::BIN:
		LoadMeshFromBIN();
		break;
	case FileType::PTS: //not functional
		LoadMeshFromPTS();
		break;
	default: ;
	}

	for (auto& vertex : m_VertexBuffer)
	{
		vertex.pPulseData->pNeighborIndicesRaw = vertex.pPulseData->pNeighborIndices.data();
		vertex.pPulseData->neighborIndicesSize = (uint32_t)vertex.pPulseData->pNeighborIndices.size();
	}

	std::cout << "Total vertex buffer size: " << m_VertexBuffer.size() * sizeof(VertexData) << std::endl;

	m_CudaUpdate.Setup(m_VertexBuffer, m_APPlot);
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to setup gpu data (error code %s)!\n",
			cudaGetErrorString(err));
	}

	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		CreateDirectXResources(pDevice, m_VertexDrawData, m_IndexBuffer);
	}
}

Mesh::~Mesh()
{
	if (m_pRasterizerStateSolid)
		m_pRasterizerStateSolid->Release();

	if (m_pRasterizerStateWireframe)
		m_pRasterizerStateWireframe->Release();

	if (m_pIndexBuffer)
		m_pIndexBuffer->Release();

	if (m_pVertexBuffer)
		m_pVertexBuffer->Release();

	if (m_pVertexLayout)
		m_pVertexLayout->Release();

	if (m_pEffect)
		delete m_pEffect;
}

void Mesh::Render(ID3D11DeviceContext* pDeviceContext, const float* worldViewProjMatrix, const float* inverseView)
{
	//Set vertex buffer
	UINT stride = sizeof(VertexInput);
	UINT offset = 0;
	pDeviceContext->IASetVertexBuffers(0, 1, &m_pVertexBuffer, &stride, &offset);

	//Set index buffer
	pDeviceContext->IASetIndexBuffer(m_pIndexBuffer, DXGI_FORMAT_R32_UINT, 0);

	//Set rasterizer state
	if (m_WireFrameEnabled)
		pDeviceContext->RSSetState(m_pRasterizerStateWireframe);
	else
		pDeviceContext->RSSetState(m_pRasterizerStateSolid);

	//Set input layout
	pDeviceContext->IASetInputLayout(m_pVertexLayout);

	//Set primitive topology
	pDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	//Set the worldviewprojectionMatrix
	m_pEffect->GetWorldViewProjMatrix()->SetMatrix(worldViewProjMatrix);

	//Set the worldMatrix
	glm::mat4 world = glm::transpose(m_WorldMatrix);

	float* data = (float*)glm::value_ptr(world);
	m_pEffect->GetWorldMatrix()->SetMatrix(data);

	//Set the InverseViewMatrix
	m_pEffect->GetViewInverseMatrix()->SetMatrix(inverseView);

	//Render mesh
	D3DX11_TECHNIQUE_DESC techDesc;
	m_pEffect->GetTechnique()->GetDesc(&techDesc);
	for (UINT p = 0; p < techDesc.Passes; ++p)
	{
		m_pEffect->GetTechnique()->GetPassByIndex(p)->Apply(0, pDeviceContext);
		if (m_DrawVertex)
		{
			pDeviceContext->Draw(uint32_t(m_VertexBuffer.size()), 0);
		}
		else
		{
			pDeviceContext->DrawIndexed(UINT(m_AmountIndices), 0, 0);
		}
		
	}
}

const glm::mat4& Mesh::GetWorldMatrix() const
{
	return m_WorldMatrix;
}

const std::vector<uint32_t>& Mesh::GetIndexBuffer() const
{
	return m_IndexBuffer;
}

const std::vector<VertexData>& Mesh::GetVertexBuffer() const
{
	return m_VertexBuffer;
}

std::vector<VertexData>& Mesh::GetVertexBufferReference()
{
	return m_VertexBuffer;
}

std::vector<VertexInput>& Mesh::GetVertexDrawData()
{
	return m_VertexDrawData;
}

const std::vector<float>& Mesh::GetAPPlot() const
{
	return m_APPlot;
}

glm::fvec2 Mesh::GetMinMax() const
{
	return glm::fvec2{ m_APMinValue, m_APMaxValue };
}

float Mesh::GetAPD() const
{
	return m_APD;
}

std::chrono::milliseconds Mesh::GetDiastolicInterval() const
{
	return m_DiastolicInterval;
}

void Mesh::SetVertexBuffer(ID3D11DeviceContext* pDeviceContext, const std::vector<VertexData>& vertexBuffer)
{
	m_VertexBuffer = vertexBuffer;
	UpdateVertexBuffer(pDeviceContext);
}

void Mesh::SetWireframe(bool enabled)
{
	m_WireFrameEnabled = enabled;
}

glm::fvec3 Mesh::GetScale()
{
	return glm::fvec3{ m_WorldMatrix[0].x, m_WorldMatrix[1].y , m_WorldMatrix[2].z };
}

glm::fvec3 Mesh::GetTranslation()
{
	return glm::fvec3{ m_WorldMatrix[3].x, m_WorldMatrix[3].y , m_WorldMatrix[3].z };
}

void Mesh::SetScale(const glm::fvec3& scale)
{
	SetScale(scale.x, scale.y, scale.z);
}

void Mesh::SetScale(float x, float y, float z)
{
	m_WorldMatrix[0].x = x;
	m_WorldMatrix[1].y = y;
	m_WorldMatrix[2].z = z;
}

void Mesh::Translate(const glm::fvec3& translation)
{
	Translate(translation.x, translation.y, translation.z);
}

void Mesh::Translate(float x, float y, float z)
{
	m_WorldMatrix[3].x = x;
	m_WorldMatrix[3].y = y;
	m_WorldMatrix[3].z = z;
}

void Mesh::SetDiastolicInterval(float diastolicInterval)
{
	m_DiastolicInterval = std::chrono::milliseconds(static_cast<long long>(diastolicInterval));
	LoadPlotData(int(m_DiastolicInterval.count()));
}

void Mesh::UseFibres(bool useFibres)
{
	m_UseFibres = useFibres;
}

bool Mesh::UseFibres()
{
	return m_FibresLoaded && m_UseFibres;
}

void Mesh::ConstantPulsing(bool constantPulsing)
{
	if (constantPulsing != m_IsPulsing)
		m_PulseCounter = 0.f;
	m_IsPulsing = constantPulsing;
}

bool Mesh::IsPulsing()
{
	return m_IsPulsing;
}

void Mesh::SetPulseRate(float pulseRate)
{
	m_PulseRate = pulseRate;
}

float Mesh::GetPulseRate()
{
	return m_PulseRate;
}

void Mesh::StartBenchmarking(const std::string& name)
{
	m_Benchmarking = true;
	Benchmarker::GetInstance()->StartBenchmark(m_PulseRate, name);
}

void Mesh::StopBenchmarking()
{
	m_Benchmarking = false;
	Benchmarker::GetInstance()->EndBenchmark();
}

void Mesh::CreateCachedBinary()
{
	std::cout << "\n[Started Writing File To Binary]\n";
	size_t pos = m_PathName.find('.');
	std::string path = m_PathName.substr(0, pos);
	path += ".bin";

	std::ofstream fileStream{ path, std::ios::out | std::ios::binary };
	if (fileStream.is_open())
	{
		//Write the number of indices & the index values
		const size_t nrOfIndices = m_IndexBuffer.size();
		fileStream.write((const char*)&nrOfIndices, sizeof(size_t));
		fileStream.write((const char*)&m_IndexBuffer[0], sizeof(uint32_t) * m_IndexBuffer.size());

		//Read the number of vertices & the vertices
		const size_t nrOfVertices = m_VertexBuffer.size();
		fileStream.write((const char*)&nrOfVertices, sizeof(size_t));

		for (size_t i = 0; i < nrOfVertices; i++)
		{
			fileStream.write((const char*)&m_VertexBuffer[i].pPulseData->position, sizeof(glm::fvec3));
			fileStream.write((const char*)&m_VertexDrawData[i].normal, sizeof(glm::fvec3));
			fileStream.write((const char*)&m_VertexDrawData[i].color1, sizeof(glm::fvec3));
			fileStream.write((const char*)&m_VertexDrawData[i].color2, sizeof(glm::fvec3));
			fileStream.write((const char*)&m_VertexDrawData[i].tangent, sizeof(glm::fvec3));
			fileStream.write((const char*)&m_VertexDrawData[i].uv, sizeof(glm::fvec2));

			const size_t nrOfNeighbours = m_VertexBuffer[i].pPulseData->pNeighborIndices.size();
			fileStream.write((const char*)&nrOfNeighbours, sizeof(size_t));

			for (const size_t& index : m_VertexBuffer[i].pPulseData->pNeighborIndices)
			{
				fileStream.write((const char*)&index, sizeof(size_t));
			}
		}
		std::cout << "[Finished Writing File To Binary]\n";
		std::cout << "File is written as a binary file in Resources/Models to decrease the loading time, type in [meshname].bin and load mesh as BIN\n";
	}
	else
	{
		std::cout << "[Failed To Write File To Binary]\n";
	}
}

void Mesh::CreateCachedFibreBinary()
{
	std::cout << "\n[Started Writing Fibres To Binary]\n";
	size_t pos = m_PathName.find_last_of('/');
	std::string path = m_PathName.substr(pos + 1);
	path = path.substr(0, path.find('.'));

	path = "Resources/FibreData/" + path + ".bin";

	std::ofstream fileStream{ path, std::ios::out | std::ios::binary };
	if (fileStream.is_open())
	{
		const size_t nrOfVertices = m_VertexBuffer.size();
		fileStream.write((const char*)&nrOfVertices, sizeof(size_t));

		for (const VertexData& vertex : m_VertexBuffer)
		{
			fileStream.write((const char*)&vertex.pPulseData->fibreDirection, sizeof(glm::fvec3));
		}

		std::cout << "\n[Finished Writing Fibres To Binary]\n";
	}
	else
	{
		std::cout << "Could not create file at " << path << "\n";
	}
}

void Mesh::LoadFibreData()
{
	std::cout << "\n[Started Reading Fibre Data]\n";

	size_t pos = m_PathName.find_last_of('/');
	std::string fibrePath = m_PathName.substr(pos);
	size_t extension = fibrePath.find('.');
	fibrePath = fibrePath.substr(0, extension);
	fibrePath = "Resources/FibreData" + fibrePath + ".txt";

	pos = m_PathName.find('.');
	std::string ptsPath = m_PathName.substr(0, pos) + ".pts";

	std::ifstream fileStream{ fibrePath, std::ios::in };
	std::ifstream ptsStream{ ptsPath, std::ios::in };

	if (fileStream.is_open() && ptsStream.is_open())
	{
		std::cout << "Loading point data from " << ptsPath << "\n";
		std::string line{};

		std::getline(ptsStream, line);

		size_t size = std::stoi(line);
		std::vector<glm::fvec3> points{};
		std::vector<glm::fvec3> fibres{};
		points.reserve(size);
		fibres.reserve(size);

		std::string value{};
		while (!ptsStream.eof())
		{
			std::getline(ptsStream, line);
			if (line != "")
			{
				value = line;
				size_t spacePos = value.find(' ');
				float x = std::stof(value.substr(0, spacePos));
				value.erase(0, spacePos + 1);
				spacePos = value.find(' ');
				float y = std::stof(value.substr(0, spacePos));
				value.erase(0, spacePos + 1);
				float z = std::stof(value);

				points.push_back(glm::fvec3{ x, y, z });
			}
		}

		std::cout << "Loading data from " << fibrePath << "\n";

		value = "";
		line = "";
		while (!fileStream.eof())
		{
			std::getline(fileStream, line);
			if (line != "")
			{
				//Read in the first three floats if the data
				value = line;
				size_t spacePos = value.find(' ');
				float fx = std::stof(value.substr(0, spacePos));
				value.erase(0, spacePos + 1);
				spacePos = value.find(' ');
				float fy = std::stof(value.substr(0, spacePos));
				value.erase(0, spacePos + 1);
				spacePos = value.find(' ');
				float fz = std::stof(value.substr(0, spacePos));

				fibres.push_back(glm::fvec3{ fx, fy, fz });
			}
		}

		//Loop over the pts file and compare the positions in there with the position of all the vertex
		//We can then assign the correct fiber to that vertex
		for (int i{}; i < points.size(); i++)
		{
			if (i % 100 == 0 || i == points.size() - 1)
			{
				printf("\33[2K\r");
				int percentage = int((float(i) / float(points.size() - 1)) * 100);
				std::cout << i << " / " << points.size() - 1 << " " << percentage << "%";
			}

			float epsilon = 0.1f;
			float epsilonIncrease = 0.5f;
			float epsilonMax = 10.f;

			const glm::fvec3& point = points[i];
			std::vector<VertexData>::iterator it = std::find_if(m_VertexBuffer.begin(), m_VertexBuffer.end(), [point, epsilon](const VertexData& vertex)
				{
					return (glm::epsilonEqual(point, vertex.pPulseData->position, epsilon).y);
				}
			);

			while (it == m_VertexBuffer.end())
			{
				epsilon += epsilonIncrease;

				if (epsilon > epsilonMax)
					break;

				it = std::find_if(m_VertexBuffer.begin(), m_VertexBuffer.end(), [point, epsilon](const VertexData& vertex)
					{
						return (glm::epsilonEqual(point, vertex.pPulseData->position, epsilon).y);
					}
				);
			}

			while (it != m_VertexBuffer.end())
			{
				it->pPulseData->fibreDirection = fibres[i];
				++it;
				it = std::find_if(it, m_VertexBuffer.end(), [point, epsilon](const VertexData& vertex)
					{
						return (glm::epsilonEqual(point, vertex.pPulseData->position, epsilon).y);
					}
				);
			}
		}

		CreateCachedFibreBinary();
		m_FibresLoaded = true;
	}
	else
	{
		if (!fileStream.is_open())
			std::cout << "Could not load fibre data from "<< fibrePath <<"\n";

		if (!ptsStream.is_open())
			std::cout << "Could not load point data from " << ptsPath << "\n";
	}
	std::cout << "\n[Finished Reading Fibre Data]\n";
}

void Mesh::LoadCachedFibres()
{
	size_t pos = m_PathName.find_last_of('/');
	std::string path = m_PathName.substr(pos + 1);
	path = path.substr(0, path.find('.'));

	path = "Resources/FibreData/" + path + ".bin";

	std::cout << "\n[Started Reading Binary Fibre Data]\n";
	std::ifstream fileStream{ path, std::ios::in | std::ios::binary };
	if (fileStream.is_open())
	{
		size_t size{};
		fileStream.read((char*)&size, sizeof(size_t));
		for (size_t i{}; i < size; i++)
		{
			if (!m_VertexBuffer.empty() && i >=0 && i < m_VertexBuffer.size())
			{
				glm::fvec3 fibre{};
				fileStream.read((char*)&fibre, sizeof(glm::fvec3));
				m_VertexBuffer[i].pPulseData->fibreDirection = fibre;
			}
		}

		std::cout << "\n[Finished Reading Binary Fibre Data]\n";
		m_FibresLoaded = true;
	}
	else
	{
		std::cout << "Could not load file at " << path << "\n";
		LoadFibreData();
	}
}

void Mesh::UpdateMeshV3(ID3D11DeviceContext* pDeviceContext)
{
	if (m_IsPulsing)
	{
		m_PulseCounter += TimeSingleton::GetInstance()->DeltaTime();
		if (m_PulseCounter > 1/m_PulseRate)
		{
			m_PulseCounter -= 1 / m_PulseRate;
			PulseVertexV3(uint32_t(0), pDeviceContext);
		}
	}

	auto startTime = std::chrono::high_resolution_clock::now();

	switch (m_UpdateSystem)
	{
	case UpdateSystem::Serial:
		UpdateSerial();
		break;
	case UpdateSystem::Multithreaded:
		UpdateThreaded();
		break;
	case UpdateSystem::GPU:
		UpdateGPU();
		break;
	}


	if(m_Benchmarking)
		Benchmarker::GetInstance()->AddDuration(std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startTime).count());

	UpdateVertexBuffer(pDeviceContext);
}

void Mesh::UpdateSerial()
{
	UpdateVertexCluster( 0, (int)m_VertexBuffer.size());
}

void Mesh::UpdateThreaded()
{
	int nrThreads = int(ThreadManager::GetInstance()->GetNrThreads());
	int jobSize = int(ceil(m_VertexBuffer.size() / float(nrThreads)));

	int firstVertex = 0;

	for (int i = 0; i < nrThreads; i++)
	{
		m_TasksFinished.push_back(ThreadManager::GetInstance()->AddJobFunction(std::bind(&Mesh::UpdateVertexCluster, this, firstVertex, jobSize)));
		firstVertex += int(jobSize);
		//m_TasksFinished.push_back(ThreadManager::GetInstance()->AddJobFunction(std::bind(&Mesh::UpdateVertexParallell, this, deltaTimeInMs, deltaTime, dist, pDeviceContext, nrThreads, i)));
	}

	for (size_t i = 0; i < m_TasksFinished.size(); i++)
	{
		m_TasksFinished[i].wait();
	}
	m_TasksFinished.clear();
}

void Mesh::UpdateGPU()
{
	m_CudaUpdate.Update(m_APMinValue, m_APD, float(m_DiastolicInterval.count()), TimeSingleton::GetInstance()->DeltaTimeInMs(), TimeSingleton::GetInstance()->DeltaTime(), m_ConductionVelocity, UseFibres());
	cudaDeviceSynchronize();
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch execute gpu update (error code %s)!\n",
			cudaGetErrorString(err));
	}
}

void Mesh::SetUpdateSystem(UpdateSystem system)
{
	m_UpdateSystem = system;
}


//void Mesh::UpdateGPU(float deltaTimeInMs, float deltaTime, float dist, ID3D11DeviceContext* pDeviceContext)
//{
//
//}

void Mesh::UpdateVertexCluster( int firstVertex, int vertexCount)
{
	for (size_t i = firstVertex; i < firstVertex+vertexCount; i++)
	{
		if (i == m_VertexBuffer.size())
		{
			break;
		}

		switch (m_VertexBuffer[i].state)
		{
		case State::APD:
		{
			m_VertexBuffer[i].timePassed += TimeSingleton::GetInstance()->DeltaTimeInMs();

			int idx = int(m_VertexBuffer[i].timePassed);

			if (!m_APPlot.empty() && idx > 0 && idx < m_APPlot.size() && (size_t(idx) + size_t(1)) < m_APPlot.size())
			{
				float value1 = m_APPlot[idx];
				float value2 = m_APPlot[(size_t(idx) + size_t(1))];
				float t = m_VertexBuffer[i].timePassed - idx;

				float lerpedValue = value1 + t * (value2 - value1);

				//float valueRange01 = (lerpedValue - m_APMinValue) / dist;

				m_VertexBuffer[i].actionPotential = lerpedValue;
			}

			if (m_VertexBuffer[i].timePassed >= m_APD)
			{
				m_VertexBuffer[i].actionPotential = m_APMinValue;
				m_VertexBuffer[i].timePassed = 0.f;
				m_VertexBuffer[i].state = State::DI;
			}

			break;
		}
		case State::DI:
			m_VertexBuffer[i].timePassed += TimeSingleton::GetInstance()->DeltaTimeInMs();

			if (m_VertexBuffer[i].timePassed >= m_DiastolicInterval.count())
			{
				m_VertexBuffer[i].timePassed = 0.f;
				m_VertexBuffer[i].state = State::Waiting;
			}
			break;

		case State::Receiving:
			m_VertexBuffer[i].timePassed -= TimeSingleton::GetInstance()->DeltaTime();
			if (m_VertexBuffer[i].timePassed <= 0.f)
			{
				m_VertexBuffer[i].state = State::Waiting;
				PulseVertexV3(&m_VertexBuffer[i], false);
			}
			break;
		}
	}
}

void Mesh::UpdateVertexParallell( int nrThreads, int threadId)
{
	auto size = m_VertexBuffer.size(); //avoid disturbing cache by accesing m_VertexBuffer every loop
	for (size_t i = threadId; i < size; i += nrThreads)
	{
		if (i == m_VertexBuffer.size())
		{
			break;
		}

		switch (m_VertexBuffer[i].state)
		{
		case State::APD:
		{
			m_VertexBuffer[i].timePassed += TimeSingleton::GetInstance()->DeltaTimeInMs();

			int idx = int(m_VertexBuffer[i].timePassed);

			if (!m_APPlot.empty() && idx > 0 && idx < m_APPlot.size() && (size_t(idx) + size_t(1)) < m_APPlot.size())
			{
				float value1 = m_APPlot[idx];
				float value2 = m_APPlot[(size_t(idx) + size_t(1))];
				float t = m_VertexBuffer[i].timePassed - idx;

				float lerpedValue = value1 + t * (value2 - value1);

				//float valueRange01 = (lerpedValue - m_APMinValue) / dist;

				m_VertexBuffer[i].actionPotential = lerpedValue;
			}

			if (m_VertexBuffer[i].timePassed >= m_APD)
			{
				m_VertexBuffer[i].actionPotential = m_APMinValue;
				m_VertexBuffer[i].timePassed = 0.f;
				m_VertexBuffer[i].state = State::DI;
			}

			break;
		}
		case State::DI:
			m_VertexBuffer[i].timePassed += TimeSingleton::GetInstance()->DeltaTimeInMs();

			if (m_VertexBuffer[i].timePassed >= m_DiastolicInterval.count())
			{
				m_VertexBuffer[i].timePassed = 0.f;
				m_VertexBuffer[i].state = State::Waiting;
			}
			break;

		case State::Receiving:
			m_VertexBuffer[i].timePassed -= TimeSingleton::GetInstance()->DeltaTime();
			if (m_VertexBuffer[i].timePassed <= 0.f)
			{
				m_VertexBuffer[i].state = State::Waiting;
				PulseVertexV3(&m_VertexBuffer[i], false);
			}
			break;
		}
	}
}

void Mesh::PulseVertexV3(uint32_t index, bool updateVertexBuffer)
{
	if (!m_VertexBuffer.empty() && index >= 0 && index < m_VertexBuffer.size())
	{
		PulseVertexV3(&m_VertexBuffer[index], updateVertexBuffer);

		if (m_UpdateSystem == UpdateSystem::GPU)
		{
			m_CudaUpdate.PulseVertex(0, m_ConductionVelocity, UseFibres());
			cudaDeviceSynchronize();
			auto err = cudaGetLastError();
			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to pulse gpu vertex (error code %s)!\n",
					cudaGetErrorString(err));
			}
		}
	}
}

void Mesh::PulseVertexV3(VertexData* vertex, bool )
{
	if (vertex)
	{
		if (vertex->state == State::Waiting || vertex->state == State::DI)
		{
			vertex->actionPotential = m_APPlot[0];
			vertex->state = State::APD;
			vertex->timePassed = 0;

			for (uint32_t index : vertex->pPulseData->pNeighborIndices)
			{
				VertexData& neighbourVertex = m_VertexBuffer[index];

				//Potential problem with fibres. c0 is in m/s while the distance is most likely not in meters.
					//This is likely the cause of it.
				float distanceSqrd = glm::distance2(vertex->pPulseData->position, neighbourVertex.pPulseData->position);
				float conductionVelocity = m_ConductionVelocity;

				if (UseFibres())
				{
					float d1 = 1; // parallel with fibre
					float d2 = d1 / 5; // perpendiculat with fibre
					float c0 = 0.6f; // m/s

					glm::fvec3 pulseDirection = neighbourVertex.pPulseData->position - vertex->pPulseData->position;
					float cosAngle = glm::dot(vertex->pPulseData->fibreDirection, pulseDirection);

					float c = c0 * sqrtf(d2 + (d1 - d2) * powf(cosAngle, 2));
					conductionVelocity = c * 100;
					//std::cout << c << "\n";
				}

				float travelTime = distanceSqrd / conductionVelocity;

				if (neighbourVertex.state == State::Waiting)
				{
					neighbourVertex.timePassed = travelTime;
					neighbourVertex.state = State::Receiving;
				}
			}
		}
	}

}

void Mesh::PulseMesh()
{
	for (int i{}; i < m_VertexBuffer.size(); i++)
	{
		PulseVertexV3(i, false);
	}
}

void Mesh::ClearPulse(ID3D11DeviceContext* )
{
	for (VertexData& vertex : m_VertexBuffer)
	{
		vertex.actionPotential = m_APMinValue;
		vertex.state = State::Waiting;
		vertex.timePassed = 0.f;
	}
}

HRESULT Mesh::CreateDirectXResources(ID3D11Device* pDevice, const std::vector<VertexInput>& vertices, const std::vector<uint32_t>& indices)
{
	HRESULT result = S_OK;

	if (m_pIndexBuffer)
		m_pIndexBuffer->Release();

	if (m_pVertexBuffer)
		m_pVertexBuffer->Release();

	if (m_pVertexLayout)
		m_pVertexLayout->Release();

	//Create vertex layout
	static const uint32_t numElements{ 7 };
	D3D11_INPUT_ELEMENT_DESC vertexDesc[numElements]{};

	vertexDesc[0].SemanticName = "POSITION";
	vertexDesc[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	vertexDesc[0].AlignedByteOffset = 0;
	vertexDesc[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;

	vertexDesc[1].SemanticName = "COLOR";
	vertexDesc[1].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	vertexDesc[1].AlignedByteOffset = 12;
	vertexDesc[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;

	vertexDesc[2].SemanticName = "SECCOLOR";
	vertexDesc[2].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	vertexDesc[2].AlignedByteOffset = 24;
	vertexDesc[2].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;

	vertexDesc[3].SemanticName = "NORMAL";
	vertexDesc[3].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	vertexDesc[3].AlignedByteOffset = 36;
	vertexDesc[3].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;

	vertexDesc[4].SemanticName = "TANGENT";
	vertexDesc[4].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	vertexDesc[4].AlignedByteOffset = 48;
	vertexDesc[4].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;

	vertexDesc[5].SemanticName = "TEXCOORD";
	vertexDesc[5].Format = DXGI_FORMAT_R32G32_FLOAT;
	vertexDesc[5].AlignedByteOffset = 60;
	vertexDesc[5].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;

	vertexDesc[6].SemanticName = "POWER";
	vertexDesc[6].Format = DXGI_FORMAT_R32_FLOAT;
	vertexDesc[6].AlignedByteOffset = 68;
	vertexDesc[6].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;

	//Create the input layout
	D3DX11_PASS_DESC passDesc;
	m_pEffect->GetTechnique()->GetPassByIndex(0)->GetDesc(&passDesc);
	result = pDevice->CreateInputLayout(
		vertexDesc,
		numElements,
		passDesc.pIAInputSignature,
		passDesc.IAInputSignatureSize,
		&m_pVertexLayout
	);
	
	//Create vertex buffers
	D3D11_BUFFER_DESC bd = {};
	bd.Usage = D3D11_USAGE_DYNAMIC;
	bd.ByteWidth = UINT(sizeof(VertexInput) * vertices.size());
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bd.MiscFlags = 0;
	D3D11_SUBRESOURCE_DATA initData = { 0 };
	initData.pSysMem = vertices.data();
	result = pDevice->CreateBuffer(&bd, &initData, &m_pVertexBuffer);
	if (FAILED(result))
		return result;

	//Create index buffer
	m_AmountIndices = indices.size();
	bd.Usage = D3D11_USAGE_IMMUTABLE;
	bd.ByteWidth = UINT(sizeof(uint32_t) * m_AmountIndices);
	bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;
	initData.pSysMem = indices.data();
	result = pDevice->CreateBuffer(&bd, &initData, &m_pIndexBuffer);
	if (FAILED(result))
		return result;

	//Create wireframe rasterizer state
	D3D11_RASTERIZER_DESC rswf{};
	rswf.FillMode = D3D11_FILL_WIREFRAME;
	rswf.CullMode = D3D11_CULL_NONE;
	rswf.DepthClipEnable = true;
	result = pDevice->CreateRasterizerState(&rswf, &m_pRasterizerStateWireframe);
	if (FAILED(result))
		return result;

	//Create solid rasterizer state
	D3D11_RASTERIZER_DESC rss{};
	rss.FillMode = D3D11_FILL_SOLID;
	rss.CullMode = D3D11_CULL_NONE;
	rss.FrontCounterClockwise = true;
	result = pDevice->CreateRasterizerState(&rss, &m_pRasterizerStateSolid);
	if (FAILED(result))
		return result;

	return result;
}

void Mesh::LoadMeshFromOBJ(uint32_t )
{
	auto timeStart = std::chrono::high_resolution_clock::now();

	std::cout << "\n[Started Loading Mesh]\n";
	std::cout << "\n--- Started Reading Mesh File ---\n";
	objl::Loader loader;

	bool loadout = loader.LoadFile(m_PathName);
	if (loadout)
	{
		if (loader.LoadedMeshes.size() > 0)
		{
			glm::fvec3 color1 = {  50 / 255.f, 151 / 255.f, 142 / 255.f };
			glm::fvec3 color2 = { 225 / 255.f,  73 / 255.f,  80 / 255.f } ;

			size_t count = 0;
			for (const objl::Vertex& vertex : loader.LoadedVertices)
			{
				m_VertexBuffer.push_back(VertexData{ {vertex.Position.X, vertex.Position.Y, vertex.Position.Z} });
				m_VertexDrawData.push_back(VertexInput{ { vertex.Position.X, vertex.Position.Y, vertex.Position.Z },
					color1,
					color2,
					{ vertex.Normal.X, vertex.Normal.Y, vertex.Normal.Z },
					{ vertex.TextureCoordinate.X, vertex.TextureCoordinate.Y },
					count });

				++count;
			}

			for (unsigned int index : loader.LoadedIndices)
			{
				m_IndexBuffer.push_back(index);
			}

			m_AmountIndices = size_t(m_IndexBuffer.size());
			std::cout << "--- Finished Reading Mesh File ---\n";
			std::cout << m_VertexBuffer.size() << " Vertices Read\n";
			std::cout << m_IndexBuffer.size() << " Indices Read\n";

			LoadCachedFibres();

			//Remove indices pointing towards duplicate vertices
			/*if (!m_SkipOptimization)
				OptimizeIndexBuffer();*/

			OptimizeVertexAndIndexBuffer();

			CalculateTangents();

			//Remove the duplicate vertices from the vertex buffer
			//if (!m_SkipOptimization)
			//{
			//	std::cout << "\n--- Started Optimizing Vertex Buffer ---\n";
			//	OptimizeVertexBuffer();
			//	std::cout << "--- Finished Optimizing Vertex Buffer ---\n";
			//}

			//Get the neighbour of every vertex
		
			std::cout << "\n--- Started Calculating Vertex Neighbours ---\n";
			CalculateNeighbours(1);
			std::cout << "--- Finished Calculating Vertex Neighbours ---\n";
			

			//CalculateInnerNeighbours();

			std::cout << "\n" << m_VertexBuffer.size() << " Vertices After Optimization\n";

			CreateCachedBinary();

			auto timeEnd = std::chrono::high_resolution_clock::now();
			auto time = timeEnd - timeStart;
			auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time);
			std::cout << "\n[Finished Loading Mesh]\n";
			std::cout << "[Loaded In " << seconds.count() << " Seconds]\n";

		}
	}

}

void Mesh::LoadMeshFromVTK()
{
	std::cout << "\n[Started Reading Mesh]\n";
	size_t pos = m_PathName.find('.');
	std::string path = m_PathName.substr(0, pos);

	std::map<uint32_t, uint32_t> indicesToReplace{};

	std::string ptsFilePath = path + ".pts";
	std::ifstream vertexStream{ m_PathName };
	if (vertexStream.is_open())
	{
		std::string line{};
		std::getline(vertexStream, line);
		size_t vertCount = std::stoi(line);
		m_VertexBuffer.reserve(vertCount);
		m_VertexDrawData.reserve(vertCount);

		int index = 0;
		while (!vertexStream.eof())
		{
			if (index >= vertCount)
				break;

			VertexData vertex{};
			VertexInput vertexDrawData{};

			vertexStream >> vertex.pPulseData->position.x >> vertex.pPulseData->position.y >> vertex.pPulseData->position.z;

			vertex.pPulseData->position /= 1000.f;
			m_VertexBuffer.push_back(vertex);
			++index;
		}

		std::string elemFilePath = path + ".surf";
		std::string discardValue{};

		std::ifstream indexStream{ elemFilePath };
		if (indexStream.is_open())
		{
			while (!indexStream.eof())
			{
				std::getline(indexStream, line, ' ');
				std::cout << line << "\n";

				if (line.empty() || line == "\n")
					break;


				size_t indexCount = std::stoi(line);
				std::getline(indexStream, line);
				m_IndexBuffer.resize(m_IndexBuffer.size() + indexCount);

				for (int i{}; i < indexCount; i++)
				{
					glm::ivec3 indices{};
					indexStream >> discardValue >> indices.x >> indices.y >> indices.z;

					for (int j{}; j < 3; j++)
					{
						m_IndexBuffer[i] = indices[j];
					}
				}
			}
		}

		std::cout << "Vertex Buffer Size: " << m_VertexBuffer.size() << "\n";
		std::cout << "Index Buffer Size: " << m_IndexBuffer.size() << "\n";
		std::cout << "\n[Finished Reading Mesh]\n";
		//for (uint32_t i{}; i < m_IndexBuffer.size(); i+= 3)
		//{
		//	uint32_t index1 = m_IndexBuffer[i];
		//	uint32_t index2 = m_IndexBuffer[i + 1];
		//	uint32_t index3 = m_IndexBuffer[i + 2];

		//	VertexInput& vertex1 = m_VertexBuffer[index1];
		//	VertexInput& vertex2 = m_VertexBuffer[index2];
		//	VertexInput& vertex3 = m_VertexBuffer[index3];

		//	glm::fvec3 normal = (glm::cross(vertex2.position - vertex1.position, vertex3.position - vertex1.position));
		//	normal = glm::normalize(normal);
		//	vertex1.normal = normal;
		//	vertex2.normal = normal;
		//	vertex3.normal = normal;

		//	if (vertex1.index == 0 || vertex2.index == 0 || vertex3.index == 0)
		//		std::cout << normal.x << ", " << normal.y << ", " << normal.z << "\n";
		//}

		indexStream.close();
		vertexStream.close();
	}

	std::cout << "Started Calculating Neighbours\n";
	CalculateNeighbours(6);
	std::cout << "Finished Calculating Neighbours\n";
}

void Mesh::LoadMeshFromPTS()
{
	std::ifstream vertexStream{ m_PathName };
	if (vertexStream.is_open())
	{
		std::string line{};
		std::getline(vertexStream, line);
		size_t vertCount = std::stoi(line);
		m_VertexBuffer.reserve(vertCount);

		int index = 0;
		while (!vertexStream.eof())
		{
			if (index >= vertCount)
				break;

			VertexData vertex{};
			//vertex.pVisualizationData->index = index;

			vertexStream >> vertex.pPulseData->position.x >> vertex.pPulseData->position.y >> vertex.pPulseData->position.z;

			vertex.pPulseData->position /= 1000.f;
			m_VertexBuffer.push_back(vertex);
			++index;
		}

		for (int i{}; i < m_VertexBuffer.size(); i++)
		{
			m_IndexBuffer.push_back(i);
		}

		OptimizeIndexBuffer();
		OptimizeVertexBuffer();

		CreateIndexForVertices();

		CalculateNeighbours();
		CalculateInnerNeighbours();

		CreateCachedBinary();

		m_DrawVertex = true;
	}
}

void Mesh::LoadMeshFromBIN()
{
	std::ifstream fileStream{ m_PathName, std::ios::in | std::ios::binary };
	if (fileStream.is_open())
	{
		//Read in the indexbuffer
		m_IndexBuffer.clear();
		size_t nrOfIndices{};
		fileStream.read((char*)&nrOfIndices, sizeof(size_t));
		m_AmountIndices = size_t(nrOfIndices);

		m_IndexBuffer.resize(nrOfIndices);
		fileStream.read((char*)&m_IndexBuffer[0], sizeof(uint32_t) * nrOfIndices);

		//Read in the vertices
		m_VertexBuffer.clear();
		size_t nrOfVertices{};
		fileStream.read((char*)&nrOfVertices, sizeof(size_t));

		m_VertexBuffer.resize(nrOfVertices);
		m_VertexDrawData.resize(nrOfVertices);
		for (size_t i = 0; i < nrOfVertices; i++)
		{
			fileStream.read((char*)&m_VertexBuffer[i].pPulseData->position, sizeof(glm::fvec3));
			fileStream.read((char*)&m_VertexDrawData[i].normal, sizeof(glm::fvec3));
			fileStream.read((char*)&m_VertexDrawData[i].color1, sizeof(glm::fvec3));
			fileStream.read((char*)&m_VertexDrawData[i].color2, sizeof(glm::fvec3));
			fileStream.read((char*)&m_VertexDrawData[i].tangent, sizeof(glm::fvec3));
			fileStream.read((char*)&m_VertexDrawData[i].uv, sizeof(glm::fvec2));
			m_VertexDrawData[i].position = m_VertexBuffer[i].pPulseData->position;

			size_t nrOfNeighbours{};
			fileStream.read((char*)&nrOfNeighbours, sizeof(size_t));

			for (size_t j{}; j < nrOfNeighbours; j++)
			{
				size_t neighbourIndex{};
				fileStream.read((char*)&neighbourIndex, sizeof(size_t));

				m_VertexBuffer[i].pPulseData->pNeighborIndices.push_back(uint32_t(neighbourIndex));
			}
		}

		CreateIndexForVertices();

		LoadCachedFibres();

		std::string name = "Vertex Buffer " + m_PathName;
		Logger::Get().LogBuffer<VertexData>(m_VertexBuffer, name);
		name = "Index Buffer " + m_PathName;
		Logger::Get().LogBuffer<uint32_t>(m_IndexBuffer, name);
		Logger::Get().EndSession();

		std::cout << "index buffer size: " << m_IndexBuffer.size() << std::endl;
		std::cout << "vertex buffer size: " << m_VertexBuffer.size() << std::endl;
	}
}

void Mesh::CalculateTangents()
{
	for (int i = 0; i < m_IndexBuffer.size(); i++)
	{
		if (i % 3 == 2)
		{
			//Change handedness
			std::swap(m_IndexBuffer[i - 1], m_IndexBuffer[i]);
		}

		if (i % 3 == 0)
		{
			//Calculate Tangent
			size_t index0 = m_IndexBuffer[i];
			size_t index1 = m_IndexBuffer[i + 1];
			size_t index2 = m_IndexBuffer[i + 2];
			VertexInput& vertex0 = m_VertexDrawData[index0];
			VertexInput& vertex1 = m_VertexDrawData[index1];
			VertexInput& vertex2 = m_VertexDrawData[index2];

			const glm::fvec3 edge0 = (vertex1.position - vertex0.position);
			const glm::fvec3 edge1 = (vertex2.position - vertex0.position);

			const glm::fvec2 diffX = glm::fvec2(vertex1.uv.x - vertex0.uv.x, vertex2.uv.x - vertex0.uv.x);
			const glm::fvec2 diffY = glm::fvec2(vertex1.uv.y - vertex0.uv.y, vertex2.uv.y - vertex0.uv.y);
			const float r = 1.f / (diffX.x * diffY.y - diffX.y * diffY.x);

			glm::vec3 tangent = (edge0 * diffY.y - edge1 * diffY.x) * r;
			vertex0.tangent = tangent;
			vertex1.tangent = tangent;
			vertex2.tangent = tangent;
		}
	}
}

void Mesh::OptimizeIndexBuffer()
{
	std::cout << "--- Started Optimizing Index Buffer ---\n";
	//Get rid of out of bounds indices
	std::vector<uint32_t>::iterator removeIt = std::remove_if(m_IndexBuffer.begin(), m_IndexBuffer.end(), [this](uint32_t index)
		{
			return index >= m_VertexBuffer.size();
		});

	if (removeIt != m_IndexBuffer.end())
		m_IndexBuffer.erase(removeIt);

	//Loop over all the indices and check if they're pointing to duplicate vertices
	//std::set<uint32_t> seenIndices{};
	std::cout << std::endl;

	bool firstRun = true;
	TimePoint start = std::chrono::high_resolution_clock::now();
	TimePoint startActual = std::chrono::high_resolution_clock::now();

	for (uint32_t i{}; i < m_IndexBuffer.size(); i++)
	{

		if ((i % 1000 == 0 && !firstRun) || i == m_IndexBuffer.size() - 1)
		{
			printf("\33[2K\r");
			int percentage = int((float(i) / float(m_IndexBuffer.size() - 1)) * 100);
			std::cout << i << " / " << m_IndexBuffer.size() << " " << percentage << "%";
		}
		uint32_t index = m_IndexBuffer[i];
		if (index >= m_VertexBuffer.size())
			continue;

		//std::set<uint32_t>::iterator itFind = std::find(seenIndices.begin(), seenIndices.end(), index);
		//if (itFind != seenIndices.end())
		//{
		//	continue;
		//}

		VertexData vertex = m_VertexBuffer[index];

		auto CompareIndexVertex = [this, &vertex](uint32_t index)
		{
			return vertex == m_VertexBuffer[index];
		};

		std::vector<uint32_t> duplicates{};
		std::vector<uint32_t>::iterator it = std::find_if(m_IndexBuffer.begin() + i, m_IndexBuffer.end(), CompareIndexVertex);

		while (it != m_IndexBuffer.end())
		{
			duplicates.push_back(*it);
			++it;
			it = std::find_if(it, m_IndexBuffer.end(), CompareIndexVertex);
		}

		//Replace indices that are pointing to duplicates with one index
		for (uint32_t duplicate : duplicates)
		{
			std::replace(m_IndexBuffer.begin() + i, m_IndexBuffer.end(), duplicate, index);
		}
		//seenIndices.insert(index);

		if (firstRun)
		{
			TimePoint end = std::chrono::high_resolution_clock::now();
			auto seconds = std::chrono::duration_cast<std::chrono::seconds>((end - start) * m_IndexBuffer.size());
			auto minutes = std::chrono::duration_cast<std::chrono::minutes>((end - start) * m_IndexBuffer.size());
			auto hours = std::chrono::duration_cast<std::chrono::hours>((end - start) * m_IndexBuffer.size());
			std::cout << "Estimated Time: " << seconds.count() << " seconds" << "( " << hours.count() << " hours and " << minutes.count() - (hours.count() * 60 ) << " minutes )\n";
			firstRun = false;
		}
	}
	TimePoint endActual = std::chrono::high_resolution_clock::now();
	auto seconds = std::chrono::duration_cast<std::chrono::seconds>(endActual - startActual);
	auto minutes = std::chrono::duration_cast<std::chrono::minutes>(endActual - startActual);
	auto hours = std::chrono::duration_cast<std::chrono::hours>(endActual - startActual);
	std::cout << "\nOptimizing took " << seconds.count() << " seconds" << "( " << hours.count() << " hours and " << minutes.count() - (hours.count() * 60) << " minutes )\n";

	std::cout << "--- Finished Optimizing Index Buffer ---\n";
}

void Mesh::OptimizeVertexBuffer()
{
	//std::cout << "Removing Duplicate Indices\n";
	//std::vector<size_t> indicesToRemove{};
	//std::vector<VertexData>::iterator it = std::remove_if(m_VertexBuffer.begin(), m_VertexBuffer.end(), [&indicesToRemove, this](const VertexData& vertex)
	//	{
	//		std::vector<size_t>::iterator itFind = std::find(m_IndexBuffer.begin(), m_IndexBuffer.end(), vertex.pVisualizationData->index);

	//		bool shouldRemove = itFind == m_IndexBuffer.end();
	//		if (shouldRemove)
	//			indicesToRemove.push_back(vertex.pVisualizationData->index);

	//		return shouldRemove;
	//	});

	//m_VertexBuffer.erase(it, m_VertexBuffer.end());

	//std::cout << "Reconstructing Index Buffer\n";
	//std::vector<size_t> indexBufferSwap{m_IndexBuffer};
	//for (uint32_t i{}; i < indicesToRemove.size(); i++)
	//{
	//	size_t indexToRemove = indicesToRemove[i];

	//	for (uint32_t j{}; j < m_IndexBuffer.size(); j++)
	//	{
	//		if (indexBufferSwap[j] > indexToRemove)
	//		{
	//			--m_IndexBuffer[j];
	//		}
	//	}
	//}

	//std::cout << "Reassigning Indices To Vertices\n";
	//for (int i{}; i < m_VertexBuffer.size(); i++)
	//{
	//	m_VertexBuffer[i].pVisualizationData->index = i;
	//}

	// 1 3 2 2 4 3 7 9 8
	// [1] [2] [3] [4] [5] [6] [7] [8] [9]
	// Remove [5]			v	v	v	v
	// [1] [2] [3] [4]	   [5] [6] [7] [8] //All vertices with original index > 5, decrement
	// Remove [6]			v		v	v
	// [1] [2] [3] [4]	   [5]     [6] [7] //All vertices with original index > 6, decrement
}

void Mesh::OptimizeVertexAndIndexBuffer()
{
	std::unordered_map<VertexData, uint32_t> VerticesMap{};
	std::vector<uint32_t> uniqueIndices{};
	std::vector<VertexData> uniqueVertices{};
	std::vector<VertexInput> uniqueVertexDrawData{};

	for (size_t i = 0; i < m_VertexBuffer.size(); i++)
	{
		if (VerticesMap.count(m_VertexBuffer[i]) == 0)
		{
			VerticesMap[m_VertexBuffer[i]] = (uint32_t)uniqueVertices.size();
			uniqueVertices.push_back(m_VertexBuffer[i]);
			uniqueVertexDrawData.push_back(m_VertexDrawData[i]);
		}

		uniqueIndices.push_back(VerticesMap[m_VertexBuffer[i]]);
	}

	m_VertexDrawData = uniqueVertexDrawData;
	m_VertexBuffer = uniqueVertices;
	m_IndexBuffer = uniqueIndices;
}

void Mesh::CalculateNeighbours(int nrOfThreads)
{
	auto GetNeighboursInRange = [this](uint32_t start, uint32_t end)
	{
		for (uint32_t i{ start }; i < end; i += 3)
		{
			/*std::vector<uint32_t>::iterator it = std::find(m_IndexBuffer.begin() + start, m_IndexBuffer.begin() + end, i);
			while (it != m_IndexBuffer.end() && it < m_IndexBuffer.begin() + end)
			{
				uint32_t index = uint32_t(it - m_IndexBuffer.begin());
				int modulo = index % 3;
				if (modulo == 0)
				{
					if (it + 1 != m_IndexBuffer.end())
						m_VertexBuffer[i].neighbourIndices.insert(*(it + 1));
					if (it + 2 != m_IndexBuffer.end())
						m_VertexBuffer[i].neighbourIndices.insert(*(it + 2));
				}
				else if (modulo == 1)
				{
					if (it - 1 != m_IndexBuffer.end())
						m_VertexBuffer[i].neighbourIndices.insert(*(it - 1));
					if (it + 1 != m_IndexBuffer.end())
						m_VertexBuffer[i].neighbourIndices.insert(*(it + 1));
				}
				else
				{
					if (it - 1 != m_IndexBuffer.end())
						m_VertexBuffer[i].neighbourIndices.insert(*(it - 1));
					if (it - 2 != m_IndexBuffer.end())
						m_VertexBuffer[i].neighbourIndices.insert(*(it - 2));
				}

				it++;
				it = std::find(it, (m_IndexBuffer.begin() + end), i);
			}*/

			auto idx1 = m_IndexBuffer[i];
			auto idx2 = m_IndexBuffer[i + 1];
			auto idx3 = m_IndexBuffer[i + 2];

			m_VertexBuffer[idx1].pPulseData->pNeighborIndices.push_back(idx2);
			m_VertexBuffer[idx1].pPulseData->pNeighborIndices.push_back(idx3);
			m_VertexBuffer[idx2].pPulseData->pNeighborIndices.push_back(idx1);
			m_VertexBuffer[idx2].pPulseData->pNeighborIndices.push_back(idx3);
			m_VertexBuffer[idx3].pPulseData->pNeighborIndices.push_back(idx2);
			m_VertexBuffer[idx3].pPulseData->pNeighborIndices.push_back(idx1);

		}
	};

	const uint32_t threadCount = nrOfThreads;
	std::cout << "\nStarted with " << threadCount << " thread(s)\n";
	std::vector<std::thread> threads{};

	const uint32_t diff = uint32_t(m_IndexBuffer.size()) / threadCount;
	for (uint32_t i{}; i < threadCount; i++)
	{
		uint32_t start, end;
		start = i * diff;
		end = i * diff + (diff - 1);

		if (start >= uint32_t(m_IndexBuffer.size()))
			start = uint32_t(m_IndexBuffer.size() - 1);

		if (end >= uint32_t(m_IndexBuffer.size()))
			end = uint32_t(m_IndexBuffer.size() - 1);

		threads.push_back(std::thread{GetNeighboursInRange, start, end});
	}

	uint32_t joinedThreads = 0;
	while (joinedThreads != threadCount)
	{
		for (std::thread& thread : threads)
		{
			if (thread.joinable())
			{
				thread.join();
				++joinedThreads;
			}
		}
	}
}

void Mesh::CalculateInnerNeighbours()
{
	std::cout << "\n[Started Calculating Inner Neighbours]\n";
	TimePoint start = std::chrono::high_resolution_clock::now();
	float margin = -0.8f;
	float maxDistance = 20.f;

	for (int i{}; i < m_VertexBuffer.size(); i++)
	{
		if (i % 1000 == 0 || i == m_VertexBuffer.size() - 1)
		{
			printf("\33[2K\r");
			int percentage = int((float(i) / float(m_VertexBuffer.size() - 1)) * 100);
			std::cout << i << " / " << m_VertexBuffer.size() << " " << percentage << "%";
		}

		int j = i + 1;
		while (j < m_VertexBuffer.size())
		{
			float dot = glm::dot(m_VertexDrawData[j].normal, m_VertexDrawData[i].normal);
			if (dot <= margin)
			{
				float distance2 = glm::distance2(m_VertexDrawData[j].position, m_VertexDrawData[i].position);
				if (distance2 <= maxDistance*maxDistance)
				{
					m_VertexBuffer[i].pPulseData->pNeighborIndices.push_back(j);
					m_VertexBuffer[j].pPulseData->pNeighborIndices.push_back(i);
				}
			}

			j++;
		}
	}

	TimePoint end = std::chrono::high_resolution_clock::now();
	auto time = end - start;
	auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time);
	std::cout << "\nCalculating Inner Neighbours Took " << seconds.count() << " seconds\n";
	std::cout << "[Finised Calculating Inner Neighbours]\n";
}

void Mesh::UpdateVertexBuffer(ID3D11DeviceContext* pDeviceContext)
{
	if (m_UpdateSystem == UpdateSystem::GPU)
	{
		m_CudaUpdate.SetAp(m_VertexBuffer);
	}

	for (size_t i = 0; i < m_VertexBuffer.size(); i++)
	{
		if (m_VertexBuffer[i].actionPotential <= m_APMinValue)
		{
			m_VertexDrawData[i].apVisualization = 0;
		}
		else
		{
			float dist = m_APMaxValue - m_APMinValue;

			m_VertexDrawData[i].apVisualization = (m_VertexBuffer[i].actionPotential - m_APMinValue) / dist;
		}
	}

	D3D11_MAPPED_SUBRESOURCE resource;
	pDeviceContext->Map(m_pVertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
	memcpy(resource.pData, m_VertexDrawData.data(), m_VertexDrawData.size() * sizeof(VertexInput));
	pDeviceContext->Unmap(m_pVertexBuffer, 0);
}

void Mesh::CreateIndexForVertices()
{
}

void Mesh::LoadPlotData(int nrOfValuesAPD)
{
	//function to calculate near values of APD(Time) Plot
	//y = 15.311ln(x) + 219.77
	//function to calculate ln(x)
	//ln(x) = log(x) / log(2.71828)
	m_APDPlot.clear();
	m_APDPlot.resize(nrOfValuesAPD);

	for (int x{}; x < nrOfValuesAPD; x++)
	{
		float lnX = logf(float(x)) / logf(2.71828f);
		float value = 15.311f * lnX + 219.77f;
		m_APDPlot[x] = std::chrono::milliseconds(static_cast<long long>(value));
	}

	float diastolicInterval = float(m_DiastolicInterval.count());
	size_t idx = size_t(diastolicInterval);
	if (idx > 0 && idx < m_APDPlot.size() && (idx + 1) < m_APDPlot.size())
	{
		float value1 = float(m_APDPlot[idx].count());
		float value2 = float(m_APDPlot[idx + 1].count());

		float t = diastolicInterval - int(idx);

		m_APD = value1 + t * (value2 - value1);
	}

	//function to calculate near values of the AP(ms) Plot
	//y = -0.0005x² - 0.0187x + 32.118
	m_APThreshold = 0.f;

	m_APPlot.clear();
	m_APPlot.resize((size_t(m_APD) + size_t(1)));

	float minValue = FLT_MAX;
	float maxValue = FLT_MAX * -1;

	for (int x{}; float(x) < m_APD; x++)
	{
		float value = -0.0005f * powf(float(x), 2) - (0.0187f * float(x)) + 32.118f;
		m_APPlot[x] = value;

		if (minValue > value)
			minValue = value;

		if (maxValue < value)
			maxValue = value;
	}

	m_APMinValue = minValue;
	m_APMaxValue = maxValue;

	//function to calculate near value of CV(DI)
	//y = -0.0024x² + 0.6514x + 13.869
	float DI = float(m_DiastolicInterval.count());
	m_ConductionVelocity = -0.0024f * powf(DI, 2) + 0.6514f * DI + 13.869f;
}

void Mesh::CreateEffect(ID3D11Device* pDevice)
{
	m_pEffect = new BaseEffect(pDevice, L"Resources/Shader/PosCol.fx");
}

//Early versions of pulse propogation
#pragma region OldVersion

//Pulse Simulations
//void Mesh::UpdateMesh(ID3D11DeviceContext* pDeviceContext, float deltaTime)
//{
//	if (m_VerticesToUpdate.empty() && m_NeighboursToUpdate.empty())
//		return;
//
//	//Loop over all the vertices that have a pulse going through and update them
//	//If the pulse zeros out, mark them to remove them from the list.
//	std::vector<VertexInput*> verticesToRemove{};
//	for (VertexInput* vertex : m_VerticesToUpdate)
//	{
//		if (vertex->apVisualization > 0.f)
//			vertex->apVisualization -= deltaTime;
//		else
//		{
//			vertex->apVisualization = 0.f;
//			verticesToRemove.push_back(vertex);
//		}
//	}
//
//	//Remove the vertices with no pulse from the list.
//	for (VertexInput* vertex : verticesToRemove)
//	{
//		m_VerticesToUpdate.erase(vertex);
//	}
//
//	//Clear the vector to be reused and update all the neighbour vertices.
//	verticesToRemove.clear();
//	for (VertexInput* vertex : m_NeighboursToUpdate)
//	{
//		vertex->timeToTravel -= deltaTime;
//		if (vertex->timeToTravel <= 0.f)
//		{
//			verticesToRemove.push_back(vertex);
//			PulseVertex(vertex, pDeviceContext, false);
//		}
//	}
//
//	//Remove the neighbour vertices with from the list.
//	for (VertexInput* vertex : verticesToRemove)
//	{
//		m_NeighboursToUpdate.erase(vertex);
//	}
//
//	UpdateVertexBuffer(pDeviceContext);
//}

//void Mesh::PulseVertex(uint32_t index, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer)
//{
//	if (!m_VertexBuffer.empty() && index >= 0 && index < m_VertexBuffer.size())
//	{
//		PulseVertex(&m_VertexBuffer[index], pDeviceContext, updateVertexBuffer);
//	}
//}
//
//void Mesh::PulseVertex(VertexInput* vertex, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer)
//{
//	if (vertex)
//	{
//		if (vertex->apVisualization <= 0.3f)
//		{
//			vertex->apVisualization = 1;
//			m_VerticesToUpdate.insert(vertex);
//			PulseNeighbours(*vertex);
//		}
//	}
//
//	if (updateVertexBuffer)
//		UpdateVertexBuffer(pDeviceContext);
//}

//void Mesh::UpdateMeshV2(ID3D11DeviceContext* pDeviceContext, float deltaTime)
//{
//	for (VertexInput& vertex : m_VertexBuffer)
//	{
//		if (vertex.IsPulsed())
//		{
//			vertex.apVisualization -= deltaTime;
//		}
//		else if (IsAnyNeighbourActive(vertex))
//		{
//			bool vertexFound{ false };
//			float closestTime{ FLT_MAX };
//
//			for (uint32_t index : vertex.neighbourIndices)
//			{
//				VertexInput& neighbourVertex = m_VertexBuffer[index];
//				if (neighbourVertex.IsPulsed())
//				{
//					float distance = glm::distance(vertex.position, neighbourVertex.position);
//					float time = distance / neighbourVertex.propogationSpeed;
//					if (time < closestTime)
//					{
//						closestTime = time;
//						vertexFound = true;
//					}
//				}
//			}
//
//			if (vertexFound)
//			{
//				auto it = m_VerticesToUpdateV2.find(&vertex);
//				if (it == m_VerticesToUpdateV2.end())
//					m_VerticesToUpdateV2.insert(std::pair<VertexInput*, float>(&vertex, closestTime));
//			}
//		}
//	}
//
//	std::vector<VertexInput*> verticesToRemove{};
//	for (std::pair<VertexInput* const, float>&  pair : m_VerticesToUpdateV2)
//	{
//		pair.second -= deltaTime;
//
//		if (pair.second <= 0.f)
//		{
//			PulseVertexV2(pair.first, pDeviceContext);
//			verticesToRemove.push_back(pair.first);
//		}
//	}
//
//	for (VertexInput* vertex : verticesToRemove)
//	{
//		m_VerticesToUpdateV2.erase(vertex);
//	}
//}
//
//void Mesh::PulseVertexV2(uint32_t index, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer)
//{
//	if (!m_VertexBuffer.empty() && index >= 0 && index < m_VertexBuffer.size())
//	{
//		PulseVertexV2(&m_VertexBuffer[index], pDeviceContext, updateVertexBuffer);
//	}
//}
//
//void Mesh::PulseVertexV2(VertexInput* vertex, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer)
//{
//	if (vertex)
//	{
//		if (!vertex->IsPulsed())
//		{
//			vertex->apVisualization = 1;
//		}
//	}
//
//	if (updateVertexBuffer)
//		UpdateVertexBuffer(pDeviceContext);
//}

//void Mesh::PulseNeighbours(const VertexInput& vertex)
//{
//	for (uint32_t neighbourIndex : vertex.neighbourIndices)
//	{
//		VertexInput& neighbourVertex = m_VertexBuffer[neighbourIndex];
//		float distance = glm::distance(vertex.position, neighbourVertex.position);
//		neighbourVertex.timeToTravel = distance / neighbourVertex.propogationSpeed;
//
//		m_NeighboursToUpdate.insert(&neighbourVertex);
//	}
//}

//bool Mesh::IsAnyNeighbourActive(const VertexInput& vertex)
//{
//	for (uint32_t index : vertex.neighbourIndices)
//	{
//		VertexInput& neighbourVertex = m_VertexBuffer[index];
//		if (neighbourVertex.IsPulsed())
//			return true;
//	}
//
//	return false;
//}
#pragma endregion