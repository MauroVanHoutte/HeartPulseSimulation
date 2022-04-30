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
	m_TasksFinished.resize(ThreadManager::GetInstance()->GetNrThreads(), false);
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
	case FileType::VTK:
		LoadMeshFromVTK();
		break;
	case FileType::BIN:
		LoadMeshFromBIN();
		break;
	case FileType::PTS: 
		LoadMeshFromPTS();
		break;
	default: ;
	}

	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		CreateDirectXResources(pDevice, m_VertexBuffer, m_IndexBuffer);
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
			pDeviceContext->DrawIndexed(m_AmountIndices, 0, 0);
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

const std::vector<VertexInput>& Mesh::GetVertexBuffer() const
{
	return m_VertexBuffer;
}

std::vector<VertexInput>& Mesh::GetVertexBufferReference()
{
	return m_VertexBuffer;
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

void Mesh::SetVertexBuffer(ID3D11DeviceContext* pDeviceContext, const std::vector<VertexInput>& vertexBuffer)
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

		for (const VertexInput& vertex : m_VertexBuffer)
		{
			fileStream.write((const char*)&vertex.position, sizeof(glm::fvec3));
			fileStream.write((const char*)&vertex.normal, sizeof(glm::fvec3));
			fileStream.write((const char*)&vertex.color1, sizeof(glm::fvec3));
			fileStream.write((const char*)&vertex.color2, sizeof(glm::fvec3));
			fileStream.write((const char*)&vertex.tangent, sizeof(glm::fvec3));
			fileStream.write((const char*)&vertex.uv, sizeof(glm::fvec2));

			const size_t nrOfNeighbours = vertex.neighbourIndices.size();
			fileStream.write((const char*)&nrOfNeighbours, sizeof(size_t));

			for (const uint32_t& index : vertex.neighbourIndices)
			{
				fileStream.write((const char*)&index, sizeof(uint32_t));
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

		for (const VertexInput& vertex : m_VertexBuffer)
		{
			fileStream.write((const char*)&vertex.fibreDirection, sizeof(glm::fvec3));
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
			std::vector<VertexInput>::iterator it = std::find_if(m_VertexBuffer.begin(), m_VertexBuffer.end(), [point, epsilon](const VertexInput& vertex)
				{
					return (!vertex.fibreAssigned && glm::epsilonEqual(point, vertex.position, epsilon).y);
				}
			);

			while (it == m_VertexBuffer.end())
			{
				epsilon += epsilonIncrease;

				if (epsilon > epsilonMax)
					break;

				it = std::find_if(m_VertexBuffer.begin(), m_VertexBuffer.end(), [point, epsilon](const VertexInput& vertex)
					{
						return (!vertex.fibreAssigned && glm::epsilonEqual(point, vertex.position, epsilon).y);
					}
				);
			}

			while (it != m_VertexBuffer.end())
			{
				it->fibreDirection = fibres[i];
				it->fibreAssigned = true;

				++it;
				it = std::find_if(it, m_VertexBuffer.end(), [point, epsilon](const VertexInput& vertex)
					{
						return (!vertex.fibreAssigned && glm::epsilonEqual(point, vertex.position, epsilon).y);
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
				m_VertexBuffer[i].fibreDirection = fibre;
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

void Mesh::UpdateMeshV3(ID3D11DeviceContext* pDeviceContext, float deltaTime)
{
	if (m_IsPulsing)
	{
		m_PulseCounter += deltaTime;
		if (m_PulseCounter > 1/m_PulseRate)
		{
			m_PulseCounter -= 1 / m_PulseRate;
			PulseVertexV3(uint32_t(0), pDeviceContext);
		}
	}

	float dist = (m_APMaxValue - m_APMinValue);
	float deltaTimeInMs = deltaTime * 1000.f;

	auto startTime = std::chrono::high_resolution_clock::now();

	switch (m_UpdateSystem)
	{
	case UpdateSystem::Serial:
		UpdateSerial(deltaTimeInMs, deltaTime, dist, pDeviceContext);
		break;
	case UpdateSystem::Multithreaded:
		UpdateThreaded(deltaTimeInMs, deltaTime, dist, pDeviceContext);
		break;
	case UpdateSystem::GPU:
		UpdateGPU(deltaTimeInMs, deltaTime, dist, pDeviceContext);
		break;
	}

	if(m_Benchmarking)
		Benchmarker::GetInstance()->AddDuration(std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startTime).count());

	UpdateVertexBuffer(pDeviceContext);
}

void Mesh::UpdateSerial(float deltaTimeInMs, float deltaTime, float dist, ID3D11DeviceContext* pDeviceContext)
{
	for (VertexInput& vertex : m_VertexBuffer)
	{
		switch (vertex.state)
		{
		case VertexInput::State::APD:
		{
			vertex.timePassed += deltaTimeInMs;

			int idx = int(vertex.timePassed);

			if (!m_APPlot.empty() && idx > 0 && idx < m_APPlot.size() && (size_t(idx) + size_t(1)) < m_APPlot.size())
			{
				float value1 = m_APPlot[idx];
				float value2 = m_APPlot[(size_t(idx) + size_t(1))];
				float t = vertex.timePassed - idx;

				float lerpedValue = value1 + t * (value2 - value1);

				float valueRange01 = (lerpedValue - m_APMinValue) / dist;

				vertex.actionPotential = lerpedValue;
				vertex.apVisualization = valueRange01;
			}

			if (vertex.timePassed >= m_APD)
			{
				vertex.timePassed = 0.f;
				vertex.state = VertexInput::State::DI;
				vertex.apVisualization = 0.f;
			}

			break;
		}
		case VertexInput::State::DI:
			vertex.timePassed += deltaTimeInMs;

			if (vertex.timePassed >= m_DiastolicInterval.count())
			{
				vertex.timePassed = 0.f;
				vertex.state = VertexInput::State::Waiting;
			}
			break;

		case VertexInput::State::Receiving:
			vertex.timeToTravel -= deltaTime;
			if (vertex.timeToTravel <= 0.f)
			{
				vertex.state = VertexInput::State::Waiting;
				PulseVertexV3(&vertex, pDeviceContext, false);
			}
			break;
		}
	}
}

void Mesh::UpdateThreaded(float deltaTimeInMs, float deltaTime, float dist, ID3D11DeviceContext* pDeviceContext)
{
	int nrThreads = int(ThreadManager::GetInstance()->GetNrThreads());
	int jobSize = int(ceil(m_VertexBuffer.size() / float(nrThreads)));

	int firstVertex = 0;

	for (int i = 0; i < nrThreads; i++)
	{
		ThreadManager::GetInstance()->AddJobFunction(std::bind(&Mesh::UpdateVertexCluster, this, deltaTimeInMs, deltaTime, dist, pDeviceContext, firstVertex, jobSize, i));
		firstVertex += int(jobSize);
	}

	while (!std::all_of(m_TasksFinished.begin(), m_TasksFinished.end(), [](bool curr) {return curr; }))
	{
		std::this_thread::sleep_for(std::chrono::microseconds(5));
	}
}

void Mesh::UpdateGPU(float deltaTimeInMs, float deltaTime, float dist, ID3D11DeviceContext* pDeviceContext)
{
	m_CudaUpdate.Update(m_VertexBuffer, m_APPlot, m_APMinValue, m_APD, float(m_DiastolicInterval.count()), deltaTimeInMs, deltaTime, dist);

	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	for (VertexInput& vertex : m_VertexBuffer)
	{
		if (vertex.state == VertexInput::State::Receiving)
		{
			vertex.timeToTravel -= deltaTime;
			if (vertex.timeToTravel <= 0.f)
			{
				vertex.state = VertexInput::State::Waiting;
				PulseVertexV3(&vertex, pDeviceContext, false);
			}
		}
	}


	UpdateVertexBuffer(pDeviceContext);
}

void Mesh::SetUpdateSystem(UpdateSystem system)
{
	m_UpdateSystem = system;
}


//void Mesh::UpdateGPU(float deltaTimeInMs, float deltaTime, float dist, ID3D11DeviceContext* pDeviceContext)
//{
//
//}

void Mesh::UpdateVertexCluster(float deltaTimeInMs, float deltaTime, float dist, ID3D11DeviceContext* pDeviceContext, int firstVertex, int vertexCount, int taskId)
{
	TIME();
	for (size_t i = firstVertex; i < firstVertex+vertexCount; i++)
	{
		if (i == m_VertexBuffer.size())
		{
			break;
		}

		switch (m_VertexBuffer[i].state)
		{
		case VertexInput::State::APD:
		{
			m_VertexBuffer[i].timePassed += deltaTimeInMs;

			int idx = int(m_VertexBuffer[i].timePassed);

			if (!m_APPlot.empty() && idx > 0 && idx < m_APPlot.size() && (size_t(idx) + size_t(1)) < m_APPlot.size())
			{
				float value1 = m_APPlot[idx];
				float value2 = m_APPlot[(size_t(idx) + size_t(1))];
				float t = m_VertexBuffer[i].timePassed - idx;

				float lerpedValue = value1 + t * (value2 - value1);

				float valueRange01 = (lerpedValue - m_APMinValue) / dist;

				m_VertexBuffer[i].actionPotential = lerpedValue;
				m_VertexBuffer[i].apVisualization = valueRange01;
			}

			if (m_VertexBuffer[i].timePassed >= m_APD)
			{
				m_VertexBuffer[i].timePassed = 0.f;
				m_VertexBuffer[i].state = VertexInput::State::DI;
				m_VertexBuffer[i].apVisualization = 0.f;
			}

			break;
		}
		case VertexInput::State::DI:
			m_VertexBuffer[i].timePassed += deltaTimeInMs;

			if (m_VertexBuffer[i].timePassed >= m_DiastolicInterval.count())
			{
				m_VertexBuffer[i].timePassed = 0.f;
				m_VertexBuffer[i].state = VertexInput::State::Waiting;
			}
			break;

		case VertexInput::State::Receiving:
			m_VertexBuffer[i].timeToTravel -= deltaTime;
			if (m_VertexBuffer[i].timeToTravel <= 0.f)
			{
				m_VertexBuffer[i].state = VertexInput::State::Waiting;
				PulseVertexV3(&m_VertexBuffer[i], pDeviceContext, false);
			}
			break;
		}
	}

	m_TasksFinished[taskId] = true;
}

void Mesh::PulseVertexV3(uint32_t index, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer)
{
	if (!m_VertexBuffer.empty() && index >= 0 && index < m_VertexBuffer.size())
	{
		PulseVertexV3(&m_VertexBuffer[index], pDeviceContext, updateVertexBuffer);
	}
}

void Mesh::PulseVertexV3(VertexInput* vertex, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer)
{
	if (vertex)
	{
		if (vertex->state == VertexInput::State::Waiting || (vertex->actionPotential < m_APThreshold && vertex->state == VertexInput::State::DI))
		{
			vertex->actionPotential = m_APPlot[0];
			vertex->state = VertexInput::State::APD;

			for (uint32_t index : vertex->neighbourIndices)
			{
				VertexInput& neighbourVertex = m_VertexBuffer[index];

				//Potential problem with fibres. c0 is in m/s while the distance is most likely not in meters.
					//This is likely the cause of it.
				float distance = glm::distance(vertex->position, neighbourVertex.position);
				float conductionVelocity = m_ConductionVelocity;

				if (UseFibres())
				{
					float d1 = 1; // parallel with fibre
					float d2 = d1 / 5; // perpendiculat with fibre
					float c0 = 0.6f; // m/s

					glm::fvec3 pulseDirection = glm::normalize(neighbourVertex.position - vertex->position);
					float cosAngle = glm::dot(vertex->fibreDirection, pulseDirection);

					float c = c0 * sqrtf(d2 + (d1 - d2) * powf(cosAngle, 2));
					conductionVelocity = c * 100;
					//std::cout << c << "\n";
				}

				float travelTime = distance / conductionVelocity;
				float timeToRecover = m_DiastolicInterval.count() - neighbourVertex.timePassed;

				if (neighbourVertex.state == VertexInput::State::Waiting || (neighbourVertex.state == VertexInput::State::DI && timeToRecover < travelTime))
				{
					neighbourVertex.timeToTravel = travelTime;
					neighbourVertex.state = VertexInput::State::Receiving;
				}
			}
		}
	}

	if (updateVertexBuffer)
		UpdateVertexBuffer(pDeviceContext);
}

void Mesh::PulseMesh(ID3D11DeviceContext* pDeviceContext)
{
	for (int i{}; i < m_VertexBuffer.size(); i++)
	{
		PulseVertexV3(i, pDeviceContext, false);
	}
	UpdateVertexBuffer(pDeviceContext);
}

void Mesh::ClearPulse(ID3D11DeviceContext* pDeviceContext)
{
	for (VertexInput& vertex : m_VertexBuffer)
	{
		vertex.apVisualization = 0.f;
		vertex.state = VertexInput::State::Waiting;
		vertex.timePassed = 0.f;
	}

	UpdateVertexBuffer(pDeviceContext);
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
	bd.ByteWidth = sizeof(VertexInput) * (uint32_t)vertices.size();
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bd.MiscFlags = 0;
	D3D11_SUBRESOURCE_DATA initData = { 0 };
	initData.pSysMem = vertices.data();
	result = pDevice->CreateBuffer(&bd, &initData, &m_pVertexBuffer);
	if (FAILED(result))
		return result;

	//Create index buffer
	m_AmountIndices = (uint32_t)indices.size();
	bd.Usage = D3D11_USAGE_IMMUTABLE;
	bd.ByteWidth = sizeof(uint32_t) * m_AmountIndices;
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
	TIME();
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

			uint32_t count = 0;
			for (const objl::Vertex& vertex : loader.LoadedVertices)
			{
				m_VertexBuffer.push_back({
					{ vertex.Position.X, vertex.Position.Y, vertex.Position.Z },
					color1,
					color2,
					{ vertex.Normal.X, vertex.Normal.Y, vertex.Normal.Z },
					{ vertex.TextureCoordinate.X, vertex.TextureCoordinate.Y },
					count
				});

				++count;
			}

			for (unsigned int index : loader.LoadedIndices)
			{
				m_IndexBuffer.push_back(index);
			}

			m_AmountIndices = uint32_t(m_IndexBuffer.size());
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
		m_VertexBuffer.resize(vertCount);

		int index = 0;
		while (!vertexStream.eof())
		{
			if (index >= vertCount)
				break;

			VertexInput vertex{};
			vertex.index = index;

			vertexStream >> vertex.position.x >> vertex.position.y >> vertex.position.z;

			vertex.position /= 1000.f;
			m_VertexBuffer[index] = vertex;
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
		m_VertexBuffer.resize(vertCount);

		int index = 0;
		while (!vertexStream.eof())
		{
			if (index >= vertCount)
				break;

			VertexInput vertex{};
			vertex.index = index;

			vertexStream >> vertex.position.x >> vertex.position.y >> vertex.position.z;

			vertex.position /= 1000.f;
			m_VertexBuffer[index] = vertex;
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
	TIME();
	std::ifstream fileStream{ m_PathName, std::ios::in | std::ios::binary };
	if (fileStream.is_open())
	{
		//Read in the indexbuffer
		m_IndexBuffer.clear();
		size_t nrOfIndices{};
		fileStream.read((char*)&nrOfIndices, sizeof(size_t));
		m_AmountIndices = uint32_t(nrOfIndices);

		m_IndexBuffer.resize(nrOfIndices);
		fileStream.read((char*)&m_IndexBuffer[0], sizeof(uint32_t) * nrOfIndices);

		//Read in the vertices
		m_VertexBuffer.clear();
		size_t nrOfVertices{};
		fileStream.read((char*)&nrOfVertices, sizeof(size_t));

		m_VertexBuffer.resize(nrOfVertices);
		for (VertexInput& vertex : m_VertexBuffer)
		{
			fileStream.read((char*)&vertex.position, sizeof(glm::fvec3));
			fileStream.read((char*)&vertex.normal, sizeof(glm::fvec3));
			fileStream.read((char*)&vertex.color1, sizeof(glm::fvec3));
			fileStream.read((char*)&vertex.color2, sizeof(glm::fvec3));
			fileStream.read((char*)&vertex.tangent, sizeof(glm::fvec3));
			fileStream.read((char*)&vertex.uv, sizeof(glm::fvec2));

			size_t nrOfNeighbours{};
			fileStream.read((char*)&nrOfNeighbours, sizeof(size_t));

			for (size_t i{}; i < nrOfNeighbours; i++)
			{
				uint32_t neighbourIndex{};
				fileStream.read((char*)&neighbourIndex, sizeof(uint32_t));

				vertex.neighbourIndices.insert(neighbourIndex);
			}
		}

		CreateIndexForVertices();

		LoadCachedFibres();

		std::string name = "Vertex Buffer " + m_PathName;
		Logger::Get().LogBuffer<VertexInput>(m_VertexBuffer, name);
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
			int index0 = m_IndexBuffer[i];
			int index1 = m_IndexBuffer[i + 1];
			int index2 = m_IndexBuffer[i + 2];
			VertexInput& vertex0 = m_VertexBuffer[index0];
			VertexInput& vertex1 = m_VertexBuffer[index1];
			VertexInput& vertex2 = m_VertexBuffer[index2];

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

	for (size_t i{}; i < m_IndexBuffer.size(); i++)
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

		VertexInput vertex = m_VertexBuffer[index];

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
	std::cout << "Removing Duplicate Indices\n";
	std::vector<uint32_t> indicesToRemove{};
	std::vector<VertexInput>::iterator it = std::remove_if(m_VertexBuffer.begin(), m_VertexBuffer.end(), [&indicesToRemove, this](const VertexInput& vertex)
		{
			std::vector<uint32_t>::iterator itFind = std::find(m_IndexBuffer.begin(), m_IndexBuffer.end(), vertex.index);

			bool shouldRemove = itFind == m_IndexBuffer.end();
			if (shouldRemove)
				indicesToRemove.push_back(vertex.index);

			return shouldRemove;
		});

	m_VertexBuffer.erase(it, m_VertexBuffer.end());

	std::cout << "Reconstructing Index Buffer\n";
	std::vector<uint32_t> indexBufferSwap{m_IndexBuffer};
	for (uint32_t i{}; i < indicesToRemove.size(); i++)
	{
		uint32_t indexToRemove = indicesToRemove[i];

		for (uint32_t j{}; j < m_IndexBuffer.size(); j++)
		{
			if (indexBufferSwap[j] > indexToRemove)
			{
				--m_IndexBuffer[j];
			}
		}
	}

	std::cout << "Reassigning Indices To Vertices\n";
	for (int i{}; i < m_VertexBuffer.size(); i++)
	{
		m_VertexBuffer[i].index = i;
	}

	// 1 3 2 2 4 3 7 9 8
	// [1] [2] [3] [4] [5] [6] [7] [8] [9]
	// Remove [5]			v	v	v	v
	// [1] [2] [3] [4]	   [5] [6] [7] [8] //All vertices with original index > 5, decrement
	// Remove [6]			v		v	v
	// [1] [2] [3] [4]	   [5]     [6] [7] //All vertices with original index > 6, decrement
}

void Mesh::OptimizeVertexAndIndexBuffer()
{
	std::unordered_map<VertexInput, uint32_t> VerticesMap{};
	std::vector<uint32_t> uniqueIndices{};
	std::vector<VertexInput> uniqueVertices{};

	for (const auto& vertex : m_VertexBuffer)
	{
		if (VerticesMap.count(vertex) == 0)
		{
			VerticesMap[vertex] = uint32_t(uniqueVertices.size());
			uniqueVertices.push_back(vertex);
		}

		uniqueIndices.push_back(VerticesMap[vertex]);
	}

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

			m_VertexBuffer[idx1].neighbourIndices.insert({ idx2, idx3 });
			m_VertexBuffer[idx2].neighbourIndices.insert({ idx1, idx3 });
			m_VertexBuffer[idx3].neighbourIndices.insert({ idx2, idx1 });

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

		VertexInput& vertex1 = m_VertexBuffer[i];
		std::vector<VertexInput>::iterator it = m_VertexBuffer.begin() + (i + 1);
		while (it != m_VertexBuffer.end())
		{
			VertexInput& vertex2 = *it;


			float dot = glm::dot(vertex1.normal, vertex2.normal);
			if (dot <= margin)
			{
				float distance = glm::distance(vertex1.position, vertex2.position);
				if (distance <= maxDistance)
				{
					vertex1.neighbourIndices.insert(vertex2.index);
					vertex2.neighbourIndices.insert(vertex1.index);
				}
			}

			it++;
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
	//TIME();
	D3D11_MAPPED_SUBRESOURCE resource;
	pDeviceContext->Map(m_pVertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
	memcpy(resource.pData, m_VertexBuffer.data(), m_VertexBuffer.size() * sizeof(VertexInput));
	pDeviceContext->Unmap(m_pVertexBuffer, 0);
}

void Mesh::CreateIndexForVertices()
{
	for (int i{}; i < m_VertexBuffer.size(); i++)
	{
		m_VertexBuffer[i].index = i;
	}
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