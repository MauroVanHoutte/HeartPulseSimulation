#pragma once
#include "glm.hpp"
#include "BaseEffect.h"

#include <set>
#include <map>
#include <vector>
#include <chrono>
#include <mutex>

#include "VertexInput.h"
#include "GpuUpdate.h"

#pragma warning(push)
#pragma warning(disable:4616)
#pragma warning(disable:4201)
#define GLM_ENABLE_EXPERIMENTAL
#include "3rdParty/glm/gtx/hash.hpp"
#include <gtc/type_ptr.hpp>
#pragma warning(pop)

enum class FileType
{
	OBJ,
	VTK,
	BIN,
	PTS
};

enum class UpdateSystem
{
	Serial,
	Multithreaded,
	GPU
};



namespace std {
	template<> struct hash<VertexInput> {
		size_t operator()(VertexInput const& vertex) const {
			return (hash<glm::vec3>()(vertex.position));
		}
	};
}

class Mesh
{
public:
	Mesh(ID3D11Device* pDevice, const std::vector<VertexInput>& vertices, const std::vector<uint32_t>& indices);
	Mesh(ID3D11Device* pDevice, const std::string& filepath, bool skipOptimization = false, FileType fileType = FileType::OBJ, int nrOfThreads = 1);
	Mesh(const Mesh& other) = delete;
	Mesh(Mesh&& other) = delete;
	Mesh& operator=(const Mesh& other) = delete;
	Mesh& operator=(Mesh&& other) = delete;
	~Mesh();

	void Render(ID3D11DeviceContext* pDeviceContext, const float* worldViewProjMatrix, const float* inverseView);
	void UpdateVertexBuffer(ID3D11DeviceContext* pDeviceContext);

	void UpdateMeshV3(ID3D11DeviceContext* pDeviceContext, float deltaTime);
	void UpdateSerial(float deltaTimeInMs, float deltaTime, float dist, ID3D11DeviceContext* pDeviceContext);
	void UpdateThreaded(float deltaTimeInMs, float deltaTime, float dist, ID3D11DeviceContext* pDeviceContext);
	void UpdateGPU(float deltaTimeInMs, float deltaTime, float dist, ID3D11DeviceContext* pDeviceContext);
	void SetUpdateSystem(UpdateSystem system);
	void UpdateVertexCluster(float deltaTimeInMs, float deltaTime, float dist, ID3D11DeviceContext* pDeviceContext, int firstVertex, int vertexCount, int taskId);
	void PulseVertexV3(uint32_t index, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer = true);
	void PulseVertexV3(VertexInput* vertex, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer = true);

	void PulseMesh(ID3D11DeviceContext* pDeviceContext);
	void ClearPulse(ID3D11DeviceContext* pDeviceContext);
	void CalculateNeighbours(int nrOfThreads = 1);
	void CalculateInnerNeighbours();

	const glm::mat4& GetWorldMatrix() const;
	const std::vector<uint32_t>& GetIndexBuffer() const;
	const std::vector<VertexInput>& GetVertexBuffer() const;
	std::vector<VertexInput>& GetVertexBufferReference();

	const std::vector<float>& GetAPPlot() const;
	std::chrono::milliseconds GetDiastolicInterval() const;
	glm::fvec2 GetMinMax() const;
	glm::fvec3 GetScale();
	glm::fvec3 GetTranslation();
	float GetAPD() const;
	void UseFibres(bool useFibres);
	bool UseFibres();
	void UseThreading(bool useThreading);
	bool UsingThreading();
	void ConstantPulsing(bool constantPulsing);
	bool IsPulsing();
	void SetPulseRate(float pulseRate);
	float GetPulseRate();
	void StartBenchmarking(const std::string& name);
	void StopBenchmarking();

	void SetVertexBuffer(ID3D11DeviceContext* pDeviceContext, const std::vector<VertexInput>& vertexBuffer);
	void SetWireframe(bool enabled);
	void SetScale(const glm::fvec3& scale);
	void SetScale(float x, float y, float z);
	void Translate(const glm::fvec3& translation);
	void Translate(float x, float y, float z);
	void SetDiastolicInterval(float diastolicInterval);

	void CreateCachedBinary();
	void CreateCachedFibreBinary();
	void LoadFibreData();							//Should be put in an AssetLoader Class
	void LoadCachedFibres();
private:
	Mesh();

	//----- DirectX -----
	BaseEffect* m_pEffect;
	BaseEffect* m_pOptimizerEffect;
	ID3D11InputLayout* m_pVertexLayout;
	ID3D11Buffer* m_pVertexBuffer;
	ID3D11Buffer* m_pIndexBuffer;
	ID3D11RasterizerState* m_pRasterizerStateWireframe;
	ID3D11RasterizerState* m_pRasterizerStateSolid;
	bool m_WireFrameEnabled;
	bool m_DrawVertex;

	uint32_t m_AmountIndices;
	//-------------------

	//Initialization of mesh
	void LoadMeshFromOBJ(uint32_t nrOfThreads = 1);	//Should be put in an AssetLoader Class
	void LoadMeshFromVTK();							//Should be put in an AssetLoader Class
	void LoadMeshFromPTS();							//Should be put in an AssetLoader Class
	void LoadMeshFromBIN();							//Should be put in an AssetLoader Class
	void CalculateTangents();						//Should be put in an AssetLoader Class
	void OptimizeIndexBuffer();						//Should be put in an AssetLoader Class
	void OptimizeVertexBuffer();					//Should be put in an AssetLoader Class
	void OptimizeVertexAndIndexBuffer();

	bool m_SkipOptimization;						//Should be put in an AssetLoader Class

	void CreateEffect(ID3D11Device* pDevice);
	HRESULT CreateDirectXResources(ID3D11Device* pDevice, const std::vector<VertexInput>& vertices, const std::vector<uint32_t>& indices);


	//Vertex Data
	bool IsAnyNeighbourActive(const VertexInput& vertex);
	void CreateIndexForVertices();

	bool m_FibresLoaded;
	bool m_UseFibres;
	glm::mat4 m_WorldMatrix;
	std::vector<uint32_t> m_IndexBuffer;
	std::vector<VertexInput> m_VertexBuffer;
	std::vector<VertexInput> m_LineBuffer;

	//Plot Data
	void LoadPlotData(int nrOfValuesAPD);
	
	std::chrono::milliseconds m_DiastolicInterval;
	float m_APThreshold;
	float m_APMaxValue;
	float m_APMinValue;
	float m_APD;
	float m_ConductionVelocity;
	std::vector<float> m_APPlot;						// APD (mV) in function of time (ms)
	std::vector<std::chrono::milliseconds> m_APDPlot;	// APD (ms) in function of Diastolic Interval (ms)// DI (ms) in function of Conduction Velocity (cm/s)

	//File Data
	std::string m_PathName;
	std::mutex m_Mutex{};

	//Multithreading
	std::vector<bool> m_TasksFinished{};
	CudaUpdate m_CudaUpdate{};
	UpdateSystem m_UpdateSystem = UpdateSystem::Serial;

	//Constant pulsing
	float m_PulseRate = 1.5f; // in Hz
	float m_PulseCounter = 0.f;
	bool m_IsPulsing = true;
	bool m_Benchmarking = false; 
};

#pragma region OldVersion
//void UpdateMesh(ID3D11DeviceContext* pDeviceContext, float deltaTime);
//void PulseVertex(uint32_t index, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer = true);
//void PulseVertex(VertexInput* vertex, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer = true);

//void UpdateMeshV2(ID3D11DeviceContext* pDeviceContext, float deltaTime);
//void PulseVertexV2(uint32_t index, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer = true);
//void PulseVertexV2(VertexInput* vertex, ID3D11DeviceContext* pDeviceContext, bool updateVertexBuffer = true);
//void PulseNeighbours(const VertexInput& vertex);
#pragma endregion
