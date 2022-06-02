#pragma once
#include "glm.hpp"
#include <vector>
#include <memory>

enum class State
{
	Waiting,	//The vertex does not have a pulse running throug it
	Receiving,	//The vertex has an incoming pulse.
	APD,		//The vertex is in it's Action Potential Duration(APD)
	DI			//The vertex is in it's Diastolic Interval(DI)
};

struct VertexInput
{
	VertexInput(const glm::fvec3& position,
		const glm::fvec3& color1,
		const glm::fvec3& color2,
		const glm::fvec3& normal,
		const glm::fvec2& uv,
		size_t )
		: position(position)
		, color1(color1)
		, color2(color2)
		, normal(normal)
		, uv(uv)
		, tangent({ 0, 0, 0 })
		, apVisualization(0.f)
	{
	}

	VertexInput()
		: position{0, 0, 0}
		, color1{ 1, 1, 1 }
		, color2{ 0, 0, 0 }
		, normal{ 1, 0, 0 }
		, uv{}
		, tangent{ 1, 0, 0 }
		, apVisualization{}
	{
	}

	//Members part of input layout
	glm::fvec3 position{};						//World position
	glm::fvec3 color1;						//Non-pulsed color
	glm::fvec3 color2;						//Pulsed color
	glm::fvec3 normal;						//World normal
	glm::fvec3 tangent;						//World tangent
	glm::fvec2 uv;							//UV coordinate
	float apVisualization;					//[0, 1] value to visualize pulse

};

struct PulseData
{
	PulseData(const glm::fvec3& position)
		: position{ position }
	{
	};

	std::vector<uint32_t> pNeighborIndices{};		//indices of connected cells
	uint32_t* pNeighborIndicesRaw;					//needed for cuda update
	uint32_t neighborIndicesSize;
	glm::fvec3 fibreDirection{};				//The direction of the heart fibre at this point
	glm::fvec3 position{};						//World position
};

struct VertexData
{
	VertexData(const glm::fvec3& position)
	{
		pPulseData = new PulseData{position};
	}
	VertexData()
	{
		pPulseData = new PulseData{ glm::fvec3{ 0, 0, 0 } };
	}

	~VertexData()
	{
		if (pPulseData != nullptr) 
			delete pPulseData;
	}

	VertexData(const VertexData& other)
		: actionPotential{other.actionPotential}
		, timePassed{other.timePassed}
		, timeToTravel{other.timeToTravel}
		, state{other.state}
		, pPulseData{ new PulseData{other.pPulseData->position} }
	{
		pPulseData->fibreDirection = other.pPulseData->fibreDirection;
		pPulseData->pNeighborIndices = other.pPulseData->pNeighborIndices;

	}

	VertexData(const VertexData&& other)
		: actionPotential{ other.actionPotential }
		, timePassed{ other.timePassed }
		, timeToTravel{ other.timeToTravel }
		, state{ other.state }
		, pPulseData{ new PulseData{other.pPulseData->position} }
	{
		pPulseData->fibreDirection = other.pPulseData->fibreDirection;
		pPulseData->pNeighborIndices = other.pPulseData->pNeighborIndices;

	}

	VertexData& operator=(const VertexData& other)
	{
		actionPotential = other.actionPotential;
		timePassed = other.timePassed;
		timeToTravel = other.timeToTravel;
		state = other.state;

		delete pPulseData;

		pPulseData = new PulseData{other.pPulseData->position};
		
		pPulseData->fibreDirection = other.pPulseData->fibreDirection;
		pPulseData->pNeighborIndices = other.pPulseData->pNeighborIndices;


		return *this;
	}


	float actionPotential{};				//Current action potential (in mV)
	float timePassed{};						//Time passed in different states
	float timeToTravel{};					//The time before activating this vertex
	State state{};							//Current state of the vertex
	PulseData* pPulseData{};//data used when pulsed, this data is not accesed often so pointer reduces the size of the struct

	bool operator==(const VertexData& other)
	{
		return this->pPulseData->position == other.pPulseData->position;
	}

	friend bool operator==(const VertexData& rhs, const VertexData& lhs)
	{
		return rhs.pPulseData->position == lhs.pPulseData->position;
	}
};

