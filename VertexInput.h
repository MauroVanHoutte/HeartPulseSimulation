#pragma once
#include "glm.hpp"
#include <set>

struct VertexInput
{
	enum class State
	{
		Waiting,	//The vertex does not have a pulse running throug it
		Receiving,	//The vertex has an incoming pulse.
		APD,		//The vertex is in it's Action Potential Duration(APD)
		DI			//The vertex is in it's Diastolic Interval(DI)
	};

	VertexInput(const glm::fvec3& position,
		const glm::fvec3& color1,
		const glm::fvec3& color2,
		const glm::fvec3& normal,
		const glm::fvec2& uv,
		uint32_t index)
		: position(position)
		, color1(color1)
		, color2(color2)
		, normal(normal)
		, uv(uv)
		, tangent({ 0, 0, 0 })
		, apVisualization(0.f)
		, state{ State::Waiting }
		, index(index)
		, actionPotential{ 0.f }
		, timePassed(0.f)
		, timeToTravel(0.f)
		, fibreAssigned{ false }
		, fibreDirection{ 0, 0, 0 }
		, neighbourIndices({})
	{
	}

	VertexInput()
		: position{}
		, color1{ 1, 1, 1 }
		, color2{ 0, 0, 0 }
		, normal{ 1, 0, 0 }
		, uv{}
		, tangent{ 1, 0, 0 }
		, apVisualization{}
		, state{ State::Waiting }
		, index{}
		, actionPotential{ 0.f }
		, timePassed(0.f)
		, timeToTravel{}
		, fibreAssigned{ false }
		, fibreDirection{ 0, 0, 0 }
		, neighbourIndices{}
	{
	}

	//Members part of input layout
	glm::fvec3 position;					//World position
	glm::fvec3 color1;						//Non-pulsed color
	glm::fvec3 color2;						//Pulsed color
	glm::fvec3 normal;						//World normal
	glm::fvec3 tangent;						//World tangent
	glm::fvec2 uv;							//UV coordinate
	float apVisualization;					//[0, 1] value to visualize pulse

	//Members not part of input layout
	State state;							//Current state of the vertex
	uint32_t index;							//Index of the vertex (used in optimization)
	float actionPotential;					//Current action potential (in mV)
	float timePassed;						//Time passed in different states
	float timeToTravel;						//The time before activating this vertex
	bool fibreAssigned;						//A boolean indicating if a fibre was assigned to this vertex
	glm::fvec3 fibreDirection;				//The direction of the heart fibre at this point
	std::set<uint32_t> neighbourIndices;	//The indices of the neighbouring vertices

	//Operator overloading
	bool operator==(const VertexInput& other)
	{
		return this->position == other.position;
	}

	friend bool operator==(const VertexInput& rhs, const VertexInput& lhs)
	{
		return rhs.position == lhs.position;
	}
};