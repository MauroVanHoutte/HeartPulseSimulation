#pragma once
#include "glm.hpp"
#include <vector>
#include "Cell.h"
#include "Application.h"

class Renderer;
class SDL2Renderer;
class Grid;
struct GLFWwindow;

class SDL2Application final : public Application
{
public:
	SDL2Application(int cellSize);
	SDL2Application(const SDL2Application& other) = delete;
	SDL2Application(SDL2Application&& other) = delete;
	SDL2Application& operator=(const SDL2Application& other) = delete;
	SDL2Application& operator=(SDL2Application&& other) = delete;
	virtual ~SDL2Application() = default;

	void SetTickDelay(float seconds);

private:
	Grid* m_pGrid;
	int m_CellSize;
	float m_TickDelay;
	float m_CurrentDelay;
	float m_TickDelayIncrease;
	bool m_RunningSimulation;
	static bool m_IsRunning;

	SDL2Renderer* m_pSDLRenderer;

	virtual bool Initialize() override;
	virtual void PostInitialize() override;
	virtual void HandleInput() override;
	virtual void Update(float deltaTime) override;
	virtual void Cleanup() override;

	void ClickedOnCell(const glm::ivec2& position);
	void RunSimulation();
	int GetNrOfAliveNeighbours(const std::vector<Cell>& grid, int index, int gridWidth, int gridHeight);
	bool ValidIndex(int idx, int arraySize);
	bool OnSameRow(int y1, int y2, int height);

	void ToggleRunningSimulation();
	void IncreaseTickDelay(float delay);
};
