#include "SDL2Renderer.h"
#include "SDL.h"

#include <algorithm>
#include <iostream>

SDL2Renderer::SDL2Renderer(const std::string& windowName, int width, int height)
	: m_pGrid(nullptr)
	, m_Window(nullptr)
	, m_Renderer(nullptr)
	, m_WindowName(windowName)
	, m_Width(width)
	, m_Height(height)
	, m_DrawGrid(true)
{
}

bool SDL2Renderer::Initialize()
{
	//Initiliaze SDL & create a window and renderer
	SDL_Init(SDL_INIT_EVERYTHING);
	m_Window = SDL_CreateWindow(m_WindowName.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, m_Width, m_Height, SDL_WINDOW_SHOWN);
	m_Renderer = SDL_CreateRenderer(m_Window, -1, 0);

	//Store the grid to draw later.
	//m_pGrid = grid;

	return true;
}

void SDL2Renderer::Render()
{
	//Set the clear color
	SDL_SetRenderDrawColor(m_Renderer, 50, 50, 50, 255);
	//Clear the render target
	SDL_RenderClear(m_Renderer);

	//Draw objects
	Draw();

	//Swap the buffers
	SDL_RenderPresent(m_Renderer);
}

void SDL2Renderer::Cleanup()
{
}

int SDL2Renderer::GetWindowWidth() const
{
	return m_Width;
}

int SDL2Renderer::GetWindowHeight() const
{
	return m_Height;
}

void SDL2Renderer::SetGrid(Grid* pGrid)
{
	m_pGrid = pGrid;
}

void SDL2Renderer::ToggleGrid()
{
	m_DrawGrid = !m_DrawGrid;
}

void SDL2Renderer::Draw() const
{
	if (!m_pGrid)
		return;

	const std::vector<Cell>& cells = m_pGrid->GetCells();

	//Store the original color and set the draw color to white
	Uint8 r, g, b, a;
	SDL_GetRenderDrawColor(m_Renderer, &r, &g, &b, &a);
	SDL_SetRenderDrawColor(m_Renderer, 255, 255, 255, 255);

	//Loop over all the cells in the grid and draw them.
	//Fill the cell if it's alive otherwhise draw the outline
	std::for_each(cells.begin(), cells.end(), [this](const Cell& cell)
		{
			SDL_Rect rect = { int(cell.position.x * cell.size), int(cell.position.y * cell.size), cell.size, cell.size};
			if (cell.alive)
				SDL_RenderFillRect(m_Renderer, &rect);
			else if (m_DrawGrid)
				SDL_RenderDrawRect(m_Renderer, &rect);
		}
	);

	//Restore the original draw color
	SDL_SetRenderDrawColor(m_Renderer, r, g, b, a);
}
