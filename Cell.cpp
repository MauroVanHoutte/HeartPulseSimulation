#include "Cell.h"
#include <algorithm>

Grid::Grid(int width, int height, int cellSize)
	: m_Width(width)
	, m_Height(height)
	, m_CellSize(cellSize)
{
	//Clamp width and height to not be negative or 0
	NegativeCheck(m_Width);
	NegativeCheck(m_Height);
	NegativeCheck(m_CellSize);

	//Create a grid of cells with size width * height
	m_Cells.reserve(m_Width * m_Height);

	//Loop over the width and height and initialize all the cells with their proper position
	for (int y{ 0 }; y < m_Height; y++)
	{
		for (int x{ 0 }; x < m_Width; x++)
		{
			m_Cells.push_back(Cell{ glm::ivec2{x, y}, m_CellSize });
		}
	}
}

int Grid::GetWidth()
{
	return m_Width;
}

int Grid::GetHeight()
{
	return m_Height;
}

int Grid::GetCellSize()
{
	return m_CellSize;
}

void Grid::ToggleCell(int x, int y)
{
	int idx = x  + y  * m_Width;

	m_Cells[idx].alive = !m_Cells[idx].alive;
}

void Grid::ToggleCell(const glm::ivec2& position)
{
	ToggleCell(position.x, position.y);
}

void Grid::ClearGrid()
{
	std::for_each(m_Cells.begin(), m_Cells.end(), [](Cell& cell)
		{
			if (cell.alive)
				cell.alive = false;
		}
	);
}

const std::vector<Cell>& Grid::GetCells()
{
	return m_Cells;
}

std::vector<Cell> Grid::GetCellsCopy()
{
	return m_Cells;
}

void Grid::NegativeCheck(int& value)
{
	if (value <= 0)
		value = 1;
}

Cell::Cell(const glm::ivec2& position, int size, bool alive)
	: position(position)
	, size(size)
	, alive(alive)
{
}
