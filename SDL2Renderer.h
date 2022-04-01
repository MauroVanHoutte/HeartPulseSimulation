#pragma once
#include "Renderer.h"
#include <string>

class Grid;
struct SDL_Window;
struct SDL_Renderer;

class SDL2Renderer final : public Renderer
{
public:
	SDL2Renderer(const std::string& windowName, int width, int height);
	virtual ~SDL2Renderer() = default;
	SDL2Renderer(const SDL2Renderer & other) = delete;
	SDL2Renderer(SDL2Renderer && other) = delete;
	SDL2Renderer& operator=(const SDL2Renderer & other) = delete;
	SDL2Renderer& operator=(SDL2Renderer && other) = delete;

	virtual bool Initialize() override;
	virtual void Render() override;
	virtual void Cleanup() override;
	virtual int GetWindowWidth() const override;
	virtual int GetWindowHeight() const override;

	void SetGrid(Grid* pGrid);
	void ToggleGrid();
	
private:
	Grid* m_pGrid;
	SDL_Window* m_Window;
	SDL_Renderer* m_Renderer;

	std::string m_WindowName;
	int m_Width;
	int m_Height;
	bool m_DrawGrid;

	void Draw() const;
};

