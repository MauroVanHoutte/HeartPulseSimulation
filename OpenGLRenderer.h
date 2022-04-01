#pragma once
#include "Renderer.h"
#include "string"

struct GLFWwindow;

class OpenGLRenderer final : public Renderer
{
public:
	OpenGLRenderer(const std::string& windowName, int width, int height);
	virtual ~OpenGLRenderer() = default;
	OpenGLRenderer(const OpenGLRenderer & other) = delete;
	OpenGLRenderer(OpenGLRenderer && other) = delete;
	OpenGLRenderer& operator=(const OpenGLRenderer & other) = delete;
	OpenGLRenderer& operator=(OpenGLRenderer && other) = delete;

	virtual bool Initialize() override;
	virtual void Render() override;
	virtual void Cleanup() override;

	virtual int GetWindowWidth() const override;
	virtual int GetWindowHeight() const override;
private:
	Grid* m_pGrid;
	GLFWwindow* m_Window;
	const std::string m_WindowName;
	int m_Width;
	int m_Height;
	bool m_DrawGrid;

	void Draw() const;
	float ConvertToDeviceCoordinates(int screenSpace, int width) const;
};

