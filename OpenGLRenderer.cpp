#include "OpenGLRenderer.h"

#include <iostream>
#include <algorithm>

#include "GLFW/glfw3.h"
#include "SDL2Application.h"

OpenGLRenderer::OpenGLRenderer(const std::string& windowName, int width, int height)
	: Renderer()
	, m_pGrid(nullptr)
	, m_Window(nullptr)
	, m_WindowName(windowName)
	, m_Width(width)
	, m_Height(height)
	, m_DrawGrid(true)
{
}

bool OpenGLRenderer::Initialize()
{
    //m_pGrid = grid;

    //Initialize the library
    if (!glfwInit())
    {
        std::cout << "Failed to initialize OpenGl" << std::endl;
        return false;
    }

    //Create a windowed mode window and its OpenGL context
    m_Window = glfwCreateWindow(m_Width, m_Height, m_WindowName.c_str() , NULL, NULL);
    if (!m_Window)
    {
        glfwTerminate();
        std::cout << "Failed to initialize OpenGl" << std::endl;
        return false;
    }

    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    //glViewport(0, 0, m_Width, m_Height);
    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();

    //Make the window's context current
    glfwMakeContextCurrent(m_Window);

    return true;
}

void OpenGLRenderer::Render()
{
    //Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);

    //Draw the grid
    Draw();

    //Swap the buffers and handle input events
    glfwSwapBuffers(m_Window);
    glfwPollEvents();

    if (glfwWindowShouldClose(m_Window))
        SDL2Application::QuitApplication();
}

void OpenGLRenderer::Cleanup()
{
    glfwTerminate();
}

int OpenGLRenderer::GetWindowWidth() const
{
    return m_Width;
}

int OpenGLRenderer::GetWindowHeight() const
{
    return m_Height;
}

void OpenGLRenderer::Draw() const
{
    const std::vector<Cell>& cells = m_pGrid->GetCells();

    std::for_each(cells.begin(), cells.end(), [this](const Cell& cell)
        {
            float x1, x2;
            float y1, y2;
			//Convert the spositions from x = [0, screenwidth] & y = [0, screenheight]
			//to x & y = [-1 & 1]
            x1 = ConvertToDeviceCoordinates(int(cell.position.x * cell.size), m_Width);
            x2 = ConvertToDeviceCoordinates(int(cell.position.x * cell.size + cell.size), m_Width);
            y1 = ConvertToDeviceCoordinates(int(cell.position.y * cell.size), m_Height);
            y2 = ConvertToDeviceCoordinates(int(cell.position.y * cell.size + cell.size), m_Height);

			//Fill the cell if it's alive, otherwhise draw the outline.
            if (cell.alive)
                glBegin(GL_QUADS);
            else
	            glBegin(GL_LINE_LOOP);

            glVertex2f(x1, y1);
            glVertex2f(x1, y2);
            glVertex2f(x2, y2);
            glVertex2f(x2, y1);
            glEnd();
			
        }
    );
}

float OpenGLRenderer::ConvertToDeviceCoordinates(int screenSpace, int width) const
{
    //[0, screenwidth/screenheight] range -> [0, 1] range -> [-1, 1] range
    return - (1 - (2 * (screenSpace / float(width))));
}
