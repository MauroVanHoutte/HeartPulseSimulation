#include "DirectXApplication.h"
#include "DirectXRenderer.h"

#include "SDL.h"
#include "Time.h"

#include <chrono>
#include <algorithm>
#include <sstream>
#include <time.h>
#include <iomanip>
//Undifine max macro becuase it interferes with the std::max() algorithm.
#undef max


DirectXApplication::DirectXApplication(HINSTANCE hInstance)
	: Application(new DirectXRenderer{ hInstance, {"Heart Pulse Simulation: DirectX"}, 1820, 960 })
{
    std::string currentTime = GetCurrentTimeAsString();
    Session::Get().BeginSession("Resources/Output/SessionData " + currentTime + ".json");
    Logger::Get().BeginSession("Resources/Output/Data " + currentTime + ".txt");
    Logger::Get().LogCPUData();
}

bool DirectXApplication::Initialize()
{
	if (!m_pRenderer->Initialize())
		return false;

    m_pDirectXRenderer = static_cast<DirectXRenderer*>(m_pRenderer);
    if (!m_pDirectXRenderer)
    {
        throw std::exception{ "Unsupported renderer for this application" };
    }

	return true;
}

void DirectXApplication::PostInitialize()
{
    PerspectiveCamera* pCamera = m_pDirectXRenderer->GetCamera();
    pCamera->SetMovementSpeed(200.f);
}

void DirectXApplication::HandleInput()
{	
    const Uint8* state = SDL_GetKeyboardState(nullptr);

    PerspectiveCamera* pCamera = m_pDirectXRenderer->GetCamera();

	if (state[SDL_SCANCODE_W])
	{
        glm::fvec3 forwardVector{ pCamera->GetForwardVector() * pCamera->GetMovementSpeed() * m_DeltaTime };
        pCamera->Translate(forwardVector);
	}
    if (state[SDL_SCANCODE_S])
    {
        glm::fvec3 forwardVector{ pCamera->GetForwardVector() * pCamera->GetMovementSpeed() * m_DeltaTime };
        pCamera->Translate(-forwardVector);
    }
    if (state[SDL_SCANCODE_A])
    {
        glm::fvec3 rightVector{ pCamera->GetRightVector() * pCamera->GetMovementSpeed() * m_DeltaTime };
        pCamera->Translate(-rightVector);
    }
    if (state[SDL_SCANCODE_D])
    {
        glm::fvec3 rightVector{ pCamera->GetRightVector() * pCamera->GetMovementSpeed() * m_DeltaTime };
        pCamera->Translate(rightVector);
    }
    if (state[SDL_SCANCODE_E])
    {
        pCamera->RotateYaw(pCamera->GetRotationSpeed() * m_DeltaTime);
    }

    if (state[SDL_SCANCODE_Q])
    {
        pCamera->RotateYaw(-pCamera->GetRotationSpeed() * m_DeltaTime);
    }
    if (state[SDL_SCANCODE_UP])
    {
        glm::fvec3 upVector{ glm::fvec3{0, 1, 0} * pCamera->GetMovementSpeed() * m_DeltaTime };
        pCamera->Translate(upVector);
    }
    if (state[SDL_SCANCODE_DOWN])
    {
        glm::fvec3 upVector{ glm::fvec3{0, 1, 0} *pCamera->GetMovementSpeed() * m_DeltaTime };
        pCamera->Translate(-upVector);
    }


    SDL_Event e;

    while (SDL_PollEvent(&e))
    {
        if (ImGui::GetIO().WantCaptureKeyboard)
        {
            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        if (e.type == SDL_QUIT)
            DirectXApplication::QuitApplication();
    }
}

void DirectXApplication::Update(float deltaTime)
{
    //TIME();
    if (m_pDirectXRenderer->IsRunningTest() && !m_pDirectXRenderer->GetMeshes().empty())
    {
        Mesh* pMesh = m_pDirectXRenderer->GetMeshes()[0];

        if (pMesh)
        {
            VertexInput& vertex = pMesh->GetVertexBufferReference()[0];
            if (vertex.state == State::Waiting)
            {
                pMesh->PulseVertexV3(&vertex, m_pDirectXRenderer->GetDeviceContext(), false);
                m_pDirectXRenderer->IncreasePulses();
            }
        }
    }

    const std::vector<Mesh*>& meshes = m_pDirectXRenderer->GetMeshes();

    for (Mesh* const mesh : meshes)
    {
	    if (mesh)
	    {
            mesh->UpdateMeshV3(m_pDirectXRenderer->GetDeviceContext(), deltaTime);
	    }
    }
}

void DirectXApplication::Cleanup()
{
    Session::Get().EndSession();
    Logger::Get().EndSession();
	m_pRenderer->Cleanup();
    delete m_pRenderer;
}

std::string DirectXApplication::GetCurrentTimeAsString()
{
	//https://stackoverflow.com/questions/16357999/current-date-and-time-as-string/16358264
    time_t t = std::time(nullptr);
    tm tm{};
    localtime_s(&tm, &t);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
    return oss.str();
}
