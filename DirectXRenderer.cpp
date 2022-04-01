#include "DirectXRenderer.h"
#include "Mesh.h"

#include <memory>
#include <iostream>
#include "Application.h"

#include "Time.h"

#include "SDL_syswm.h"
#include "SDL_surface.h"

#pragma warning(push)
#pragma warning(disable:4616)
#pragma warning(disable:4201)
#include <gtc/type_ptr.hpp>
#pragma warning(pop)


HINSTANCE DirectXRenderer::m_Instance = NULL;
std::wstring DirectXRenderer::m_ClassTitle = L"DefaultClass";

DirectXRenderer::DirectXRenderer(HINSTANCE hInstance, const std::string& windowName, int width, int height)
	: m_pCamera(new PerspectiveCamera{{0, 0, 3}, {0, 0, 1}, float(width)/float(height)})
	, m_pWindow(nullptr)
	, m_WindowTitle(windowName)
	, m_WindowTitleWide(std::wstring{windowName.begin(), windowName.end()})
	, m_Width(width)
	, m_Height(height)
	, m_showWireframe(false)
	, m_CameraPosition()
    , m_Index(0)
	, m_LoadingMesh(false)
	, m_Mutex{}
{
    m_Instance = hInstance;
}

bool DirectXRenderer::Initialize()
{
    if (!InitializeSDLWindow())
        return false;

    if (InitializeDirectX() != S_OK)
        return false;

    if (!InitializeImGui())
        return false;

    if (m_pCamera)
    {
        m_CameraPosition = m_pCamera->GetPosition();
        m_pCamera->SetFar(99999.f);
    }
    else
    {
        throw std::exception{ "Failed to initialize camera in renderer" };
    }

    return true;
}

void DirectXRenderer::Render()
{
    TIME();
    if (!m_pDeviceContext && !m_pSwapChain)
        return;

    //Clear buffer
    glm::vec3 clearColor{ 0.2f, 0.2f, 0.6f };
    m_pDeviceContext->ClearRenderTargetView(m_pRenderTargetView, &clearColor.r);
    m_pDeviceContext->ClearDepthStencilView(m_pDepthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.f, 0);

    //Render

    if (!m_pMeshes.empty())
    {
        RenderMeshes();
    }

    RenderImGui();

    //Present
    m_pSwapChain->Present(0, 0);
}

void DirectXRenderer::Cleanup()
{
    if (m_pRenderTargetView)
    {
        m_pRenderTargetView->Release();
    }

    if (m_pRenderTargetBuffer)
    {
        m_pRenderTargetBuffer->Release();
    }

    if (m_pDepthStencilView)
    {
        m_pDepthStencilView->Release();
    }
    if (m_pDepthStencilBuffer)
    {
        m_pDepthStencilBuffer->Release();
    }

    if (m_pSwapChain)
    {
        m_pSwapChain->Release();
    }

    if (m_pDeviceContext)
    {
        m_pDeviceContext->ClearState();
        m_pDeviceContext->Flush();
        m_pDeviceContext->Release();
    }

    if (m_pDevice)
    {
        m_pDevice->Release();
    }

    if (m_pDXGIFactory)
    {
        m_pDXGIFactory->Release();
    }
	
	if (m_Handle != NULL)
	{
		UnregisterClass(m_ClassTitle.c_str(), m_Instance);
		DestroyWindow(m_Handle);
	}

    if (m_pCamera)
        delete m_pCamera;

    for (std::thread& thread : m_CreationThreads)
    {
        if (thread.joinable())
            thread.join();
    }

    for (Mesh* mesh : m_pMeshes)
    {
        if (mesh)
        {
            delete mesh;
            mesh = nullptr;
        }
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    if (m_pWindow)
    {
        SDL_DestroyWindow(m_pWindow);
		SDL_Quit();
    }

    if (m_Buffer)   
	   delete[] m_Buffer;
}

int DirectXRenderer::GetWindowWidth() const
{
	return m_Width;
}

int DirectXRenderer::GetWindowHeight() const
{
	return m_Height;
}

void DirectXRenderer::ToggleGrid()
{
}


PerspectiveCamera* DirectXRenderer::GetCamera() const
{
    return m_pCamera;
}

ID3D11Device* const DirectXRenderer::GetDevice() const
{
    return m_pDevice;
}

ID3D11DeviceContext* const DirectXRenderer::GetDeviceContext() const
{
    return m_pDeviceContext;
}


bool DirectXRenderer::IsRunningTest()
{
    return m_RunningTest;
}

void DirectXRenderer::IncreasePulses()
{
    ++m_NrOfPules;
    if (m_NrOfPules >= m_MaxNrOfPules)
    {
        m_RunningTest = false;
        m_NrOfPules = 0;
        if (m_pMeshes.size() > 0)
            RemoveMesh(m_pMeshes[0]);

        std::cout << "-- Test Completed --\n";
        //if (!m_Test1Complete)
        //{
        //    std::cout << "Running Test Wihout fibres\n";
        //    m_Test1Complete = true;
        //    m_NrOfPules = 0;
        //    CreateMesh("Resources/Models/HeartSmallVolume.bin", FileType::BIN, false);
        //}
        //else if (!m_Test2Complete)
        //{
        //    m_Test2Complete = true;
        //    m_NrOfPules = 0;
        //    std::cout << "-- Test Completed --\n";
        //    Application::QuitApplication();
        //}
    }
}


const std::vector<Mesh*>& DirectXRenderer::GetMeshes() const
{
    return m_pMeshes;
}

void DirectXRenderer::CreateMesh(const std::string& filePath, FileType fileType, bool useFibres)
{
    Mesh* mesh = nullptr;
    m_LoadingMesh = true;
    std::thread creationThread{ [this, &mesh, filePath, fileType, useFibres]()
        {
            TIME();
            mesh = new Mesh(m_pDevice, filePath, false, fileType);
            mesh->UseFibres(useFibres);
            if (!mesh->GetVertexBuffer().empty())
            {
                glm::fvec3 pos = mesh->GetVertexBuffer()[0].position;
                m_pCamera->Translate(pos);
            }

            AddMesh(mesh);
            m_LoadingMesh = false;
        }
    };
    m_CreationThreads.push_back(std::move(creationThread));
}

void DirectXRenderer::AddMesh(Mesh* pMesh)
{
    if (pMesh)
    {
        if (!pMesh->GetVertexBuffer().empty())
        {
            const std::lock_guard<std::mutex> lock(m_Mutex);
            m_pMeshes.push_back(pMesh);
        }
        else
            std::cout << "Trying to add mesh without vertices!" << std::endl;
    }
}

void DirectXRenderer::RemoveMesh(Mesh* pmesh)
{
    if (pmesh)
    {
        std::vector<Mesh*>::const_iterator it = std::find(m_pMeshes.begin(), m_pMeshes.end(), pmesh);
        if (it != m_pMeshes.end())
        {
            m_pMeshes.erase(it);
        }

        delete pmesh;
    }
}


bool DirectXRenderer::InitializeSDLWindow()
{
    SDL_Init(SDL_INIT_VIDEO);

    m_pWindow = SDL_CreateWindow(
        m_WindowTitle.c_str(),
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        m_Width, m_Height, 0);

    return m_pWindow != nullptr;
}

HRESULT DirectXRenderer::InitializeDirectX()
{
    //Create a Debug Device when Bebug flag is enabled
    uint32_t createDeviceFlags = 0;
#if defined(DEBUG) || defined(_DEBUG)
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3,
        D3D_FEATURE_LEVEL_9_1
    };

    //Create Device and Device Context, using hardware acceleration
    HRESULT result = D3D11CreateDevice(0, D3D_DRIVER_TYPE_HARDWARE, 0, createDeviceFlags, 0, 0, D3D11_SDK_VERSION, &m_pDevice, featureLevels, &m_pDeviceContext);
    if (FAILED(result))
        return result;

    //Create DXGI Factory to create SwapChain based on hardware
    result = CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&m_pDXGIFactory));
    if (FAILED(result))
        return result;

    //Create the swapchain discriptor
    DXGI_SWAP_CHAIN_DESC swapChainDesc{};
    swapChainDesc.BufferDesc.Width = m_Width;
    swapChainDesc.BufferDesc.Height = m_Height;
    swapChainDesc.BufferDesc.RefreshRate.Numerator = 1;
    swapChainDesc.BufferDesc.RefreshRate.Denominator = 60;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount = 1;
    swapChainDesc.Windowed = true;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    swapChainDesc.Flags = 0;

    //Get the handle HWND from the sdl backbuffer
    SDL_SysWMinfo sysWMInfo{};
    SDL_VERSION(&sysWMInfo.version);
    SDL_GetWindowWMInfo(m_pWindow, &sysWMInfo);
    swapChainDesc.OutputWindow = sysWMInfo.info.win.window;
    //swapChainDesc.OutputWindow = m_Handle;

    //Create swapchain and hook it into the handle of the SDL window
    result = m_pDXGIFactory->CreateSwapChain(m_pDevice, &swapChainDesc, &m_pSwapChain);
    if (FAILED(result))
        return result;

    //Create the depth/stensil buffer and view
    D3D11_TEXTURE2D_DESC depthStensilDesc{};
    depthStensilDesc.Width = m_Width;
    depthStensilDesc.Height = m_Height;
    depthStensilDesc.MipLevels = 1;
    depthStensilDesc.ArraySize = 1;
    depthStensilDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    depthStensilDesc.SampleDesc.Count = 1;
    depthStensilDesc.SampleDesc.Quality = 0;
    depthStensilDesc.Usage = D3D11_USAGE_DEFAULT;
    depthStensilDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    depthStensilDesc.CPUAccessFlags = 0;
    depthStensilDesc.MiscFlags = 0;

    D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc{};
    depthStencilViewDesc.Format = depthStensilDesc.Format;
    depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    depthStencilViewDesc.Texture2D.MipSlice = 0;

    result = m_pDevice->CreateTexture2D(&depthStensilDesc, 0, &m_pDepthStencilBuffer);
    if (FAILED(result))
        return result;

    result = m_pDevice->CreateDepthStencilView(m_pDepthStencilBuffer, &depthStencilViewDesc, &m_pDepthStencilView);
    if (FAILED(result))
        return result;

    //Create the render target view
    result = m_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&m_pRenderTargetBuffer));
    if (FAILED(result))
        return result;

    result = m_pDevice->CreateRenderTargetView(m_pRenderTargetBuffer, 0, &m_pRenderTargetView);
    if (FAILED(result))
        return result;

    //Bind the views to the output Merger Stage
    m_pDeviceContext->OMSetRenderTargets(1, &m_pRenderTargetView, m_pDepthStencilView);

    //Set the viewport
    D3D11_VIEWPORT viewPort{};
    viewPort.Width = static_cast<float>(m_Width);
    viewPort.Height = static_cast<float>(m_Height);
    viewPort.TopLeftX = 0.f;
    viewPort.TopLeftY = 0.f;
    viewPort.MinDepth = 0.f;
    viewPort.MaxDepth = 1.f;
    m_pDeviceContext->RSSetViewports(1, &viewPort);

    return HRESULT{ S_OK };
}

HRESULT DirectXRenderer::CreateHandle()
{
    if (m_Instance == NULL)
        m_Instance = (HINSTANCE)GetModuleHandle(NULL);

    HICON hIcon = NULL;
    WCHAR szExePath[MAX_PATH];
    GetModuleFileName(NULL, szExePath, MAX_PATH);

    // If the icon is NULL, then use the first one found in the exe
    if (hIcon == NULL && m_Instance != NULL)
        hIcon = ExtractIcon(m_Instance, szExePath, 0);

    // Register the windows class
    WNDCLASS wndClass;
    wndClass.style = CS_DBLCLKS;
    wndClass.lpfnWndProc = WindowProcedure;
    wndClass.cbClsExtra = 0;
    wndClass.cbWndExtra = 0;
    wndClass.hInstance = m_Instance;
    wndClass.hIcon = hIcon;
    wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wndClass.lpszMenuName = NULL;
    wndClass.lpszClassName = m_ClassTitle.c_str();

    if (!RegisterClass(&wndClass))
    {
        DWORD dwError = GetLastError();
        if (dwError != ERROR_CLASS_ALREADY_EXISTS)
            return HRESULT_FROM_WIN32(dwError);
    }

    m_Rect;
    int x = CW_USEDEFAULT;
    int y = CW_USEDEFAULT;

    // No menu in this example.
    m_Menu = NULL;

    // This example uses a non-resizable 640 by 480 viewport for simplicity.
    //int nDefaultWidth = 640;
    //int nDefaultHeight = 480;
    SetRect(&m_Rect, 0, 0, m_Width, m_Height);
    AdjustWindowRect(
        &m_Rect,
        WS_OVERLAPPEDWINDOW,
        (m_Menu != NULL) ? true : false
    );

    // Create the window for our viewport.
    m_Handle = CreateWindow(
        m_ClassTitle.c_str(),
        m_WindowTitleWide.c_str(),
        WS_OVERLAPPEDWINDOW,
        x, y,
        (m_Rect.right - m_Rect.left), (m_Rect.bottom - m_Rect.top),
        0,
        m_Menu,
        m_Instance,
        0
    );

    if (m_Handle == NULL)
    {
        DWORD dwError = GetLastError();
        return HRESULT_FROM_WIN32(dwError);
    }

    return S_OK;
}

HWND DirectXRenderer::GetHandle() const
{
    return m_Handle;
}

LRESULT CALLBACK DirectXRenderer::WindowProcedure(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_CLOSE:
    {
        HMENU hMenu;
        hMenu = GetMenu(hWnd);
        if (hMenu != NULL)
        {
            DestroyMenu(hMenu);
        }
        DestroyWindow(hWnd);
        UnregisterClass(
            m_ClassTitle.c_str(),
            m_Instance
        );
        return 0;
    }

    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    }

    return DefWindowProc(hWnd, uMsg, wParam, lParam);
}


bool DirectXRenderer::InitializeImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    if (!ImGui_ImplSDL2_InitForD3D(m_pWindow))
        return false;

    if (!ImGui_ImplDX11_Init(m_pDevice, m_pDeviceContext))
        return false;

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    return true;
}

void DirectXRenderer::RenderImGui()
{
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplSDL2_NewFrame(m_pWindow);
    ImGui::NewFrame();

    //bool openDemo{ true };
    //ImGui::ShowDemoWindow(&openDemo);

    ImGui::Begin("Data");

    //Framerate counter
    ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    glm::fvec3 position = m_pCamera->GetPosition();
    std::string currentPosition = std::to_string(position.x) + ", " + std::to_string(position.y) + ", " + std::to_string(position.z);
    ImGui::Text(currentPosition.c_str());
    ImGui::InputFloat3("Camera Position", &m_CameraPosition.x);
    if (ImGui::Button("Set Camera Position"))
    {
        m_pCamera->SetPosition(m_CameraPosition);
    }

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();

    ImGuiDrawMatricesHeader();

    //Vertex Data
    Mesh* pMesh = nullptr;
    bool updateBuffer = false;

    if (ImGui::CollapsingHeader("Vertex Data", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (m_pMeshes.size() > 0 && m_pMeshes[0]->GetVertexBuffer().size() > 0 &&
            m_Index >= 0 && m_Index < m_pMeshes[0]->GetVertexBuffer().size())
        {
            ImGuiDrawMeshData(pMesh, updateBuffer);
        }
	    else if (m_pMeshes.empty())
	    {
            if (!m_LoadingMesh)
	            ImGuiDrawMeshLoadingHeader();
            else
            {
                ImGui::Spacing();
                ImGui::Spacing();
                ImGui::Spacing();
                ImGui::Text("Mesh is loading. See console for more details!");
                ImGui::Spacing();
                ImGui::Spacing();
                ImGui::Spacing();
            }
	    }
	    else if (m_Index < 0)
	    {
		    m_Index = 0;
	    }
	    else if (m_Index > m_pMeshes[0]->GetVertexBuffer().size())
	    {
		    m_Index = int(m_pMeshes[0]->GetVertexBuffer().size()) - 1;
	    }
	    else
	    {
		    if (m_pMeshes.size() > 0)
			    m_Index = int(m_pMeshes[0]->GetVertexBuffer().size()) - 1;
	    }
    }
    ImGui::End();

    if (pMesh)
    {
	    ImGui::Begin("Plots");
	    const std::vector<float>& values = pMesh->GetAPPlot();
	    glm::fvec2 minMax = pMesh->GetMinMax();

	    ImGui::PlotLines("APD Plot", values.data(), int(values.size()));

	    std::string string{"XMin: 0 "};
	    string += "XMax: " + std::to_string(values.size());
	    ImGui::Text(string.c_str());
	    string = "YMin: " + std::to_string(minMax.x) + " YMax: " + std::to_string(minMax.y);
	    ImGui::Text(string.c_str());

	    string = "Active Potential Duration: " + std::to_string(pMesh->GetAPD());
	    ImGui::Text(string.c_str());

        string = "Diastolic Interval: " + std::to_string(pMesh->GetDiastolicInterval().count());
        ImGui::Text(string.c_str());

	    ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
}

void DirectXRenderer::RenderMeshes() const
{
    const glm::mat4 viewMatrix = m_pCamera->GetViewMatrix();
    glm::mat4 inverseViewMatrix = glm::inverse(viewMatrix);

    //Credit to Pepijn Langeraert for finding this issue
    const glm::mat4 projectionMatrix = glm::transpose(m_pCamera->GetProjectionMatrix());

    inverseViewMatrix = glm::transpose(inverseViewMatrix);
    float* inverseView = (float*)glm::value_ptr(inverseViewMatrix);

    for (Mesh* const mesh : m_pMeshes)
    {
        glm::mat4 worldViewProjectionMatrix = projectionMatrix * viewMatrix * mesh->GetWorldMatrix();
        worldViewProjectionMatrix = glm::transpose(worldViewProjectionMatrix);
        float* worldViewProjection = (float*)glm::value_ptr(worldViewProjectionMatrix);

        mesh->Render(m_pDeviceContext, worldViewProjection, inverseView);
    }
}

void DirectXRenderer::JoinCreationThreads()
{
    std::vector<int> threadsToRemove{};

    for (int i{}; i < m_CreationThreads.size(); i++)
    {
        std::thread& thread = m_CreationThreads[i];
        if (thread.joinable())
        {
            thread.join();
            threadsToRemove.push_back(i);
        }
    }

    int nrOfRemovedThreads = 0;
    for (int threadIdx : threadsToRemove)
    {
        int idx = threadIdx - nrOfRemovedThreads;
        m_CreationThreads.erase(m_CreationThreads.begin() + idx);
    }
}

void DirectXRenderer::ImGuiDrawMatricesHeader() const
{
    if (ImGui::CollapsingHeader("Matrices"))
    {
        ImGui::BeginGroup();
        ImGui::DragFloat4("Lookat[0]", glm::value_ptr(m_pCamera->GetLookAt()[0]));
        ImGui::DragFloat4("Lookat[1]", glm::value_ptr(m_pCamera->GetLookAt()[1]));
        ImGui::DragFloat4("Lookat[2]", glm::value_ptr(m_pCamera->GetLookAt()[2]));
        ImGui::DragFloat4("Lookat[3]", glm::value_ptr(m_pCamera->GetLookAt()[3]));
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::DragFloat4("Projection[0]", glm::value_ptr(m_pCamera->GetProjectionMatrix()[0]));
        ImGui::DragFloat4("Projection[1]", glm::value_ptr(m_pCamera->GetProjectionMatrix()[1]));
        ImGui::DragFloat4("Projection[2]", glm::value_ptr(m_pCamera->GetProjectionMatrix()[2]));
        ImGui::DragFloat4("Projection[3]", glm::value_ptr(m_pCamera->GetProjectionMatrix()[3]));
        ImGui::EndGroup();
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();
    }
}

void DirectXRenderer::ImGuiDrawMeshData(Mesh* pMesh, bool& updateBuffer)
{
        pMesh = m_pMeshes[0];
        std::vector<VertexInput>& vertexBuffer = pMesh->GetVertexBufferReference();
        VertexInput initialVertex = vertexBuffer[m_Index];

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();
        //Color Tab
        bool open = true;
        ImGui::Begin("Color", &open);
        const float* data = value_ptr(initialVertex.color1);
        float color1[3] = { (data)[0], (data)[1], (data)[2] };
        if (ImGui::ColorPicker3("Start Color", color1))
        {
            if (color1[0] != data[0] || color1[1] != data[1] || color1[2] != data[2])
            {
                initialVertex.color1 = glm::fvec3{ color1[0], color1[1], color1[2] };
                updateBuffer = true;
            }
        }

        data = value_ptr(initialVertex.color2);
        float color2[3] = { (data)[0], (data)[1], (data)[2] };
        if (ImGui::ColorPicker3("Pulse Color", color2))
        {
            if (color2[0] != data[0] || color2[1] != data[1] || color2[2] != data[2])
            {
                initialVertex.color2 = glm::fvec3{ color2[0], color2[1], color2[2] };
                updateBuffer = true;
            }
        }
        ImGui::End();

        //Vertex Data
        ImGui::Text(("Amount of vertices : " + std::to_string(vertexBuffer.size())).c_str());
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::InputInt("Vertex Index", &m_Index);
        if (m_Index < 0)
            m_Index = 0;
        else if (m_Index >= vertexBuffer.size())
            m_Index = int(vertexBuffer.size()) - 1;
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();
        float scale = pMesh->GetScale().x;
        ImGui::InputFloat("Scale", &scale);
        if (scale != pMesh->GetScale().x)
        {
            pMesh->SetScale(scale, scale, scale);
        }

        glm::fvec3 translation = pMesh->GetTranslation();
        ImGui::InputFloat3("Translation", &translation.x);
        if (translation != pMesh->GetTranslation())
        {
            pMesh->Translate(translation.x, translation.y, translation.z);
        }
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();

        auto vec3ToString = [](const glm::fvec3& vector)
        {
            std::string vecString{};
            for (int i{}; i < 3; i++)
            {
                vecString += std::to_string(vector[i]) + " ";
            }

            return vecString;
        };
        auto vec2ToString = [](const glm::fvec2& vector)
        {
            std::string vecString{};
            for (int i{}; i < 2; i++)
            {
                vecString += std::to_string(vector[i]) + " ";
            }

            return vecString;
        };
        auto indicesToString = [](const std::set<uint32_t>& indices)
        {
            std::string indicesString{};

            for (uint32_t index : indices)
            {
                indicesString += std::to_string(index) + ", ";
            }

            return indicesString;
        };

        ImGui::Text(("Position: " + vec3ToString(initialVertex.position)).c_str());
        ImGui::Text(("Color: " + vec3ToString(initialVertex.color1)).c_str());
        ImGui::Text(("Normal: " + vec3ToString(initialVertex.normal)).c_str());
        ImGui::Text(("Tangent: " + vec3ToString(initialVertex.tangent)).c_str());
        ImGui::Text(("UV: " + vec2ToString(initialVertex.uv)).c_str());
        ImGui::Text(("Power: " + std::to_string(initialVertex.apVisualization)).c_str());
        ImGui::Text(("Active Potential: " + std::to_string(initialVertex.actionPotential)).c_str());
        ImGui::Text(("Neighbour Indices: " + indicesToString(initialVertex.neighbourIndices)).c_str());

        ImGui::Spacing();
        ImGui::Text(("Fibre Direction: " + vec3ToString(initialVertex.fibreDirection)).c_str());

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();

        bool useFibres = pMesh->UseFibres();
		if (ImGui::Checkbox("Use Fibres", &useFibres))
		{
            pMesh->UseFibres(useFibres);
		}

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();

        int DI = int(pMesh->GetDiastolicInterval().count());
        ImGui::InputInt("Diastolic Interval", &DI);
        if (DI != int(pMesh->GetDiastolicInterval().count()))
        {
            pMesh->SetDiastolicInterval(float(DI));
        }

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();

        std::string state{};
        switch (vertexBuffer[m_Index].state)
        {
        case State::Waiting: state = "Waiting"; break;
        case State::Receiving: state = "Receiving"; break;
        case State::APD: state = "Action Potential"; break;
        case State::DI: state = "Diastolic Interval"; break;
        }
        ImGui::Text(state.c_str());

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();
        if (ImGui::Button("Pulse Vertex"))
        {
            pMesh->PulseVertexV3(m_Index, m_pDeviceContext);
        }
        if (ImGui::Button("Pulse Mesh"))
        {
            pMesh->PulseMesh(m_pDeviceContext);
        }

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();

        if (ImGui::Button("Clear All Pulses"))
        {
            pMesh->ClearPulse(m_pDeviceContext);
        }

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();

        if (ImGui::Button("Write to binary file"))
        {
            pMesh->CreateCachedBinary();
        }

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();

        if (ImGui::Button("Load Cached Fibre Data"))
        {
            pMesh->LoadCachedFibres();
        }

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::LabelText("", "Settings");

        //Camera speed
        float cameraSpeed = m_pCamera->GetMovementSpeed();
        ImGui::InputFloat("Movement Speed", &cameraSpeed);
        if (abs(cameraSpeed - m_pCamera->GetMovementSpeed()) > 0.001f)
        {
            m_pCamera->SetMovementSpeed(cameraSpeed);
        }

        //Wireframe toggle
        bool showWireframe = m_showWireframe;
        if (ImGui::Checkbox("Show Wireframe", &m_showWireframe))
        {
            if (showWireframe != m_showWireframe)
            {
                for (Mesh* mesh : m_pMeshes)
                {
                    mesh->SetWireframe(m_showWireframe);
                }
            }
        }

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();

        if (updateBuffer)
        {
            for (VertexInput& vertex : vertexBuffer)
            {
                vertex.color1 = initialVertex.color1;
                vertex.color2 = initialVertex.color2;
            }

            pMesh->UpdateVertexBuffer(m_pDeviceContext);
        }

        ImGui::PushStyleColor(ImGuiCol_Button, { 158 / 255.f, 21 / 255.f, 27 / 255.f, 1 });
        if (ImGui::Button("Unload Mesh"))
        {
            RemoveMesh(pMesh);
        }
        ImGui::PopStyleColor();
    
}

void DirectXRenderer::ImGuiDrawMeshLoadingHeader()
{
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, { 158 / 255.f, 21 / 255.f, 27 / 255.f, 1 });
    ImGui::Text("No Mesh Loaded");
    ImGui::PopStyleColor();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::Button("Load Heart Mesh Binary"))
    {
        CreateMesh("Resources/Models/HeartSmallVolume.bin", FileType::BIN);
    }
    if (ImGui::Button("Load Coarser Mesh"))
    {
        CreateMesh("Resources/Models/CoarserScaled.obj", FileType::OBJ);
    }
    if (ImGui::Button("Load Coarser Mesh Binary"))
    {
        CreateMesh("Resources/Models/CoarserScaled.bin", FileType::BIN);
    }
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::Text("Load a mesh from the Resource/Models folder in the project");
    ImGui::InputText("Mesh", &m_Buffer[0], m_Size);
    ImGui::Spacing();
    ImGui::Spacing();
    if (ImGui::Button("Load OBJ"))
    {
        CreateMesh("Resources/Models/" + std::string{ m_Buffer }, FileType::OBJ);
    }
    ImGui::Spacing();
    ImGui::Spacing();
    if (ImGui::Button("Load BIN"))
    {
        CreateMesh("Resources/Models/" + std::string{ m_Buffer }, FileType::BIN);
    }
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::InputInt("Number of repeated puleses", &m_MaxNrOfPules);
    if (!m_RunningTest && ImGui::Button("Run Test 1"))
    {
        std::cout << "-- Started Running Test --\n";
        m_RunningTest = true;
        CreateMesh("Resources/Models/CoarserScaled.bin", FileType::BIN);
    }
    if (!m_RunningTest && ImGui::Button("Run Test 2"))
    {
        std::cout << "-- Started Running Test --\n";
        m_RunningTest = true;
        CreateMesh("Resources/Models/CoarserScaled.bin", FileType::BIN, false);
    }
    //if (ImGui::Button("Load PTS"))
    //{
    //    CreateMesh("Resources/Models/" + std::string{ m_Buffer }, FileType::PTS);
    //}
    //if (ImGui::Button("Load VTK"))
    //{
    //    CreateMesh("Resources/Models/" + std::string{ m_Buffer }, FileType::VTK);
    //}

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
}