#include <windows.h>
#include <iostream>
#include <vld.h>

#include "SDL2Application.h"
#include "DirectXApplication.h"

void CreateApplication(HINSTANCE hInstance);

int wmain()
{
    wWinMain(GetModuleHandle(0), 0, 0, SW_SHOW);
}

//Keep reference for parameter names
//int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow)
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR, int)
{
	//Enable run time memory check in debug mode
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	
	CreateApplication(hInstance);
	return 0;
}

int GetApplication()
{
	std::string number{};
	
	std::cout << "0: SDL2 Application" << std::endl;
	std::cout << "1: DirectX Application" << std::endl;
	std::cout << "Choose Application: ";
	std::getline(std::cin, number);
	int value = 0;
	
	try
	{
		value = std::stoi(number);
	}
	catch (...)
	{
		
	}

	return value;
}

void CreateApplication(HINSTANCE hInstance)
{
	//Spacebar:     start/stop simulation
	//Enter:        show/hide the grid
	//Backspace:    clear the grid

	//Create an application, you can give it the size of a cell
	//int appNr = GetApplication();
	int appNr = 1;

	Application* app = nullptr;
	switch (appNr)
	{
	case 0:
		app = new SDL2Application{15};
		break;
	case 1:
		app = new DirectXApplication{hInstance};
		break;
	default:
		app = new SDL2Application{15};
		break;
	}

	//Run the application
	try
	{
		app->Run();
	}
	catch (std::exception exception)
	{
		std::cout << exception.what() << std::endl;
		std::cin.get();
	}
	catch (...)
	{
		std::cout << "An unexpected exception occured\n";
		std::cin.get();
	}

	delete app;
}
