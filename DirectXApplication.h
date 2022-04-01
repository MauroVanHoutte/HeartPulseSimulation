#pragma once
#include "Application.h"
#include "DirectXRenderer.h"
#include <string>

class DirectXApplication : public Application
{
public:
	DirectXApplication(HINSTANCE hInstance);
	DirectXApplication(const DirectXApplication & other) = delete;
	DirectXApplication(DirectXApplication && other) = delete;
	DirectXApplication& operator=(const DirectXApplication & other) = delete;
	DirectXApplication& operator=(DirectXApplication && other) = delete;
	virtual ~DirectXApplication() = default;

private:
	virtual bool Initialize() override;
	virtual void PostInitialize() override;
	virtual void HandleInput() override;
	virtual void Update(float deltaTime) override;
	virtual void Cleanup() override;

	std::string GetCurrentTimeAsString();

	DirectXRenderer* m_pDirectXRenderer = nullptr;
};

