#include "BaseEffect.h"
#include <sstream>
#include <iostream>

BaseEffect::BaseEffect(ID3D11Device* pDevice, const std::wstring& assetFile)
{
	m_pEffect = LoadEffect(pDevice, assetFile);
	m_pTechnique = m_pEffect->GetTechniqueByName("DefaultTechnique");
	if (!m_pTechnique->IsValid())
	{
		std::wcout << L"Technique is not valid\n";
	}

	m_pMatWorldViewProjVariable = m_pEffect->GetVariableByName("gWorldViewProj")->AsMatrix();
	ShowEffectVarWarning(m_pMatWorldViewProjVariable, L"m_pMatWorldViewProjVariable not valid");

	m_pMatWorldVariable = m_pEffect->GetVariableByName("gWorldMatrix")->AsMatrix();
	ShowEffectVarWarning(m_pMatWorldVariable, L"m_pMatWorldVariable not valid");

	m_pMatViewInverseVariable = m_pEffect->GetVariableByName("gInverseViewMatrix")->AsMatrix();
	ShowEffectVarWarning(m_pMatViewInverseVariable, L"m_pMatViewInverseVariable not valid");

	m_pFilterMethod = m_pEffect->GetVariableByName("gFilterMethod")->AsScalar();
	ShowEffectVarWarning(m_pFilterMethod, L"m_pFilterMethod not valid");

	m_pLightIntensity = m_pEffect->GetVariableByName("gLightIntesity")->AsScalar();
	ShowEffectVarWarning(m_pLightIntensity, L"m_pLightIntensity not valid");

	m_pLightDirection = m_pEffect->GetVariableByName("gLightDirection")->AsVector();
	ShowEffectVarWarning(m_pLightDirection, L"m_pLightDirection not valid");
}

BaseEffect::~BaseEffect()
{
	m_pMatViewInverseVariable->Release();
	m_pMatWorldVariable->Release();
	m_pMatWorldViewProjVariable->Release();
	m_pTechnique->Release();
	m_pEffect->Release();
}

ID3DX11Effect* BaseEffect::LoadEffect(ID3D11Device* pDevice, const std::wstring& assetFile)
{
	HRESULT result = S_OK;
	ID3D10Blob* pErrorBlob = nullptr;
	ID3DX11Effect* pEffect;

	DWORD shaderFlags = 0;
#if defined(DEBUG) || defined(_DEBUG)
	shaderFlags |= D3DCOMPILE_DEBUG;
	shaderFlags |= D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

	result = D3DX11CompileEffectFromFile(assetFile.c_str(),
		nullptr,
		nullptr,
		shaderFlags,
		0,
		pDevice,
		&pEffect,
		&pErrorBlob
	);

	if (FAILED(result))
	{
		if (pErrorBlob != nullptr)
		{
			char* pErrors = (char*)pErrorBlob->GetBufferPointer();

			std::wstringstream ss;
			for (unsigned int i = 0; i < pErrorBlob->GetBufferSize(); i++)
				ss << pErrors[i];

			OutputDebugStringW(ss.str().c_str());
			pErrorBlob->Release();
			pErrorBlob = nullptr;

			std::wcout << ss.str() << std::endl;
		}
		else
		{
			std::wstringstream ss;
			ss << "EffectLoader: Failed to CreateEffectFromFile!\nPath: " << assetFile;
			std::wcout << ss.str() << std::endl;
			return nullptr;
		}
	}

	if (pErrorBlob)
		pErrorBlob->Release();

	return pEffect;
}

ID3DX11Effect* BaseEffect::GetEffect() const
{
	return m_pEffect;
}

ID3DX11EffectTechnique* BaseEffect::GetTechnique() const
{
	return m_pTechnique;
}

ID3DX11EffectMatrixVariable* BaseEffect::GetWorldViewProjMatrix() const
{
	return m_pMatWorldViewProjVariable;
}

ID3DX11EffectMatrixVariable* BaseEffect::GetWorldMatrix() const
{
	return m_pMatWorldVariable;
}

ID3DX11EffectMatrixVariable* BaseEffect::GetViewInverseMatrix() const
{
	return m_pMatViewInverseVariable;
}

ID3DX11EffectScalarVariable* BaseEffect::GetFilterMethod() const
{
	return m_pFilterMethod;
}

ID3DX11EffectScalarVariable* BaseEffect::GetLightIntensity() const
{
	return m_pLightIntensity;
}

ID3DX11EffectVectorVariable* BaseEffect::GetLightDirection() const
{
	return m_pLightDirection;
}

ID3DX11EffectScalarVariable* BaseEffect::GetPower() const
{
	return m_pPower;
}

void BaseEffect::ShowEffectVarWarning(ID3DX11EffectVariable* var, const std::wstring& warning) const
{
	if (!var->IsValid())
	{
		std::wcout << warning << "\n";
	}
}
