//-----------------------------
//	Variables
//-----------------------------
float4x4 gWorldViewProj : WorldViewProjection;
float4x4 gWorldMatrix : WORLD;
float4x4 gInverseViewMatrix : VIEWINVERSE;
int gFilterMethod : FILTER;

float3 gLightDirection = float3(0.577f, -0.577f, 0.577);
float gLightIntesity = float(1.f);

cbuffer gPowerValues : register(b0)
{
    float gPower;
}

//-----------------------------
//	Rasterizer states
//-----------------------------
RasterizerState gRasterizerState
{
	CullMode = back;
	FrontCounterClockWise = true;
};

//-----------------------------
//	Blend states
//-----------------------------
BlendState gBlendState
{
    AlphaToCoverageEnable = false;
    BlendEnable[0] = false;
    SrcBlend = one;
    DestBlend = zero;
    BlendOp = add;
    SrcBlendAlpha = one;
    DestBlendAlpha = zero;
    BlendOpAlpha = add;
    RenderTargetWriteMask[0] = 0x0F;
};

//-----------------------------
//	DepthStencil states
//-----------------------------
DepthStencilState gDepthStencilState
{
    DepthEnable = true;
    DepthWriteMask = all;
    DepthFunc = less;
    StencilEnable = false;

    StencilReadMask = 0x0F;
    StencilWriteMask = 0x0F;

    FrontFaceStencilFunc = always;
    BackFaceStencilFunc = always;

    FrontFaceStencilDepthFail = keep;
    BackFaceStencilDepthFail = keep;

    FrontFaceStencilPass = keep;
    BackFaceStencilPass = keep;

    FrontFaceStencilFail = keep;
    BackFaceStencilFail = keep;
};

//-----------------------------
//	Input/Output structs
//-----------------------------
struct VS_INPUT
{
	float3 Position : POSITION;
    float3 Color : COLOR;
    float3 ColorTwo : SECCOLOR;
    float3 Normal : NORMAL;
    float3 Tangent : TANGENT;
    float2 UV : TEXCOORD;
    float Power : POWER;
};

struct VS_OUTPUT
{
	float4 Position : SV_POSITION;
    float3 Color : COLOR;
    float3 ColorTwo : SECCOLOR;
	float3 WorldPosition : WORLDPOSITION;
    float3 Normal : NORMAL;
    float3 Tangent : TANGENT;
    float Power : POWER;
};

VS_OUTPUT VS(VS_INPUT input)
{
	VS_OUTPUT output = (VS_OUTPUT)0;
	output.Position = mul(gWorldViewProj, float4(input.Position, 1.f));
    output.WorldPosition = mul(input.Position, (float3x3)gWorldMatrix);
	output.Normal = mul(normalize(input.Normal), (float3x3)gWorldMatrix);
	output.Tangent = mul(normalize(input.Tangent), (float3x3)gWorldMatrix);
    output.Color = input.Color;
    output.ColorTwo = input.ColorTwo;
    output.Power = input.Power;
    return output;
}

//-----------------------------
//	Pixel Shader
//-----------------------------
float4 PS(VS_OUTPUT input) : SV_TARGET
{
    float lambert = 0.5f + dot(-input.Normal, gLightDirection);
    lambert = lambert * (lambert > 0.f);

    float3 color1 = input.Color;
    float3 color2 = input.ColorTwo;
    
    //Interpolate between the colors
    float r = (color2.r - color1.r) * input.Power + color1.r;
    float g = (color2.g - color1.g) * input.Power + color1.g;
    float b = (color2.b - color1.b) * input.Power + color1.b;
    
    return float4(r, g, b, 1) * lambert;
}

//-----------------------------
//	Technique
//-----------------------------
technique11 DefaultTechnique
{
	pass P0
	{
		//SetRasterizerState(gRasterizerState);
        //SetDepthStencilState(gDepthStencilState, 0);
        //SetBlendState(gBlendState, float4(0.f, 0.f, 0.f, 0.f), 0xFFFFFFFF);
		SetVertexShader(CompileShader(vs_5_0, VS()));
		SetGeometryShader(NULL);
		SetPixelShader(CompileShader(ps_5_0, PS()));
	}
}