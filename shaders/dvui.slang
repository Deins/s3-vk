struct VSInput
{
    float2 pos;
    float4 col;
    float2 uv;
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float4 col;
    float2 uv;
};

struct ScreenTransform
{
    float2 view_scale;
    float2 view_translate;
};
[[push_constant]]
ConstantBuffer<ScreenTransform> screen_transform;

layout(binding = 1) // i had some trouble binding glsl code to texture1, so this is here just to maintain compatibility
Sampler2D texture_sampler;

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput output;

    float2 transformed_pos = input.pos * screen_transform.view_scale + screen_transform.view_translate;
    output.pos = float4(transformed_pos, 0.0, 1.0);
    output.col = input.col;
    output.uv = input.uv;

    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input)
{
    float4 tex = texture_sampler.Sample(input.uv);
    return tex * input.col;
}

