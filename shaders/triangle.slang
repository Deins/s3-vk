struct VSInput
{
    float2 pos;
    float4 col;
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float4 col;
};

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput output;
    output.pos = float4(input.pos, 0, 1);
    output.col = input.col;
    return output;
}

//
// Fragment shader
//

[shader("fragment")]
float4 fragmentMain(VSOutput input)
{
    return input.col;
}

