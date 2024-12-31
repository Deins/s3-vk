#version 450

layout(push_constant) uniform ScreenTransform { vec2 view_scale; vec2 view_translate; };

layout(binding = 0) uniform UniformBufferObject {
    vec2 viewport_size;
} ubo;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec4 a_col;
layout(location = 2) in vec2 a_uv;

layout (location = 0) out vec4 frag_col;
layout (location = 1) out vec2 frag_uv;

// Converts a color from linear light gamma to sRGB gamma
vec4 fromLinear(vec4 linearRGB)
{
    bvec3 cutoff = lessThan(linearRGB.rgb, vec3(0.0031308));
    vec3 higher = vec3(1.055)*pow(linearRGB.rgb, vec3(1.0/2.4)) - vec3(0.055);
    vec3 lower = linearRGB.rgb * vec3(12.92);

    return vec4(mix(higher, lower, cutoff), linearRGB.a);
}

// Converts a color from sRGB gamma to linear light gamma
vec4 toLinear(vec4 sRGB)
{
    bvec3 cutoff = lessThan(sRGB.rgb, vec3(0.04045));
    vec3 higher = pow((sRGB.rgb + vec3(0.055))/vec3(1.055), vec3(2.4));
    vec3 lower = sRGB.rgb/vec3(12.92);
    return vec4(mix(higher, lower, cutoff), sRGB.a);
}

void main() {
    gl_Position = vec4(a_pos * view_scale + view_translate, 0.0, 1.0);
    frag_col = a_col;
    // frag_col = toLinear(a_col);
    frag_uv = a_uv;
}
