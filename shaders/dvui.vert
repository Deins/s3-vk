#version 450

layout(push_constant) uniform ScreenTransform { vec2 view_scale; vec2 view_translate; };

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec4 a_col;
layout(location = 2) in vec2 a_uv;

layout (location = 0) out vec4 frag_col;
layout (location = 1) out vec2 frag_uv;

void main() {
    gl_Position = vec4(a_pos * view_scale + view_translate, 0.0, 1.0);
    frag_col = a_col;
    // frag_col = toLinear(a_col);
    frag_uv = a_uv;
}
