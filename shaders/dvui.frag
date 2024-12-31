#version 450


layout(location = 0) in vec4 frag_col;
layout(location = 1) in vec2 frag_uv;
layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 f_color;

void main() {
    vec4 tex = texture(texSampler, frag_uv);
    vec4 col = frag_col;
    f_color =  tex * frag_col;
    //if (f_color.a < 3.0/255.0) discard;
}
