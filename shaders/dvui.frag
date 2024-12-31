#version 450


layout(location = 0) in vec4 frag_col;
layout(location = 1) in vec2 frag_uv;
layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 f_color;

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
    vec4 tex = texture(texSampler, frag_uv);
    vec4 col = frag_col;
    // tex = toLinear(tex);
    // tex = fromLinear(tex);
    // col = toLinear(col);
    //  col = fromLinear(col);
    f_color =  tex * frag_col;
    // f_color.rgb = pow(f_color.rgb, vec3(1.0/2.2));
    //f_color = toLinear(f_color);
}
