` zig build run`

## with shader compilation
Currently either glsl or spirv source can be used but not both. Both shaders currently do exactly the same thing, pick one.
`slangc` or `glslc` must be installed on system (comes with vulkan sdk).
* `zig build run -Dslangc` 
* `zig build run -Dglslc` 