pub usingnamespace @cImport({
    @cDefine("VMA_IMPLEMENTATION", "1");
    @cDefine("VMA_VULKAN_VERSION", "1002000"); // Vulkan 1.2
    @cInclude("vk_mem_alloc.h");
});

pub fn init() void {}
