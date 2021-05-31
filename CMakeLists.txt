cmake_minimum_required(VERSION 3.17)
project(vgfx2 CXX)

find_program(glslc glslc)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)
function(target_build_shaders target)
    cmake_parse_arguments("" "" "" "" ${ARGN})
    foreach(shader_in ${_UNPARSED_ARGUMENTS})
        get_filename_component(p ${shader_in} NAME)
        set(shader_out ${CMAKE_BINARY_DIR}/shaders/${p}.spv)
        get_filename_component(p ${shader_in} ABSOLUTE)
        add_custom_command(
            OUTPUT ${shader_out}
            COMMAND ${glslc} -o ${shader_out} ${p}
            DEPENDS ${shader_in}
            IMPLICIT_DEPENDS CXX ${shader_in}
            VERBATIM
        )
        set_source_files_properties(${shader_out} PROPERTIES GENERATED TRUE)
        target_sources(${target} PRIVATE ${shader_out})
    endforeach(shader_in)
endfunction(target_build_shaders)

add_executable(vgfx main.cpp vg.cpp)
target_link_libraries(vgfx glfw dl vulkan)
target_compile_features(vgfx PRIVATE cxx_std_20)
target_compile_options(vgfx PRIVATE -Wall -Wpedantic)
target_build_shaders(vgfx shader.vert shader.frag)
