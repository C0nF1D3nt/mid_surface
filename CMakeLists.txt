cmake_minimum_required(VERSION 3.16)

project(mid_surface_mapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(VTK REQUIRED COMPONENTS
  CommonCore
  CommonDataModel
  IOLegacy
)

if (VTK_VERSION VERSION_LESS "8.90.0")
  include(${VTK_USE_FILE})
endif()

add_executable(mid_surface_mapper mid_surface_mapper.cpp)
target_link_libraries(mid_surface_mapper PRIVATE ${VTK_LIBRARIES})
target_compile_definitions(mid_surface_mapper PRIVATE MID_SURFACE_MAPPER_BUILD_CLI)

if (NOT VTK_VERSION VERSION_LESS "8.90.0")
  vtk_module_autoinit(
    TARGETS mid_surface_mapper
    MODULES ${VTK_LIBRARIES}
  )
endif()
