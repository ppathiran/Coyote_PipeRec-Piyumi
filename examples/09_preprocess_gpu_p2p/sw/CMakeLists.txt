cmake_minimum_required(VERSION 3.14)
project(fpga_p2p_pybind)

set(CYT_DIR ${CMAKE_SOURCE_DIR}/../../../)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CYT_DIR}/cmake)

set(pybind11_DIR "$ENV{HOME}/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(CoyoteSW REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

message("*** Coyote Pybind11 Wrapper Build ***")

# Directory containing the executable(s) to be compiled
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src/")
set(SRC_FILE "${SRC_DIR}/fpga_p2p_pybind.cpp")

# Target GPU architecture - gfx90 corresponds to MI210 - change as needed
set(AMD_GPU "gfx90a" CACHE STRING "Target GPU platform")

# Pybind11 shared module target
add_library(fpga_p2p_pybind MODULE ${SRC_FILE})

target_include_directories(fpga_p2p_pybind PRIVATE
    ${CYT_DIR}/include
    ${CYT_DIR}/hw/common
    ${SRC_DIR}
)

target_link_libraries(fpga_p2p_pybind
    PRIVATE
    Coyote
    pybind11::module
    amdhip64
)

set_property(TARGET fpga_p2p_pybind PROPERTY HIP_ARCHITECTURES ${AMD_GPU})

# Remove lib prefix and enforce .so suffix for Python
set_target_properties(fpga_p2p_pybind PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

