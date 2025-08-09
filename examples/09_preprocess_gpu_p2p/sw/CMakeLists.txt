# CMake configuration
cmake_minimum_required(VERSION 3.14)
project(fpga_p2p_pybind)

set(CYT_DIR ${CMAKE_SOURCE_DIR}/../../../)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CYT_DIR}/cmake)
find_package(CoyoteSW REQUIRED)

# detect correct Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

message("*** Coyote Example 11: DLRM Training with FPGA Preprocessing [Pybind11 Wrapper Build] ***")

# directory containing the executable(s) to be compiled
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src/")
set(SRC_FILE "${SRC_DIR}/fpga_p2p_pybind.cpp")

# pybind11 shared module target
pybind11_add_module(fpga_p2p_pybind MODULE ${SRC_FILE})

target_include_directories(fpga_p2p_pybind PRIVATE
    ${CYT_DIR}/include
    ${CYT_DIR}/hw/common
    ${SRC_DIR}
)

target_link_libraries(fpga_p2p_pybind
    PRIVATE
    Coyote
    amdhip64
)

# target GPU architecture - gfx90 corresponds to MI210 - change as needed
set(AMD_GPU "gfx90a" CACHE STRING "Target GPU platform")
set_property(TARGET fpga_p2p_pybind PROPERTY HIP_ARCHITECTURES ${AMD_GPU})

set_target_properties(fpga_p2p_pybind PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

