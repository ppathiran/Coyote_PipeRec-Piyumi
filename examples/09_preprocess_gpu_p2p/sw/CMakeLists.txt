cmake_minimum_required(VERSION 3.14)
project(fpga_p2p_pybind)

# Paths
set(CYT_DIR ${CMAKE_SOURCE_DIR}/../../../)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CYT_DIR}/cmake)

# Detect correct Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Coyote support
find_package(CoyoteSW REQUIRED)

message("*** Coyote Pybind11 Wrapper Build ***")

# Source files
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src/")
set(SRC_FILE "${SRC_DIR}/fpga_p2p_pybind.cpp")

# GPU architecture
set(AMD_GPU "gfx90a" CACHE STRING "Target GPU platform")

# Pybind11 shared module target
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

# Set HIP target architecture
set_property(TARGET fpga_p2p_pybind PROPERTY HIP_ARCHITECTURES ${AMD_GPU})

# Make sure .so is correct for Python
set_target_properties(fpga_p2p_pybind PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

