# CMake configuration
cmake_minimum_required(VERSION 3.5)
project(example_08_perf_rdma)

set(CYT_DIR ${CMAKE_SOURCE_DIR}/../../../)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CYT_DIR}/cmake)
find_package(CoyoteSW REQUIRED)

# Add source files
set(INSTANCE "client" CACHE STRING "RDMA software build targe: client or server")
if(INSTANCE STREQUAL "server")
    set(TARGET_DIR "${CMAKE_SOURCE_DIR}/src/server")
    message("*** Coyote Example 8: RDMA Server [Software Build] ***")
    include_directories("${CMAKE_SOURCE_DIR}/src/include")
endif()
if(INSTANCE STREQUAL "client")
    set(TARGET_DIR "${CMAKE_SOURCE_DIR}/src/client")
    message("*** Coyote Example 8: RDMA Client [Software Build] ***")
    include_directories("${CMAKE_SOURCE_DIR}/src/include")
endif()

# Create build targets and link against required libraries
set(EXEC test)
add_executable(${EXEC} ${TARGET_DIR}/main.cpp)
target_link_libraries(${EXEC} PUBLIC Coyote)
target_link_directories(${EXEC} PUBLIC /usr/local/lib)

# Target GPU architecture - gfx90 corresponds to MI210 - change as needed
set(AMD_GPU "gfx90a" CACHE STRING "Target GPU platform")
set_property(TARGET ${EXEC} PROPERTY HIP_ARCHITECTURES ${AMD_GPU})
