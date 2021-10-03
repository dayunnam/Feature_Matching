
cmake_minimum_required(VERSION 3.3)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} [OpenCV.cmake PATH])
project(Feature_Matching)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
 link_directories( ${OpenCV_LIB_DIR} )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)

add_executable(Feature_Matching Feature_Matching.cpp)

target_include_directories(Feature_Matching PRIVATE ${OpenCV_INCLUDE_DIR})

target_link_libraries(Feature_Matching PRIVATE ${OpenCV_LIBRARIES})

set_target_properties(Feature_Matching PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
