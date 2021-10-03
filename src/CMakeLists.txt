
cmake_minimum_required( VERSION 2.6 )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if (NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "build type" FORCE)
endif()


project( Feature_Matching )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

find_package( OpenCV REQUIRED )
include_directories( ${OPENCV_INCLUDE_DIRS} )
link_directories( ${OpenCV_LIB_DIR} )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)

add_executable( Feature_Matching
   Feature_Matching.cpp
)

#target_compile_features( Feature_Matching PRIVATE cxx_alias_templates )
target_include_directories(Feature_Matching PRIVATE ${OpenCV_INCLUDE_DIR})
target_link_libraries( Feature_Matching ${OpenCV_LIBS} )
set_target_properties(Feature_Matching PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
