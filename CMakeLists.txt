cmake_minimum_required(VERSION 3.1)

project(SemanticStixels)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -msse4.1")
endif (CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if (OpenCV_VERSION VERSION_LESS 3.3)
	message(FATAL_ERROR "Error: OpenCV version requires at least 3.3")
endif()

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

file(GLOB srcs *.cpp *.h*)

add_executable(semantic_stixels ${srcs})
target_link_libraries(semantic_stixels ${OpenCV_LIBS})
