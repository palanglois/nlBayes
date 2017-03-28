cmake_minimum_required(VERSION 2.8)
project(nlBayes)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

#Adding reference to the media directory
add_definitions(-DNLBAYES_MEDIA_DIR="\\"${CMAKE_CURRENT_SOURCE_DIR}/media/\\"")

#Adding current directory for include
include_directories(${CMAKE_CURRENT_LIST_DIR})

#Adding lodepng
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/ext/lodepng)
include_directories(${CMAKE_CURRENT_LIST_DIR}/ext/lodepng)

#Adding Eigen
include_directories(${CMAKE_CURRENT_LIST_DIR}/ext/Eigen)

#Adding the standard nl bayes implementation
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/lib/stdNlBayes)
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/stdNlBayes)

add_executable(nlBayes main.cpp)
target_link_libraries(nlBayes stdNlBayes)

