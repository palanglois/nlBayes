cmake_minimum_required(VERSION 2.8)
project(nlBayes)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

#Adding reference to the media directory
add_definitions(-DICPSPARSE_MEDIA_DIR="\\"${CMAKE_CURRENT_SOURCE_DIR}/media/\\"")

#Adding current directory for include
include_directories(${CMAKE_CURRENT_LIST_DIR})

add_executable(nlBayes main.cpp)

