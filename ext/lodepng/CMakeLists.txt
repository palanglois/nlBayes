file(GLOB_RECURSE lodepngFiles ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(lodepng STATIC ${lodepngFiles})
