file(GLOB_RECURSE stdNlBayesFiles ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(stdNlBayes STATIC ${stdNlBayesFiles})

target_link_libraries(stdNlBayes lodepng)
