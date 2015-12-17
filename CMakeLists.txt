 cmake_minimum_required(VERSION 2.8.9)
    project(HELM)

#Compiler flags


SET(GCC_COVERAGE_COMPILE_FLAGS "-DNO_FREETYPE")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
    include_directories("${CMAKE_SOURCE_DIR}/include")
    include_directories("${CMAKE_SOURCE_DIR}/src")

SET(GCC_COVERAGE_COMPILE_FLAGSO "-O3")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGSO}" )
    include_directories("${CMAKE_SOURCE_DIR}/include")
    include_directories("${CMAKE_SOURCE_DIR}/src")

SET(GCC_COVERAGE_COMPILE_FLAGSopen "-fopenmp -m64 -std=gnu++11")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGSopen}" )
    include_directories("${CMAKE_SOURCE_DIR}/include")
    include_directories("${CMAKE_SOURCE_DIR}/src")


    add_library(enc src/hELMAutoencoder.cpp include/HELMAutoencoder.h)
    add_library(helm src/HELM.cpp include/HELM.h)
    add_library(mush src/Mushroom.cpp include/Mushroom.h)
    add_library(mnist src/loadMNIST.cpp include/LoadMNIST.h)
    add_library(sat src/Sat.cpp include/Sat.h)
    add_library(stat src/Stat.cpp include/Stat.h)

    add_executable(main main.cpp)

    target_link_libraries(main helm)
    target_link_libraries(main enc)
    target_link_libraries(main mush)
    target_link_libraries(main sat)
    target_link_libraries(main mnist)
    target_link_libraries(main stat)  
    target_link_libraries(main ${LIBS})
