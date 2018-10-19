cmake_minimum_required (VERSION 3.5)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

project(AutoDefRuntime)

set(CMAKE_BUILD_TYPE Release)#Release) #RelWithDebInfo
set(CMAKE_CXX_STANDARD 14)

#Output directories
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )

#grab important packages
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#MKL
include_directories($ENV{MKLROOT}/include)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")

find_package(Eigen3 REQUIRED)

#Tensorflow
find_package(TensorflowCC REQUIRED)

# Optimization lib
include_directories(${PROJECT_SOURCE_DIR}/../../extern/LBFGSpp/include)

# JSON lib
include_directories(${PROJECT_SOURCE_DIR}/../../extern/json/include)

# BOOST
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost COMPONENTS filesystem REQUIRED) 

#Gauss
set(LIBIGL_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/../../extern/libigl/include) # Weird that I have to do this for GAUSS seperately
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fopenmp")# -fno-inline")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(Gauss_ROOT_DIR ${PROJECT_SOURCE_DIR}/../../extern/GAUSS/)
find_package(Gauss REQUIRED)

#Libigl
set(LIBIGL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../extern/libigl/include)
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" ON)
option(LIBIGL_WITH_ANTTWEAKBAR      "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL             "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO           "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK             "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE           "Use Embree"         OFF)
option(LIBIGL_WITH_LIM              "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB           "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK            "Use MOSEK"          OFF)
option(LIBIGL_WITH_NANOGUI          "Use Nanogui menu"   OFF)
option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
option(LIBIGL_WITH_PNG              "Use PNG"            ON)
option(LIBIGL_WITH_PYTHON           "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN           "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE         "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER           "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML              "Use XML"            OFF)

find_package(LIBIGL REQUIRED QUIET)


file(GLOB SRC_CODE
    ${AutoDefRuntime_SOURCE_DIR}/src/*.h
    ${AutoDefRuntime_SOURCE_DIR}/src/*.cpp
)

include_directories(${Gauss_INCLUDE_DIRS})

#Example codes
link_directories(${Gauss_LIB_DIR_RELEASE})
add_executable(AutoDefRuntime ${SRC_CODE})
target_link_libraries(AutoDefRuntime ${Gauss_LIBS} ${UI_LIBS} -lpthread igl::core igl::viewer igl::png TensorflowCC::Static Boost::filesystem -lgomp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lm -ldl)

