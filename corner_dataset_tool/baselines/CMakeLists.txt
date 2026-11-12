cmake_minimum_required(VERSION 3.16)
project(corner_dataset_baselines LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL REQUIRED COMPONENTS common search features keypoints)
add_executable(pcl_baseline pcl_baseline.cpp)
target_include_directories(pcl_baseline PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcl_baseline PRIVATE ${PCL_LIBRARIES})
target_compile_options(pcl_baseline PRIVATE ${PCL_DEFINITIONS})

if(ORTHOGONAL_SOURCE)
    find_package(Ceres REQUIRED)
    if(CERES_VERSION VERSION_LESS 1.14 OR CERES_VERSION VERSION_GREATER_EQUAL 2.2)
        message(FATAL_ERROR "Author code requires Ceres LocalParameterization (Ceres >=1.14, <2.2)")
    endif()
    find_package(OpenCV REQUIRED COMPONENTS core)
    set(SRC "${ORTHOGONAL_SOURCE}")
    add_executable(orthogonal_baseline
        "${CMAKE_CURRENT_BINARY_DIR}/orthogonal_main.cpp"
        "${SRC}/3rdParty/tinyply/tinyply/tinyply.cpp"
        "${SRC}/include/PPF/PairDetector.cpp"
        "${SRC}/include/graph/PlaneGraph.cpp"
        "${SRC}/include/graph/ParallelPlaneGraph_filtering.cpp"
        "${SRC}/include/graph/ParallelPlaneGraph_clustering.cpp"
        "${SRC}/include/graph/ParallelPlaneGraph_optimization.cpp"
        "${SRC}/include/corners/Corner.cpp")
    target_include_directories(orthogonal_baseline PRIVATE
        "${SRC}/include" "${SRC}/3rdParty/Sophus" "${SRC}/3rdParty/CLI11/include"
        "${SRC}/3rdParty/nanoflann" "${SRC}/3rdParty/tinyply" ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(orthogonal_baseline PRIVATE Ceres::ceres ${OpenCV_LIBS})
endif()
