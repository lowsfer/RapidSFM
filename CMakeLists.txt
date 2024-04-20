cmake_minimum_required(VERSION 3.5)

project(rapidsfm2 LANGUAGES CXX CUDA)
# cuda 11.3/11.5 causes register spill. 11.4 is ok only with sm-75
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")
# set(CUDA_TARGET_ROOT "/usr/local/cuda-11.4/targets/x86_64-linux")
# set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda-11.4/include")
# set(CUDA_TARGET_ROOT "/opt/cuda/targets/x86_64-linux")

add_subdirectory(extern/cuda-utils)
add_subdirectory(extern/RapidSIFT)
add_subdirectory(extern/RapidBoW)
add_subdirectory(extern/RapidBA)
add_subdirectory(extern/lambdatwist)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GeographicLib)
if (NOT GeographicLib_FOUND) # version 1.x does not provide cmake config and use a different lib name
    set(GeographicLib_LIBRARIES Geographic)
endif()

find_package(exiv2 REQUIRED)
if (exiv2_VERSION VERSION_GREATER_EQUAL 0.28)
    add_definitions(-DEXIV2_HAS_TO_INT64=1)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RAPIDSFM_ENABLE_SOSNET "The licence of upstream SOSNet weights (https://github.com/scape-research/SOSNet) is unspecified. Use it at your own risk. You have been warned!" OFF)

if (RAPIDSFM_ENABLE_SOSNET)
    set(RAPIDSIFT_ENABLE_SOSNET ON CACHE BOOL "Enable SOSNet" FORCE)
else()
    set(RAPIDSIFT_ENABLE_SOSNET OFF CACHE BOOL "Enable SOSNet" FORCE)
endif()


include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    /usr/include/opencv4
    extern/cuda-utils
    extern/RapidSIFT
    extern/RapidBoW
    extern/RapidBA
    extern/lambdatwist)
link_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib)

add_definitions(-DENABLE_SANITY_CHECK=0)
# accelerate debugging
add_definitions(-DEIGEN_NDEBUG=1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread -Wfatal-errors -Wreturn-type -Wall -Wextra -Wno-unknown-pragmas")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -keep --expt-relaxed-constexpr") #-res-usage
set(CUDA_PTXAS_FLAGS_RELEASE "${CUDA_PTXAS_FLAGS_RELEASE} -warn-lmem-usage -warn-spills -Werror") # -v
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math -DNDEBUG -Xptxas='${CUDA_PTXAS_FLAGS_RELEASE}'")


set(LEGACY_SFM_SRC
    legacy/PropagationMatchFilter.h legacy/PropagationMatchFilter.cpp
    legacy/geometry.hpp legacy/geometry.cpp
    legacy/rt_optimiser.hpp legacy/rt_optimiser.cpp legacy/rt_optimiser_helper.hpp legacy/aligned_array.hpp
    )

# bruteForceMatchIMMA.cu needs sm_75 or higer. Use a dedicated target to set CUDA_ARCHITECTURES
add_library(bruteForceMatchIMMA STATIC
	bruteForceMatch.h bruteForceMatch.cuh immaUtils.cuh bruteForceMatchIMMA.cu
	casHash/computeHash.cu casHash/buildBuckets.cu casHash/matchWithoutRemap.cu
	casHash/findBestMatch.cu casHash/accumulateDesc.cu
)
# set_target_properties(bruteForceMatchIMMA PROPERTIES CUDA_ARCHITECTURES 75)

add_library(RapidSFM
    RapidSFM.h RapidSFM.cpp
    ImagePair.h ImagePair.cpp
    degensac.h degensac.cpp
    Types.h Types.cpp Types.hpp
    fwd.h Config.h
    HandleGenerator.h
    Builder.h Builder.cpp BuilderMethods.cpp BuilderDbgMethods.cpp Builder.cu
    Scheduler.h Scheduler.cpp
    Image.h Image.cpp
    SfmUtils.h SfmUtils.cpp SfmUtils.hpp
    ransac.hpp
    models/ModelBase.h
    models/DefaultModel.h models/DefaultModel.cpp
    models/GlobalModel.h
    models/IncreModel.h models/IncreModel.cpp models/IncreModelMethods.cpp
    models/ModelBuilder.h models/ModelBuilder.cpp
    models/PnPOptimizer.h models/PnPOptimizer.cpp
    models/DebugModelViewer.h models/DebugModelViewer.cpp
    distortion.h distortion.cpp
    rsm.h rsm.hpp
    bruteForceMatch.h bruteForceMatch.cu bruteForceMatch.cpp
    bruteForceMatcher.h bruteForceMatcher.cpp
    parser/parser.h parser/parser.cpp
    RansacMatchFilter.h RansacMatchFilter.cpp RansacMatchFilter.cuh RansacMatchFilter.cu
    ${LEGACY_SFM_SRC}
)

target_link_libraries(RapidSFM PRIVATE bruteForceMatchIMMA cudart RapidBA RapidSIFT RapidBoW cuda-utils cudart nvToolsExt boost_fiber boost_context pthread opencv_highgui opencv_imgproc opencv_imgcodecs opencv_core opencv_features2d yaml-cpp boost_filesystem exiv2lib jpeg turbojpeg heif Eigen3::Eigen lambdatwist)

if (RAPIDSFM_ENABLE_SOSNET)
    link_directories(
        ${CMAKE_SOURCE_DIR}/../sosnet-ptq/build/qtcreator-release)
    target_link_libraries(RapidSFM PRIVATE  SOSNet_runtime sosnet_rc nvinfer)
else()
endif()

add_executable(rsfm rsfm/rsfm.cpp exifUtils.h)
target_link_libraries(rsfm PRIVATE RapidSFM ${GeographicLib_LIBRARIES} boost_program_options)

add_executable(worker rsfm/worker.cpp)
target_link_libraries(worker PRIVATE RapidSFM ${GeographicLib_LIBRARIES} boost_program_options)

add_executable(fiberTest fiberTest.cpp)
target_link_libraries(fiberTest PRIVATE cuda-utils boost_fiber boost_context pthread cudart exiv2lib jpeg turbojpeg heif)

add_executable(tests_rapidsfm
	test/geometryTest.cpp
	bruteForceMatch.cu bruteForceMatch.h bruteForceMatch.cuh test/bruteForceMatchKernelTest.cpp
	test/casHashComputeHashTest.cpp
	test/casHashBuildBucketsTest.cpp
	test/casHashFindBestMatchTest.cpp
	test/casHashAccumulateDescTest.cpp
)
target_link_libraries(tests_rapidsfm PRIVATE RapidSFM gtest gtest_main cuda-utils bruteForceMatchIMMA exiv2lib jpeg turbojpeg heif)

add_executable(converter converter.cpp rsm.hpp rsm.h Types.cpp distortion.cpp)