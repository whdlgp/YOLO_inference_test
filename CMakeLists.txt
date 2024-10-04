cmake_minimum_required(VERSION 3.10)

# Set the project name and languages used
project(DNN_Inference_test LANGUAGES CXX)

# Set the C++ standard (adjust as needed)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV library
find_package(OpenCV REQUIRED HINTS "/path/to/opencv/build")

# ONNX Runtime
set(ONNXRUNTIME_DIR "/home/choyg/Desktop/workspace/onnxruntime-linux-x64-gpu-1.19.2")
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_DIR}/lib)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)

# Set main.cpp
set(MAIN_SOURCES
    main.cpp
)

# Set module directory sources
set(MODULE_SOURCES
    module/util.hpp
    module/interface.hpp
    module/make_models.hpp
    module/backend_opencv_onnx.cpp
    module/backend_opencv_onnx.hpp
    module/backend_opencv_darknet.cpp
    module/backend_opencv_darknet.hpp
    module/backend_onnxruntime.cpp
    module/backend_onnxruntime.hpp
    module/postprocessor_yolov5.cpp
    module/postprocessor_yolov5.hpp
    module/postprocessor_yolov8.cpp
    module/postprocessor_yolov8.hpp
    module/postprocessor_yolov10.cpp
    module/postprocessor_yolov10.hpp
    module/postprocessor_darknet.cpp
    module/postprocessor_darknet.hpp
)

# Extra libraries
set(EXTRA_HEADERS
    extra/json.hpp # from https://github.com/nlohmann/json
)

# Create the executable
add_executable(${PROJECT_NAME} ${MAIN_SOURCES} ${MODULE_SOURCES})

# OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime
target_include_directories(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
target_link_directories(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(${PROJECT_NAME} onnxruntime onnxruntime_providers_cuda onnxruntime_providers_shared)

# Include directories
include_directories(module, extra)

# Set the output directory for the built executable to the testspace directory located at the same level as CMakeLists.txt
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/testspace"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/testspace"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/testspace"
)