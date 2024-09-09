cmake_minimum_required(VERSION 3.10)

# Set the project name and languages used
project(DNN_Inference_test LANGUAGES CXX)

# Set the C++ standard (adjust as needed)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV library
find_package(OpenCV REQUIRED HINTS "/path/to/opencv/build")

# Set main.cpp
set(MAIN_SOURCES
    main.cpp
)

# Set module directory sources
set(MODULE_SOURCES
    module/dnn_model_darknet.cpp
    module/dnn_model_yolov5.cpp
    module/dnn_model_yolov6.cpp
    module/dnn_model_yolov8.cpp
    module/dnn_model_yolov9.cpp
    module/dnn_model_yolov10.cpp
)

set(MODULE_HEADERS
    module/dnn_model_base.hpp
    module/dnn_model_darknet.hpp
    module/dnn_model_yolov5.hpp
    module/dnn_model_yolov6.hpp
    module/dnn_model_yolov8.hpp
    module/dnn_model_yolov9.hpp
    module/dnn_model_yolov10.hpp
)

set(EXTRA_HEADERS
    extra/json.hpp
)

# Create the executable
add_executable(${PROJECT_NAME} ${MAIN_SOURCES} ${MODULE_SOURCES})

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(module, extra)

# Set the output directory for the built executable to the testspace directory located at the same level as CMakeLists.txt
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/testspace"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/testspace"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/testspace"
)