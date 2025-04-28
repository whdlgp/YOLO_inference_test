#pragma once

#include <memory>

// Backend import
#include "backend_opencv_darknet.hpp"
#include "backend_opencv_onnx.hpp"
#include "backend_onnxruntime.hpp"

// Post Processor import
#include "postprocessor_darknet.hpp"
#include "postprocessor_yolov5.hpp"
#include "postprocessor_yolov8.hpp"
#include "postprocessor_yolov10.hpp"

// Make OpenCV Darknet Model
inline std::shared_ptr<DetectionBase> make_darknet()
{
    // Backend
    std::unique_ptr<BackendBase<float>> backend = std::make_unique<BackendOpenCVDarknet>();

    // Post Processor
    std::unique_ptr<PostProcessor<float>> postproc = std::make_unique<PostProcessorDarknet>();

    return std::make_shared<DetectionBase>(std::move(backend), std::move(postproc));
}

// Make OpenCV ONNX YOLOv5 Model
inline std::shared_ptr<DetectionBase> make_yolov5()
{
    // Backend
    std::unique_ptr<BackendBase<float>> backend = std::make_unique<BackendOpenCVONNX>();

    // Post Processor
    std::unique_ptr<PostProcessor<float>> postproc = std::make_unique<PostProcessorYOLOv5>();

    return std::make_shared<DetectionBase>(std::move(backend), std::move(postproc));
}

// Make OpenCV ONNX YOLOv6 Model
inline std::shared_ptr<DetectionBase> make_yolov6()
{
    // Backend
    std::unique_ptr<BackendBase<float>> backend = std::make_unique<BackendOpenCVONNX>();

    // Post Processor
    // YOLO v6 have save process of post processor
    std::unique_ptr<PostProcessor<float>> postproc = std::make_unique<PostProcessorYOLOv5>();

    return std::make_shared<DetectionBase>(std::move(backend), std::move(postproc));
}

// Make OpenCV ONNX YOLOv8 Model
inline std::shared_ptr<DetectionBase> make_yolov8()
{
    // Backend
    std::unique_ptr<BackendBase<float>> backend = std::make_unique<BackendOpenCVONNX>();

    // Post Processor
    std::unique_ptr<PostProcessor<float>> postproc = std::make_unique<PostProcessorYOLOv8>();

    return std::make_shared<DetectionBase>(std::move(backend), std::move(postproc));
}

// Make OpenCV ONNX YOLOv9 Model
inline std::shared_ptr<DetectionBase> make_yolov9()
{
    // Backend
    std::unique_ptr<BackendBase<float>> backend = std::make_unique<BackendOpenCVONNX>();

    // Post Processor
    // YOLO v9 have save process of post processor
    std::unique_ptr<PostProcessor<float>> postproc = std::make_unique<PostProcessorYOLOv8>();

    return std::make_shared<DetectionBase>(std::move(backend), std::move(postproc));
}

// Make OpenCV ONNX YOLOv10 Model
inline std::shared_ptr<DetectionBase> make_yolov10()
{
    // Backend
    std::unique_ptr<BackendBase<float>> backend = std::make_unique<BackendOpenCVONNX>();

    // Post Processor
    std::unique_ptr<PostProcessor<float>> postproc = std::make_unique<PostProcessorYOLOv10>();

    return std::make_shared<DetectionBase>(std::move(backend), std::move(postproc));
}

// Make ONNXRuntime model(TODO)
inline std::shared_ptr<DetectionBase> make_onnxruntime()
{
    // Backend
    std::unique_ptr<BackendBase<float>> backend = std::make_unique<BackendONNXRuntime>();

    // Post Processor
    std::unique_ptr<PostProcessor<float>> postproc = std::make_unique<PostProcessorYOLOv8>();

    return std::make_shared<DetectionBase>(std::move(backend), std::move(postproc));
}