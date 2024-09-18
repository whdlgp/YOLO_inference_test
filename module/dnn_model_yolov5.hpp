#pragma once

#include "dnn_model_base.hpp"

#include <memory>
#include <opencv2/dnn.hpp>

#include "backend_opencv_onnx.hpp"
#include "postprocessor_yolov5.hpp"

inline std::shared_ptr<DetectionBase> make_yolov5()
{
    // Backend
    std::unique_ptr<BackendBase<float>> backend = std::make_unique<BackendOpenCVONNX>();

    // Post Processor
    std::unique_ptr<PostProcessor<float>> postproc = std::make_unique<PostProcessorYOLOv5>();

    return std::make_shared<DetectionBase>(std::move(backend), std::move(postproc));
}

// YOLOv5Model model class
class YOLOv5Model : public ObjectDetection
{
public:
    // Initialize the model
    void init(nlohmann::json init_params) override;

    // Perform inference
    virtual std::vector<Output> infer(Input& input) override;

private:
    cv::dnn::Net net;

    // Input 타입을 Backend가 처리할 수 있는 타입으로 변환
    std::vector<uint8_t> preprocess(const ObjectDetection::Input& input);

    // Perform inference using backend
    std::vector<float> infer(const std::vector<uint8_t>& preprocessed_input);

    // Postprocess the inference results
    std::vector<ObjectDetection::Output> postprocess(const std::vector<float>& inference_output, int origin_width, int origin_height);
};
