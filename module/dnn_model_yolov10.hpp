#pragma once

#include "dnn_model_base.hpp"
#include "opencv2/dnn.hpp"

#include "backend_opencv_onnx.hpp"
#include "postprocessor_yolov10.hpp"

inline std::shared_ptr<DetectionBase> make_yolov10()
{
    // Backend
    std::unique_ptr<BackendBase<float>> backend = std::make_unique<BackendOpenCVONNX>();

    // Post Processor
    std::unique_ptr<PostProcessor<float>> postproc = std::make_unique<PostProcessorYOLOv10>();

    return std::make_shared<DetectionBase>(std::move(backend), std::move(postproc));
}

// YOLOv10Model model class(ONNX)
class YOLOv10Model : public ObjectDetection
{
public:
    // Initialize the model
    void init(nlohmann::json init_params) override;

    // Perform inference
    virtual std::vector<Output> infer(Input& input) override;

private:
    cv::dnn::Net net;

    // Convert Object Detection input to cv::Mat
    cv::Mat convert_to_Mat(const ObjectDetection::Input& input);

    // Preprocess the image
    cv::Mat preprocess(const cv::Mat &image);

    // Perform inference
    std::vector<cv::Mat> infer(const cv::Mat &blob);

    // Postprocess the inference results
    std::vector<ObjectDetection::Output> postprocess(const cv::Mat &frame, const std::vector<cv::Mat> &outs);
};