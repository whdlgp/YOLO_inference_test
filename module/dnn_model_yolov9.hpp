#pragma once

#include "dnn_model_base.hpp"
#include "opencv2/dnn.hpp"

// YOLOv9Model model class(ONNX)
class YOLOv9Model : public ObjectDetection
{
public:
    // Initialize the model
    void init(const InitParams &model_files
            , float confidence_threshold = 0.5f
            , float nms_threshold = 0.4f
            , int inference_width = 608
            , int inference_height = 608) override;

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