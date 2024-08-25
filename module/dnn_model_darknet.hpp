#pragma once

#include "dnn_model_base.hpp"

// Darknet based model class
class DarknetModel : public DNNModel
{
public:
    void init(const InitParams &model_files, float confidence_threshold = 0.5, float nms_threshold = 0.4) override;

    // Preprocess the image
    cv::Mat preprocess(const cv::Mat &image) override;

    // Perform inference
    std::vector<cv::Mat> infer(const cv::Mat &blob) override;

    // Postprocess the inference results
    void postprocess(const cv::Mat &frame, const std::vector<cv::Mat> &outs) override;

    // Utility function to draw the results
    void draw_results(cv::Mat &frame) override;

    // Getter functions to access the postprocessing results
    const std::vector<cv::Rect>& get_boxes() const { return boxes; }
    const std::vector<int>& get_class_ids() const { return class_ids; }
    const std::vector<float>& get_confidences() const { return confidences; }
private:
    cv::dnn::Net net;

    float confidence_threshold;
    float nms_threshold;

    std::vector<std::string> class_names;
    int num_classes = 0;
    
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    // colors for bounding boxes
    const int num_colors = 4;
    const cv::Scalar colors[4] = {
        {0, 255, 255},
        {255, 255, 0},
        {0, 255, 0},
        {255, 0, 0}
    };
};