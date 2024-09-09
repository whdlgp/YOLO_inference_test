#pragma once

#include <any>
#include <vector>
#include <cstdint>
#include <string>

#include "util.hpp"

// Basic object detection model class interface
class ObjectDetection
{
public:
    using InitParams = std::any;

    virtual ~ObjectDetection() {}

    // Input form
    class Input
    {
    public:
        int width;
        int height;
        int chan;
        std::vector<uint8_t> data;
    };

    // Output form
    class Output
    {
    public:
        int bbox_x;
        int bbox_y;
        int bbox_width;
        int bbox_height;
        int class_id;
        std::string class_name;
        float confidence;
    };

    // Initialize the model
    virtual void init(nlohmann::json init_params)
    {
        if(!check_and_get(init_params, "confidence_threshold", this->confidence_threshold))
            throw std::invalid_argument("wrong confidence_threshold");
        if(!check_and_get(init_params, "nms_threshold", this->nms_threshold))
            throw std::invalid_argument("wrong nms_threshold");
        if(!check_and_get(init_params, "inference_width", this->inference_width))
            throw std::invalid_argument("wrong inference_width");
        if(!check_and_get(init_params, "inference_height", this->inference_height))
            throw std::invalid_argument("wrong inference_height");

        // Add your model imple initialization
    }

    // Perform inference
    virtual std::vector<Output> infer(Input& input) = 0;
protected:
    float confidence_threshold = 0.5;
    float nms_threshold = 0.4;
    int inference_width = 608;
    int inference_height = 608;

    std::vector<std::string> class_names;
    int num_classes = 0;
};
