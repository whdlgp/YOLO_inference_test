#pragma once

#include "interface.hpp"

// Imple of Post Process of YOLO v5 Object Detection models
class PostProcessorYOLOv5 : public PostProcessor<float> 
{
public:
    // Initialize the Post Processor
    void init(nlohmann::json init_params) override;
    
    // Perform Post Process
    std::vector<DetectedOutput> run(std::vector<Matrix<float>>& inputs, int dst_width, int dst_height) override;
private:
    // Thresholds
    float confidence_threshold = 0.5;
    float nms_threshold = 0.4;
    // Model Input Shape
    int inference_width = 608;
    int inference_height = 608;
    // Names of classes
    std::vector<std::string> class_names;
    int num_classes = 0;
};