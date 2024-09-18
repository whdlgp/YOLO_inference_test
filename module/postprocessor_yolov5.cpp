#include "postprocessor_yolov5.hpp"

#include <fstream>

// Initialize the Post Processor
void PostProcessorYOLOv5::init(nlohmann::json init_params)
{
    std::string names_file;
    if (!check_and_get(init_params, "names_file", names_file))
        throw std::invalid_argument("wrong names_file");
    if (!check_and_get(init_params, "inference_width", this->inference_width))
        throw std::invalid_argument("wrong inference_width");
    if (!check_and_get(init_params, "inference_height", this->inference_height))
        throw std::invalid_argument("wrong inference_height");
    if (!check_and_get(init_params, "confidence_threshold", this->confidence_threshold))
        throw std::invalid_argument("wrong confidence_threshold");
    if (!check_and_get(init_params, "nms_threshold", this->nms_threshold))
        throw std::invalid_argument("wrong nms_threshold");
    
    // Load class names
    std::ifstream class_file(names_file);
    if (!class_file)
    {
        throw std::runtime_error("failed to open classes.txt");
    }

    std::string line;
    while (std::getline(class_file, line))
    {
        class_names.push_back(line);
    }

    num_classes = static_cast<int>(class_names.size());
}

// Perform Post Process
std::vector<DetectedOutput> PostProcessorYOLOv5::run(std::vector<Matrix<float>>& inputs, int dst_width, int dst_height)
{
    std::vector<DetectedOutput> final_results;

    // Assume all mats in inputs has same shape
    for (auto& input : inputs)
    {
        std::vector<std::vector<BoundingBox>> boxes_per_class(num_classes);
        std::vector<std::vector<float>> confidences_per_class(num_classes);
        std::vector<std::vector<int>> indices(num_classes);

        const int num_boxes = input.data.size() / (5 + num_classes);
        const float scale_x = static_cast<float>(dst_width) / inference_width;
        const float scale_y = static_cast<float>(dst_height) / inference_height;

        for (int i = 0; i < num_boxes; ++i)
        {
            const float* data = &input.data[i * (5 + num_classes)];

            float x_center = data[0] * scale_x;
            float y_center = data[1] * scale_y;
            float width = data[2] * scale_x;
            float height = data[3] * scale_y;
            float obj_conf = data[4];  // Objectness confidence

            BoundingBox box{
                static_cast<int>(x_center - width / 2),
                static_cast<int>(y_center - height / 2),
                static_cast<int>(width),
                static_cast<int>(height)
            };

            for (int c = 0; c < num_classes; ++c)
            {
                float class_conf = data[5 + c];
                float confidence = obj_conf * class_conf;

                if (confidence >= confidence_threshold)
                {
                    boxes_per_class[c].push_back(box);
                    confidences_per_class[c].push_back(confidence);
                }
            }
        }

        for (int c = 0; c < num_classes; ++c)
        {
            indices[c] = nonMaximumSuppression(
                boxes_per_class[c],
                confidences_per_class[c],
                nms_threshold
            );

            for (int idx : indices[c])
            {
                DetectedOutput output;
                output.bbox_x = boxes_per_class[c][idx].x;
                output.bbox_y = boxes_per_class[c][idx].y;
                output.bbox_width = boxes_per_class[c][idx].width;
                output.bbox_height = boxes_per_class[c][idx].height;
                output.class_id = c;
                output.class_name = class_names[c];
                output.confidence = confidences_per_class[c][idx];

                final_results.push_back(output);
            }
        }
    }
    
    return final_results;
}