#include "postprocessor_yolov8.hpp"

#include <fstream>

// Initialize the Post Processor
void PostProcessorYOLOv8::init(nlohmann::json init_params)
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
std::vector<DetectedOutput> PostProcessorYOLOv8::run(std::vector<Matrix<float>>& inputs, int dst_width, int dst_height)
{
    std::vector<DetectedOutput> final_results;

    // Assume all mats in inputs has same shape
    for (auto& input : inputs)
    {
        // Initialize vectors to store NMS results
        std::vector<std::vector<BoundingBox>> boxes_per_class(num_classes);
        std::vector<std::vector<float>> confidences_per_class(num_classes);
        std::vector<std::vector<int>> indices(num_classes);

        // [1, BBOX+NUM_CLASSES, NUM_BOXES], BBox data = (Xc, Yc, W, H)
        const int num_boxes = input.data.size() / (4 + num_classes);
        const float scale_x = static_cast<float>(dst_width) / inference_width;
        const float scale_y = static_cast<float>(dst_height) / inference_height;

        const float* x_centers = input.data.data();
        const float* y_centers = x_centers + num_boxes;
        const float* widths = y_centers + num_boxes;
        const float* heights = widths + num_boxes;
        const float* confs = heights + num_boxes;

        for (int c = 0; c < num_classes; c++)
        {
            const float* conf = confs + (c * num_boxes);
            for (int b = 0; b < num_boxes; b++)
            {
                if(conf[b] >= confidence_threshold)
                {
                    const float x_center = x_centers[b] * scale_x;
                    const float y_center = y_centers[b] * scale_y;
                    const float width = widths[b] * scale_x;
                    const float height = heights[b] * scale_y;

                    BoundingBox box{
                        static_cast<int>(x_center - width / 2),
                        static_cast<int>(y_center - height / 2),
                        static_cast<int>(width),
                        static_cast<int>(height)
                    };

                    boxes_per_class[c].push_back(box);
                    confidences_per_class[c].push_back(conf[b]);
                }
            }
        }

        for (int c = 0; c < num_classes; ++c)
        {
            indices[c] = non_maximum_suppression(
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