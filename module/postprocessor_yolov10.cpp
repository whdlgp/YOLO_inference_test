#include "postprocessor_yolov10.hpp"

#include <fstream>
#include <iostream>

// Initialize the Post Processor
void PostProcessorYOLOv10::init(nlohmann::json init_params)
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
std::vector<DetectedOutput> PostProcessorYOLOv10::run(std::vector<Matrix<float>>& inputs, int dst_width, int dst_height)
{
    // Latest YOLOv10 no need to use NMS
    // But, OpenCV not work with original YOLOv10
    // OpenCV need modified version of YOLOv10 until fix this issue
    /*
    // YOLOv10 output is [1, 300, 6] where 300 bbox with 6 = [left, top, right, bottom, conf, class id]
    const int num_boxes = inputs[0].shape[1]; // 300

    // Scale factor for resize BBox
    const float scale_x = static_cast<float>(dst_width) / inference_width;
    const float scale_y = static_cast<float>(dst_height) / inference_height;

    // Final Result output
    // YOLO v10 no need to do NMS
    std::vector<DetectedOutput> final_results;
    final_results.reserve(20);
    for (int i = 0; i < num_boxes; i++)
    {
        const float* data = inputs[0].data.data();
        float left = data[i * 6 + 0];
        float top = data[i * 6 + 1];
        float right = data[i * 6 + 2];
        float bottom = data[i * 6 + 3];
        float confidence = data[i * 6 + 4];
        int class_id = static_cast<int>(data[i * 6 + 5]);

        if (confidence >= confidence_threshold)
        {
            // Create an Output object and populate it
            DetectedOutput output;
            output.bbox_x = static_cast<int>(left * scale_x);
            output.bbox_y = static_cast<int>(top * scale_y);
            output.bbox_width = static_cast<int>((right - left) * scale_x);
            output.bbox_height = static_cast<int>((bottom - top) * scale_y);
            output.class_id = class_id;
            output.class_name = class_names[class_id];  // Assumes you have class_names vector available
            output.confidence = confidence;

            // Add the Output object to the outputs vector
            final_results.push_back(output);
        }
    }

    return final_results;
    */

    std::vector<DetectedOutput> final_results;

    // Assume all mats in inputs has same shape
    for (auto& input : inputs)
    {
        // Initialize vectors to store NMS results
        std::vector<std::vector<BoundingBox>> boxes_per_class(num_classes);
        std::vector<std::vector<float>> confidences_per_class(num_classes);
        std::vector<std::vector<int>> indices(num_classes);

        // [1, BBOX+NUM_CLASSES, NUM_BOXES], BBox data = (Left, Top, Right, Bottom)
        const int num_boxes = input.data.size() / (4 + num_classes);
        const float scale_x = static_cast<float>(dst_width) / inference_width;
        const float scale_y = static_cast<float>(dst_height) / inference_height;

        const float* lefts = input.data.data();
        const float* tops = lefts + num_boxes;
        const float* rights = tops + num_boxes;
        const float* bottoms = rights + num_boxes;
        const float* confs = bottoms + num_boxes;

        for (int c = 0; c < num_classes; c++)
        {
            const float* conf = confs + (c * num_boxes);
            for (int b = 0; b < num_boxes; b++)
            {
                if(conf[b] >= confidence_threshold)
                {
                    const float left = lefts[b] * scale_x;
                    const float top = tops[b] * scale_y;
                    const float right = rights[b] * scale_x;
                    const float bottom = bottoms[b] * scale_y;

                    BoundingBox box{
                        static_cast<int>(left),
                        static_cast<int>(top),
                        static_cast<int>(right - left),
                        static_cast<int>(bottom - top)
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