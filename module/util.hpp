#pragma once

#include <json.hpp>

// Helper for check JSON data is correct for input argument
template<typename T>
bool check_and_get(const nlohmann::json& input, const std::string& key, T& value) 
{
    if (input.contains(key) && input[key].is_primitive()) 
    {
        if (std::is_same_v<T, std::string> && input[key].is_string()) 
        {
            value = input[key];
            return true;
        } 
        else if (std::is_same_v<T, int> && input[key].is_number_integer()) 
        {
            value = input[key];
            return true;
        } 
        else if (std::is_same_v<T, bool> && input[key].is_boolean()) 
        {
            value = input[key];
            return true;
        }
        else if (std::is_same_v<T, float> && input[key].is_number_float()) 
        {
            value = input[key];
            return true;
        }
    }
    return false;
}

// Define a simple struct to represent bounding boxes
class BoundingBox
{
public:
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

// Intersection over Union (IoU) calculation
inline float IoU(const BoundingBox& box1, const BoundingBox& box2)
{
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int inter_area = std::max(0, x2 - x1 + 1) * std::max(0, y2 - y1 + 1);
    int box1_area = box1.width * box1.height;
    int box2_area = box2.width * box2.height;

    return static_cast<float>(inter_area) / (box1_area + box2_area - inter_area);
}

// Custom Non-Maximum Suppression (NMS) function
inline std::vector<int> non_maximum_suppression(
    const std::vector<BoundingBox>& boxes,
    const std::vector<float>& confidences,
    float nms_threshold)
{
    std::vector<int> indices;
    std::vector<int> idxs(boxes.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    std::sort(idxs.begin(), idxs.end(), [&confidences](int i1, int i2){ return confidences[i1] > confidences[i2]; });

    while (!idxs.empty())
    {
        int idx = idxs.front();
        indices.push_back(idx);
        idxs.erase(idxs.begin());

        for (auto it = idxs.begin(); it != idxs.end();)
        {
            if (IoU(boxes[idx], boxes[*it]) > nms_threshold)
                it = idxs.erase(it);
            else
                ++it;
        }
    }

    return indices;
}

inline std::vector<float> RGB2NCHW_swaprb(const std::vector<uint8_t>& image_data, const int image_width, const int image_height, const int image_channels,
                                 const int input_width, const int input_height, const float scale_factor, const std::vector<float>& mean)
{
    std::vector<float> processed_data(input_width * input_height * 3);
    const float scale_x = static_cast<float>(image_width) / input_width;
    const float scale_y = static_cast<float>(image_height) / input_height;

    for (int y = 0; y < input_height; ++y) 
    {
        for (int x = 0; x < input_width; ++x) 
        {
            const int nearest_x = std::min(static_cast<int>(x * scale_x), image_width - 1);
            const int nearest_y = std::min(static_cast<int>(y * scale_y), image_height - 1);
            const int src_idx = (nearest_y * image_width + nearest_x) * image_channels;

            const int dst_idx_r =                                    (y * input_width + x); // Red channel index
            const int dst_idx_g = (    input_height * input_width) + (y * input_width + x); // Green channel index
            const int dst_idx_b = (2 * input_height * input_width) + (y * input_width + x); // Blue channel index

            // Swap R and B channels (RGB to BGR)
            processed_data[dst_idx_r] = (image_data[src_idx + 2] / 255.0f - mean[0]) * scale_factor; // Red
            processed_data[dst_idx_g] = (image_data[src_idx + 1] / 255.0f - mean[1]) * scale_factor; // Green
            processed_data[dst_idx_b] = (image_data[src_idx    ] / 255.0f - mean[2]) * scale_factor; // Blue
        }
    }

    return processed_data;
}

inline std::vector<float> RGB2NCHW_no_swaprb(const std::vector<uint8_t>& image_data, const int image_width, const int image_height, const int image_channels,
                                    const int input_width, const int input_height, const float scale_factor, const std::vector<float>& mean)
{
    std::vector<float> processed_data(input_width * input_height * 3);
    const float scale_x = static_cast<float>(image_width) / input_width;
    const float scale_y = static_cast<float>(image_height) / input_height;

    for (int y = 0; y < input_height; ++y) 
    {
        for (int x = 0; x < input_width; ++x) 
        {
            const int nearest_x = std::min(static_cast<int>(x * scale_x), image_width - 1);
            const int nearest_y = std::min(static_cast<int>(y * scale_y), image_height - 1);
            const int src_idx = (nearest_y * image_width + nearest_x) * image_channels;

            const int dst_idx_r =                                    (y * input_width + x); // Red channel index
            const int dst_idx_g = (    input_height * input_width) + (y * input_width + x); // Green channel index
            const int dst_idx_b = (2 * input_height * input_width) + (y * input_width + x); // Blue channel index

            // No channel swapping
            processed_data[dst_idx_r] = (image_data[src_idx    ] / 255.0f - mean[0]) * scale_factor; // Red
            processed_data[dst_idx_g] = (image_data[src_idx + 1] / 255.0f - mean[1]) * scale_factor; // Green
            processed_data[dst_idx_b] = (image_data[src_idx + 2] / 255.0f - mean[2]) * scale_factor; // Blue
        }
    }

    return processed_data;
}

inline std::vector<float> RGB2NCHW(const std::vector<uint8_t>& image_data, const int image_width, const int image_height, const int image_channels,
                            const int input_width, const int input_height, const float scale_factor = 1.0f, 
                            const std::vector<float>& mean = {0.0f, 0.0f, 0.0f}, const bool swap_rb = true)
{
    if (swap_rb) 
        return RGB2NCHW_swaprb(image_data, image_width, image_height, image_channels, input_width, input_height, scale_factor, mean);
    else 
        return RGB2NCHW_no_swaprb(image_data, image_width, image_height, image_channels, input_width, input_height, scale_factor, mean);
}