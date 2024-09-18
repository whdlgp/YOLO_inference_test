#include "util.hpp"

// Intersection over Union (IoU) calculation
float IoU(const BoundingBox& box1, const BoundingBox& box2)
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
std::vector<int> nonMaximumSuppression(
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
