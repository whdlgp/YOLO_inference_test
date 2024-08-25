#include "dnn_model_darknet.hpp"

#include <fstream>

void DarknetModel::init(const InitParams &model_files, float confidence_threshold, float nms_threshold)
{
    // Set confidence threshold, NMS threshold, and number of colors
    this->confidence_threshold = confidence_threshold;
    this->nms_threshold = nms_threshold;

    if (model_files.type() == typeid(std::tuple<std::string, std::string, std::string>))
    {
        auto [config_path, weights_path, names_file] = std::any_cast<std::tuple<std::string, std::string, std::string>>(model_files);
        net = cv::dnn::readNetFromDarknet(config_path, weights_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

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

        // Set the number of classes based on the size of class_names
        num_classes = static_cast<int>(class_names.size());
    }
    else
    {
        throw std::logic_error("Darknet model requires config, weights, and class names paths");
    }
}

// Preprocess the image
cv::Mat DarknetModel::preprocess(const cv::Mat &image)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
    return blob;
}

// Perform inference
std::vector<cv::Mat> DarknetModel::infer(const cv::Mat &blob)
{
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    auto output_names = net.getUnconnectedOutLayersNames();
    net.forward(outs, output_names);
    return outs;
}

// Postprocess the inference results
void DarknetModel::postprocess(const cv::Mat &frame, const std::vector<cv::Mat> &outs)
{
    boxes.clear();
    class_ids.clear();
    confidences.clear();

    // Initialize vectors to store NMS results
    std::vector<std::vector<int>> indices(num_classes);
    std::vector<std::vector<cv::Rect>> boxes_per_class(num_classes);
    std::vector<std::vector<float>> confidences_per_class(num_classes);

    for (const auto& output : outs)
    {
        const auto num_boxes = output.rows;
        for (int i = 0; i < num_boxes; i++)
        {
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            cv::Rect rect(x - width / 2, y - height / 2, width, height);

            for (int c = 0; c < num_classes; c++)
            {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence >= confidence_threshold)
                {
                    boxes_per_class[c].push_back(rect);
                    confidences_per_class[c].push_back(confidence);
                }
            }
        }
    }

    // Perform NMS and aggregate the results
    for (int c = 0; c < num_classes; c++)
    {
        cv::dnn::NMSBoxes(boxes_per_class[c], confidences_per_class[c], confidence_threshold, nms_threshold, indices[c]);

        for (auto idx : indices[c])
        {
            boxes.push_back(boxes_per_class[c][idx]);
            class_ids.push_back(c);
            confidences.push_back(confidences_per_class[c][idx]);
        }
    }
}

// Utility function to draw the results
void DarknetModel::draw_results(cv::Mat &frame)
{
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const auto color = colors[class_ids[i] % num_colors];
        const auto& rect = boxes[i];
        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

        std::ostringstream label_ss;
        label_ss << class_names[class_ids[i]] << ": " << std::fixed << std::setprecision(2) << confidences[i];
        auto label = label_ss.str();
        
        int baseline;
        auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
        cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
    }
}
