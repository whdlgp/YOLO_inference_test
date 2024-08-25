#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <vector>
#include <fstream>
#include <filesystem>
#include <any>

#include "module/dnn_model_darknet.hpp"

namespace fs = std::filesystem;


int main()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path config_file = models_dir / "yolov4" / "yolov4.cfg";
    fs::path weights_file = models_dir / "yolov4" / "yolov4.weights";
    fs::path names_file = models_dir / "yolov4" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Initialize Darknet model
    DarknetModel model;
    model.init(DNNModel::InitParams(std::make_tuple(config_file.string(), weights_file.string(), names_file.string())), 0.5, 0.4);

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Preprocess the image
    cv::Mat blob = model.preprocess(img);

    // Record start time for inference
    auto total_start = std::chrono::steady_clock::now();

    // Perform inference
    std::vector<cv::Mat> detections = model.infer(blob);

    // Record end time for inference
    auto dnn_start = std::chrono::steady_clock::now();

    // Postprocess the inference results
    model.postprocess(img, detections);

    // Draw the results
    model.draw_results(img);

    auto total_end = std::chrono::steady_clock::now();

    // Calculate and display FPS
    float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_start - total_start).count();
    float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    std::ostringstream stats_ss;
    stats_ss << std::fixed << std::setprecision(2);
    stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
    auto stats = stats_ss.str();

    int baseline;
    auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
    cv::rectangle(img, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(img, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

    // Display the output
    cv::namedWindow("output");
    cv::imshow("output", img);
    cv::waitKey(0);

    return 0;
}