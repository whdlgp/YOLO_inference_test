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
    model.init(ObjectDetection::InitParams(std::make_tuple(config_file.string(), weights_file.string(), names_file.string())), 0.5f, 0.4f, 416, 416);

    // Load the image
    cv::Mat img = cv::imread(image_file.string());
    ObjectDetection::Input input;
    input.chan = img.channels();
    input.height = img.rows;
    input.width = img.cols;
    input.data.resize(input.chan*input.height*input.width);
    std::memcpy(input.data.data(), img.data, input.data.size());

    auto outputs = model.infer(input);

    // Print the outputs
    for (const auto& output : outputs)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    DarknetModel::draw(img, outputs);
    
    // Display the output
    cv::namedWindow("output");
    cv::imshow("output", img);
    cv::waitKey(0);

    return 0;
}