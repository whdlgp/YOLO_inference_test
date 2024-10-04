#include <vector>
#include <fstream>
#include <filesystem>
#include <any>

#include <opencv2/opencv.hpp>

#include "module/make_models.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

// Draw Detection with OpenCV
void draw(const std::vector<DetectedOutput>& detections, cv::Mat img, std::string window_name)
{
    // colors for bounding boxes
    const int num_colors = 4;
    const cv::Scalar colors[4] = {
        {0, 255, 255},
        {255, 255, 0},
        {0, 255, 0},
        {255, 0, 0}
    };

    for (size_t i = 0; i < detections.size(); ++i)
    {
        const auto color = colors[detections[i].class_id % num_colors];
        const auto& rect = cv::Rect(detections[i].bbox_x, detections[i].bbox_y, detections[i].bbox_width, detections[i].bbox_height);
        cv::rectangle(img, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

        std::ostringstream label_ss;
        label_ss << detections[i].class_name << ": " << std::fixed << std::setprecision(2) << detections[i].confidence;
        auto label = label_ss.str();
        
        int baseline;
        auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(img, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
        cv::putText(img, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
    }
    
    // Display the output
    cv::namedWindow(window_name);
    cv::imshow(window_name, img);
    cv::waitKey(0);
}

void test_onnxruntime()
{
    /*
    // Load class names
    std::vector<std::string> class_names;
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

    int num_classes = static_cast<int>(class_names.size());

    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;
    const float CONF_THRESHOLD = 0.5f;
    const float IOU_THRESHOLD = 0.4f;

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, onnx_file.string().c_str(), session_options);

    // Get input/output tensor type
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    ONNXTensorElementDataType input_tensor_type = input_type_info.GetTensorTypeAndShapeInfo().GetElementType();
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    ONNXTensorElementDataType output_tensor_type = output_type_info.GetTensorTypeAndShapeInfo().GetElementType();

    // Get input/output layer name
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);

    std::cout << "Input Name: " << input_name.get() << std::endl;
    std::cout << "Output Name: " << output_name.get() << std::endl;

    // Preprocess the image
    cv::Mat image = cv::imread(image_file.string());
    std::vector<uint8_t> image_data(image.data, image.data + (image.total() * image.elemSize()));
    std::vector<float> preprocessed = RGB2NCHW(image_data, image.cols, image.rows, image.channels(), INPUT_WIDTH, INPUT_HEIGHT);

    // Prepare input tensor
    std::vector<int64_t> input_shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH}; // NCHW format
    size_t input_tensor_size = INPUT_WIDTH * INPUT_HEIGHT * 3;

    // Create input tensor object from data values
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, preprocessed.data(), input_tensor_size, input_shape.data(), input_shape.size());

    // Run inference
    std::vector<const char*> input_names = {input_name.get()};
    std::vector<const char*> output_names = {output_name.get()};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // Get output tensor shape
    Ort::Value& output_tensor = output_tensors.front(); // Assuming single output
    Ort::TensorTypeAndShapeInfo output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_tensor_info.GetShape();

    // Print output tensor shape
    std::cout << "Output Tensor Shape: ";
    for (const auto& dim : output_shape)
    {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // Convert the output tensor to vector of floats
    float* output_data = output_tensor.GetTensorMutableData<float>();
    std::vector<float> output_vector(output_data, output_data + output_tensor_info.GetElementCount());
    std::cout << "output vector len : " << output_vector.size() << std::endl;

    int dst_width = image.cols;
    int dst_height = image.rows;
    int inference_width = INPUT_WIDTH;
    int inference_height = INPUT_HEIGHT;
    float confidence_threshold = CONF_THRESHOLD;
    float nms_threshold = IOU_THRESHOLD;

    std::vector<DetectedOutput> final_results;

    // Assume all mats in inputs has same shape
    // Initialize vectors to store NMS results
    std::vector<std::vector<BoundingBox>> boxes_per_class(num_classes);
    std::vector<std::vector<float>> confidences_per_class(num_classes);
    std::vector<std::vector<int>> indices(num_classes);

    // [1, BBOX+NUM_CLASSES, NUM_BOXES], BBox data = (Xc, Yc, W, H)
    const int num_boxes = output_vector.size() / (4 + num_classes);
    const float scale_x = static_cast<float>(dst_width) / inference_width;
    const float scale_y = static_cast<float>(dst_height) / inference_height;

    std::cout << "Num boxes : " << num_boxes << std::endl;

    const float* x_centers = output_vector.data();
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
    */
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov8" / "yolov8l.onnx";
    fs::path names_file = models_dir / "yolov8" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_onnxruntime();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "ONNXRuntime");
}

void test_yolov4_darknet()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    fs::path config_file = models_dir / "yolov4" / "yolov4.cfg";
    fs::path weights_file = models_dir / "yolov4" / "yolov4.weights";
    fs::path names_file = models_dir / "yolov4" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["config_path"] = config_file.string();
    init_param["weights_path"] = weights_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_darknet();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v4");
}

void test_yolov5_onnx()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov5" / "yolov5l.onnx";
    fs::path names_file = models_dir / "yolov5" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_yolov5();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v5");
}

void test_yolov6_onnx()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov6" / "yolov6l.onnx";
    fs::path names_file = models_dir / "yolov6" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_yolov6();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v6");
}

void test_yolov7_darknet()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path config_file = models_dir / "yolov7" / "yolov7.cfg";
    fs::path weights_file = models_dir / "yolov7" / "yolov7.weights";
    fs::path names_file = models_dir / "yolov7" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["config_path"] = config_file.string();
    init_param["weights_path"] = weights_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_darknet();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v7");
}

void test_yolov8_onnx()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov8" / "yolov8l.onnx";
    fs::path names_file = models_dir / "yolov8" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_yolov8();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v8");
}

void test_yolov9_onnx()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov9" / "yolov9-c-converted.onnx";
    fs::path names_file = models_dir / "yolov9" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_yolov9();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v9");
}

void test_yolov10_onnx()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov10" / "yolov10s.onnx";
    fs::path names_file = models_dir / "yolov10" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_yolov10();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v10");
}

int main()
{
    /*
    test_yolov4_darknet();
    test_yolov5_onnx();
    test_yolov6_onnx();
    test_yolov7_darknet();
    test_yolov8_onnx();
    test_yolov9_onnx();
    test_yolov10_onnx();
    */
    test_onnxruntime();
    return 0;
}