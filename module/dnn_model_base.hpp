#pragma once

#include <any>
#include <vector>
#include <cstdint>
#include <string>

#include "util.hpp"

// Color space of decoded image buffer
enum class Color : int
{
    BGR = 0,
    RGB,
};

// Basic input image type
class Image
{
public:
    // Buffer of image
    std::vector<uint8_t> data;
    // size of image
    int width = 0, height = 0, chan = 0;
    // Color space of decoded image buffer
    Color color = Color::BGR;

    int total() { return width*height*chan; }
};

// Basic output type
template <typename T>
class Matrix
{
public:
    // Buffer of image
    std::vector<T> data;
    // Shape of image, ex) if YOLO v5, [1, 25200, 85] 
    std::vector<int> shape;
};

// Object Detection output type
class DetectedOutput
{
public:
    int bbox_x;
    int bbox_y;
    int bbox_width;
    int bbox_height;
    int class_id;
    std::string class_name;
    float confidence;
};

// Basic Interface for Backend of DNN inference
template <typename InferenceOutType>
class BackendBase
{
public:
    virtual ~BackendBase() {}

    // Initialize the model
    virtual void init(nlohmann::json init_params) = 0;

    // Perform inference
    virtual std::vector<Matrix<InferenceOutType>> infer(Image& image) = 0;
protected:
    bool is_init = false;
};

// Basic Interface for Post Process of Object Detection models
template <typename InferenceOutType>
class PostProcessor
{
public:
    virtual ~PostProcessor() {}

    // Initialize the Post Processor
    virtual void init(nlohmann::json init_params) = 0;
    
    // Perform Post Process
    virtual std::vector<DetectedOutput> run(std::vector<Matrix<InferenceOutType>>& raw_output, int dst_width, int dst_height) = 0;
};

// Default Object Detection Interface
class DetectionBase
{
public:
    // Need Backend and Post Processor
    inline DetectionBase(std::unique_ptr<BackendBase<float>> backend, std::unique_ptr<PostProcessor<float>> postproc)
        :backend_(std::move(backend)), postproc_(std::move(postproc))
        {}
    
    inline ~DetectionBase() {}

    // Initialize the Model and Post Processor
    inline void init(nlohmann::json init_params)
    {
        backend_->init(init_params);
        postproc_->init(init_params);
    }

    // Perform inference and Post process
    inline std::vector<DetectedOutput> run(Image& image)
    {
        auto infer_results = backend_->infer(image);
        auto detections = postproc_->run(infer_results, image.width, image.height);

        return detections;
    }

private:
    // Backend
    std::unique_ptr<BackendBase<float>> backend_;
    
    // Post Processor
    std::unique_ptr<PostProcessor<float>> postproc_;
};

// Basic object detection model class interface
class ObjectDetection
{
public:
    virtual ~ObjectDetection() {}

    // Input form
    class Input
    {
    public:
        int width;
        int height;
        int chan;
        std::vector<uint8_t> data;
    };

    // Output form
    class Output
    {
    public:
        int bbox_x;
        int bbox_y;
        int bbox_width;
        int bbox_height;
        int class_id;
        std::string class_name;
        float confidence;
    };

    // Initialize the model
    virtual void init(nlohmann::json init_params)
    {
        if(!check_and_get(init_params, "confidence_threshold", this->confidence_threshold))
            throw std::invalid_argument("wrong confidence_threshold");
        if(!check_and_get(init_params, "nms_threshold", this->nms_threshold))
            throw std::invalid_argument("wrong nms_threshold");
        if(!check_and_get(init_params, "inference_width", this->inference_width))
            throw std::invalid_argument("wrong inference_width");
        if(!check_and_get(init_params, "inference_height", this->inference_height))
            throw std::invalid_argument("wrong inference_height");

        // Add your model imple initialization
    }

    // Perform inference
    virtual std::vector<Output> infer(Input& input) = 0;
protected:
    float confidence_threshold = 0.5;
    float nms_threshold = 0.4;
    int inference_width = 608;
    int inference_height = 608;

    std::vector<std::string> class_names;
    int num_classes = 0;
};
