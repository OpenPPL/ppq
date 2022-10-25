#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include "logger.h"
#include <map>
#include <nlohmann/json.hpp>

// stuff we know about the network and the input/output blobs
static const int INPUT_C = 1;
static const int INPUT_H = 32;
static const int INPUT_W = 32;

static const int OUTPUT_SIZE = 10;

const char* INPUT_BLOB_NAME = "input.1";
const char* OUTPUT_BLOB_NAME = "32";
int BatchSize = 1;

using json = nlohmann::json;
using samplesCommon::SampleUniquePtr;
using namespace nvinfer1;


// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

//!
//! \brief  Sets custom dynamic range for network tensors
//!
bool setDynamicRange(SampleUniquePtr<nvinfer1::INetworkDefinition>& network, const std::string& json_file)
{
    std::unordered_map<std::string, float> mPerTensorDynamicRangeMap;

    std::ifstream fin(json_file);
    assert(fin.is_open() && "Unable to load json file.");
    
    json range_dict;
    fin >> range_dict;
    fin.close();

    for (auto& act_quant : range_dict) 
    {
        for (auto& el : act_quant.items()) 
        {
            mPerTensorDynamicRangeMap[el.key()] = el.value();
        }
    }

    // set dynamic range for network input tensors
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        std::string tName = network->getInput(i)->getName();
        if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
        {
            sample::gLogInfo << "Start to write quantization parameters: " << tName << std::endl;
            if (!network->getInput(i)->setDynamicRange(
                    -mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName)))
            {
                return false;
            }
        }
        else
        {
            sample::gLogWarning << "Missing dynamic range for tensor: " << tName << std::endl;
        }
    }

    // set dynamic range for layer output tensors
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        auto lyr = network->getLayer(i);
        for (int j = 0, e = lyr->getNbOutputs(); j < e; ++j)
        {
            std::string tName = lyr->getOutput(j)->getName();
            if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
            {
                sample::gLogInfo << "Start to write quantization parameters: " << tName << std::endl;

                // Calibrator generated dynamic range for network tensor can be overriden or set using below API
                if (!lyr->getOutput(j)->setDynamicRange(
                        -mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName)))
                {
                    return false;
                }
            }

            else
            {
                sample::gLogWarning << "Missing dynamic range for tensor: " << tName << std::endl;
            }
        }
    }
    
    for (auto iter = mPerTensorDynamicRangeMap.begin(); iter != mPerTensorDynamicRangeMap.end(); ++iter)
    {
        sample::gLogInfo << "Tensor: " << iter->first << ". Max Absolute Dynamic Range: " << iter->second
                            << std::endl;
    }
    return true;
}
                                                                                                 // DataType::kFLOAT
bool createLenetEngine(const std::string& weight_file, const std::string& json_file, const std::string& engine_path)
{
    // Open json file
    std::ifstream quant_param(json_file);
    assert(quant_param.is_open() && "Unable to load json file.");

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) 
    {
        return false;
    }
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) 
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) 
    {
        return false;
    }

    // Create input tensor of shape { 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims4{BatchSize, INPUT_C, INPUT_H, INPUT_W});
    assert(data);

    // Add convolution layer with 6 outputs and a 5x5 filter.
    std::map<std::string, Weights> weightMap = loadWeights(weight_file);
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 6, DimsHW{5, 5}, weightMap["conv1.weight"], weightMap["conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->getOutput(0)->setName("onnx::Relu_11");

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    relu1->getOutput(0)->setName("onnx::Pad_12");

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->getOutput(0)->setName("onnx::AveragePool_14");


    // Add second convolution layer with 16 outputs and a 5x5 filter.
    IConvolutionLayer* conv2 = network->addConvolutionNd(*pool1->getOutput(0), 16, DimsHW{5, 5}, weightMap["conv2.weight"], weightMap["conv2.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->getOutput(0)->setName("input");

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    relu2->getOutput(0)->setName("onnx::Relu_16");

    // Add second max pooling layer with stride of 2x2 and kernel size of 2x2>
    IPoolingLayer* pool2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool2);
    pool2->setStrideNd(DimsHW{2, 2});
    pool2->getOutput(0)->setName("onnx::Pad_17");

    // Add fully connected layer
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 120, weightMap["fc1.weight"], weightMap["fc1.bias"]);
    assert(fc1);
    fc1->getOutput(0)->setName("onnx::AveragePool_19");

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu3 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    relu3->getOutput(0)->setName("onnx::Relu_27");

    // Add second fully connected layer
    IFullyConnectedLayer* fc2 = network->addFullyConnected(*relu3->getOutput(0), 84, weightMap["fc2.weight"], weightMap["fc2.bias"]);
    assert(fc2);
    fc2->getOutput(0)->setName("onnx::Gemm_28");


    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu4 = network->addActivation(*fc2->getOutput(0), ActivationType::kRELU);
    assert(relu4);
    relu4->getOutput(0)->setName("onnx::Relu_29");

    // Add third fully connected layer
    IFullyConnectedLayer* fc3 = network->addFullyConnected(*relu4->getOutput(0), OUTPUT_SIZE, weightMap["fc3.weight"], weightMap["fc3.bias"]);
    assert(fc3);
    fc3->getOutput(0)->setName("onnx::Gemm_30");
    

    // Add softmax layer to determine the probability.
    ISoftMaxLayer* prob = network->addSoftMax(*fc3->getOutput(0));
    assert(prob);
    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*prob->getOutput(0));

    // Iterate through network layers.
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        // Write output tensors of a layer to the file.
        for (int j = 0; j < network->getLayer(i)->getNbOutputs(); ++j)
        {
            std::string tName = network->getLayer(i)->getOutput(j)->getName();
            std::cout << "TensorName: " << tName << std::endl;
        }
    }


    builder->setMaxBatchSize(BatchSize);

    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setMaxWorkspaceSize(1_GiB);
    config->setFlag(BuilderFlag::kINT8);

    // set INT8 Per Tensor Dynamic range
    if (!setDynamicRange(network, json_file))
    {
        sample::gLogError << "Unable to set per-tensor dynamic range." << std::endl;
        return false;
    }


    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream) 
    {
        return false;
    }
    config->setProfileStream(*profileStream);
    
    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine) 
    {
        return false;
    }

    std::ofstream planFile;
    planFile.open(engine_path);
    IHostMemory* serializedEngine = mEngine->serialize();
    planFile.write((char*)serializedEngine->data(), serializedEngine->size());
    planFile.close();
    serializedEngine->destroy();


    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return true;
}

int main(int argc, char** argv)
{
    std::string weight_file = argv[1];    // Weight parameter
    std::string quant_json_file = argv[2];         // Quantization parameter json
    std::string engine_file = argv[3];        // tensorrt engine file

    bool success = createLenetEngine(weight_file, quant_json_file, engine_file);
    assert(success);

    return 0;
}
