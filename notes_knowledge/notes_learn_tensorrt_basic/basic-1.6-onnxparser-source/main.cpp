//
// Created by junjie on 7/25/23.
//

// TensorRT include
// 编译用的头文件
#include "NvInfer.h"
#include "NvOnnxParser.h"
//#include "onnx_tensorrt/NvOnnxParser.h"


// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

// ------------------------------ 1. Logger -------------------------------------------------//
inline const char *severity_string(nvinfer1::ILogger::Severity t) {
    switch (t) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:
            return "error";
        case nvinfer1::ILogger::Severity::kWARNING:
            return "warning";
        case nvinfer1::ILogger::Severity::kINFO:
            return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE:
            return "verbose";
        default:
            return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override {
        if (severity <= Severity::kINFO) {
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if (severity == Severity::kWARNING) {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            } else if (severity <= Severity::kERROR) {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            } else {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

//------------------------------------------------------------------------------------------------//


//------------------------------------------- 构建模型 --------------------------------------------//
bool build_model() {
    TRTLogger logger;

    //构建模型需要的组件
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    /**
     * 通过onnxparse解析结果填充到network
     *
     */

    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile("demo.onnx", 1)) {
        printf("Failed to parse demo.onnx\n");
        return false;
    }
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    //配置输入，多输入需要配置多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    int input_channel = input_tensor->getDimensions().d[1];

    // 配置输入的最小、最优、最大的范围
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr) {
        printf("build engine failed. \n");
        return false;
    }

    //模型序列号TRT，存储为文件
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen("engine.trt", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;
}

int main() {
    build_model();
    return 0;
}