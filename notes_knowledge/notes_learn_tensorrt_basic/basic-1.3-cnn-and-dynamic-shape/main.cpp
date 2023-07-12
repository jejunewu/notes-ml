//
// Created by junjie on 7/11/23.
//

// tensorRT include
#include "NvInfer.h"
#include "NvInferRuntime.h"

//  cuda include
#include "cuda_runtime.h"

//  system include
#include "stdio.h"
#include "math.h"
#include "iostream"
#include "fstream"
#include "vector"

using namespace std;

/**
*   定义logger
*/
class TRTLogger : public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override {
        if (severity <= Severity::kINFO) {
            printf("%d: %s\n", severity, msg);
        }
    }
};

/**
* 定义TRT权重生成函数
*/
nvinfer1::Weights make_weights(float *ptr, int n) {
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}


/**
* 定义模型
*/
bool build_model() {
    TRTLogger logger;//捕捉 warning & info

    //---------------------- 1. 定义builder, config, network -------------------//
    //定义 TRT builder
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    //创建配置，TRT模型只能在特定的配置下运行
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    //创建网络定义，设置1表示采用显性 batch_size
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    // 构建一个模型
    /*
        Network definition:

        image
          |
        conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
          |
        relu
          |
        prob
    */
    //---------------------- 2. 定义网络模型结构、输出等 --------------------------//
    const int num_input = 1;
    const int num_output = 1;
    //权重数组
    float layer1_weight_values[] = {
            1.0, 2.0, 3.1,
            0.1, 0.1, 0.1,
            0.2, 0.2, 0.2
    };
    //偏置
    float layer1_bias_values[] = {0.0};

    // 如果要使用动态shape，必须让NetworkDefinition的维度定义为-1，in_channel是固定的
    nvinfer1::ITensor *input = network->addInput("image", nvinfer1::DataType::kFLOAT,
                                                 nvinfer1::Dims4(-1, num_input, -1, -1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 9);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, 1);
    //增加卷积层 (过时警告)
    auto layer1 = network->addConvolution(*input, num_output, nvinfer1::DimsHW(3, 3), layer1_weight, layer1_bias);
    layer1->setPadding(nvinfer1::DimsHW(1, 1));
    //增加激活函数 relu
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kRELU);
    // 将我们需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));
    


}