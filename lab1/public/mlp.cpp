#include "mlp.h"
#include <algorithm>
#include <chrono>
#include <numeric>

MLP::MLP(int input_size, const std::vector<int> &hidden_sizes, int output_size,
         double dropout_rate)
    : input_size_(input_size), hidden_sizes_(hidden_sizes),
      output_size_(output_size), dropout_rate_(dropout_rate),
      rng_(std::chrono::steady_clock::now().time_since_epoch().count()),
      uniform_dist_(0.0, 1.0), normal_dist_(0.0, 0.1) {

  // 初始化网络结构
  weights_.resize(hidden_sizes_.size() + 1);
  biases_.resize(hidden_sizes_.size() + 1);

  // 输入层到第一个隐藏层
  int prev_size = input_size_;
  for (size_t i = 0; i < hidden_sizes_.size(); ++i) {
    weights_[i].resize(hidden_sizes_[i]);
    biases_[i].resize(hidden_sizes_[i]);
    for (int j = 0; j < hidden_sizes_[i]; ++j) {
      weights_[i][j].resize(prev_size);
      biases_[i][j] = 0.0;
    }
    prev_size = hidden_sizes_[i];
  }

  // 最后一个隐藏层到输出层
  int last_layer = hidden_sizes_.size();
  weights_[last_layer].resize(output_size_);
  biases_[last_layer].resize(output_size_);
  for (int j = 0; j < output_size_; ++j) {
    weights_[last_layer][j].resize(prev_size);
    biases_[last_layer][j] = 0.0;
  }

  initialize_weights();
}

void MLP::initialize_weights() {
  // TODO: 正确地初始化参数
}

double MLP::relu(double x) {
  if (std::isnan(x) || std::isinf(x)) {
    std::cerr << "警告: ReLU输入包含NaN或Inf: " << x << std::endl;
    return 0.0;
  }
  return std::max(0.0, x);
}

double MLP::relu_derivative(double x) {
  if (std::isnan(x) || std::isinf(x)) {
    std::cerr << "警告: ReLU导数输入包含NaN或Inf: " << x << std::endl;
    return 0.0;
  }
  return x > 0 ? 1.0 : 0.0;
}

double MLP::sigmoid(double x) {
  if (std::isnan(x)) {
    std::cerr << "警告: Sigmoid输入包含NaN: " << x << std::endl;
    return 0.0;
  }
  if (x < -709) {
    return 0.0;
  } else if (x > 709) {
    return 1.0;
  }
  return 1.0 / (1.0 + std::exp(-x));
}

double MLP::sigmoid_derivative(double x) {
  if (std::isnan(x)) {
    std::cerr << "警告: Sigmoid导数输入包含NaN: " << x << std::endl;
    return 0.0;
  }
  double s = sigmoid(x);
  return s * (1.0 - s);
}

// TODO: 可以实现 dropout

double MLP::mse_loss(const std::vector<double> &prediction,
                     const std::vector<double> &target) {
  if (prediction.size() != target.size()) {
    throw std::invalid_argument("预测和目标大小不匹配");
  }

  double sum = 0.0;
  for (size_t i = 0; i < prediction.size(); ++i) {
    double diff = prediction[i] - target[i];
    sum += diff * diff;
  }

  return sum / prediction.size();
}

std::vector<double>
MLP::mse_loss_derivative(const std::vector<double> &prediction,
                         const std::vector<double> &target) {
  std::vector<double> derivative(prediction.size());
  // TODO: 计算 MSE 损失函数的导数

  return derivative;
}

std::vector<double> MLP::forward(const std::vector<double> &input) {
  if (input.size() != input_size_) {
    throw std::invalid_argument("输入大小不匹配");
  }

  std::vector<double> current_input = input;

  // 这段代码可能会是性能瓶颈，请思考应该怎么设计合理的数据结构以及相应的算法来提高性能
  // 仅思考即可，我们将在后续的实验中展示当前主流的做法
  for (size_t layer = 0; layer < hidden_sizes_.size(); ++layer) {
    std::vector<double> next_input(hidden_sizes_[layer]);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int neuron = 0; neuron < hidden_sizes_[layer]; ++neuron) {
      double sum = biases_[layer][neuron];
      for (size_t i = 0; i < current_input.size(); ++i) {
        sum += weights_[layer][neuron][i] * current_input[i];
      }
      // 可以尝试不同的激活函数
      next_input[neuron] = relu(sum);
    }

    current_input = next_input;
  }

  std::vector<double> output(output_size_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int neuron = 0; neuron < output_size_; ++neuron) {
    double sum = biases_[hidden_sizes_.size()][neuron];
    for (size_t i = 0; i < current_input.size(); ++i) {
      sum += weights_[hidden_sizes_.size()][neuron][i] * current_input[i];
    }
    output[neuron] = sum;
  }

  return output;
}

std::vector<double> MLP::predict(const std::vector<double> &input) {
  return forward(input);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
MLP::forward_with_cache(const std::vector<double> &input) {
  // TODO: 实现前向传播，并保存反向传播必需的相关值，如每层的激活值和线性和
}

void MLP::backward(const std::vector<double> &input,
                   const std::vector<double> &target,
                   const std::vector<std::vector<double>> &activations,
                   const std::vector<std::vector<double>> &linear_sums,
                   double learning_rate) {
  // TODO: 实现反向传播
}

void MLP::train(const std::vector<std::vector<double>> &inputs,
                const std::vector<std::vector<double>> &targets,
                double learning_rate, int batch_size) {

  // TODO: 实现模型训练方法
}

double MLP::evaluate(const std::vector<std::vector<double>> &inputs,
                     const std::vector<std::vector<double>> &targets) {

  if (inputs.size() != targets.size()) {
    throw std::invalid_argument("输入和目标数量不匹配");
  }

  double total_loss = 0.0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto prediction = forward(inputs[i]);
    total_loss += mse_loss(prediction, targets[i]);
  }

  return total_loss / inputs.size();
}

void MLP::save_model(const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("无法打开文件进行保存: " + filename);
  }

  // 保存网络结构
  file << input_size_ << " " << hidden_sizes_.size() << " " << output_size_
       << " " << dropout_rate_ << std::endl;
  for (int size : hidden_sizes_) {
    file << size << " ";
  }
  file << std::endl;

  // 保存权重和偏置
  for (size_t layer = 0; layer < weights_.size(); ++layer) {
    for (size_t neuron = 0; neuron < weights_[layer].size(); ++neuron) {
      for (size_t weight = 0; weight < weights_[layer][neuron].size();
           ++weight) {
        file << weights_[layer][neuron][weight] << " ";
      }
      file << biases_[layer][neuron] << " ";
    }
    file << std::endl;
  }

  file.close();
  std::cout << "模型已保存到: " << filename << std::endl;
}

void MLP::load_model(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("无法打开文件进行加载: " + filename);
  }

  // 加载网络结构
  int hidden_layers_count;
  file >> input_size_ >> hidden_layers_count >> output_size_ >> dropout_rate_;

  hidden_sizes_.resize(hidden_layers_count);
  for (int i = 0; i < hidden_layers_count; ++i) {
    file >> hidden_sizes_[i];
  }

  // 重新初始化权重和偏置
  weights_.resize(hidden_sizes_.size() + 1);
  biases_.resize(hidden_sizes_.size() + 1);

  int prev_size = input_size_;
  for (size_t i = 0; i < hidden_sizes_.size(); ++i) {
    weights_[i].resize(hidden_sizes_[i]);
    biases_[i].resize(hidden_sizes_[i]);
    for (int j = 0; j < hidden_sizes_[i]; ++j) {
      weights_[i][j].resize(prev_size);
      for (int k = 0; k < prev_size; ++k) {
        file >> weights_[i][j][k];
      }
      file >> biases_[i][j];
    }
    prev_size = hidden_sizes_[i];
  }

  // 输出层
  int last_layer = hidden_sizes_.size();
  weights_[last_layer].resize(output_size_);
  biases_[last_layer].resize(output_size_);
  for (int j = 0; j < output_size_; ++j) {
    weights_[last_layer][j].resize(prev_size);
    for (int k = 0; k < prev_size; ++k) {
      file >> weights_[last_layer][j][k];
    }
    file >> biases_[last_layer][j];
  }

  file.close();
  std::cout << "模型已从文件加载: " << filename << std::endl;
}

int MLP::get_parameter_count() const {
  int count = 0;
  for (size_t layer = 0; layer < weights_.size(); ++layer) {
    for (size_t neuron = 0; neuron < weights_[layer].size(); ++neuron) {
      count += weights_[layer][neuron].size() + 1;
    }
  }
  return count;
}

void MLP::print_architecture() const {
  std::cout << "MLP网络架构:" << std::endl;
  std::cout << "输入层: " << input_size_ << " 个神经元" << std::endl;

  for (size_t i = 0; i < hidden_sizes_.size(); ++i) {
    std::cout << "隐藏层 " << (i + 1) << ": " << hidden_sizes_[i] << " 个神经元"
              << std::endl;
  }

  std::cout << "输出层: " << output_size_ << " 个神经元" << std::endl;
  std::cout << "总参数数量: " << get_parameter_count() << std::endl;
}
