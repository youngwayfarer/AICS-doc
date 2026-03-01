#include "mlp.h"
#include <algorithm>
#include <chrono>
#include <numeric>

MLP::MLP(int input_size, const std::vector<int> &hidden_sizes, int output_size,
         double dropout_rate, double l2_lambda)
    : input_size_(input_size), hidden_sizes_(hidden_sizes),
      output_size_(output_size), dropout_rate_(dropout_rate),
      l2_lambda_(l2_lambda), t_(0),
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
      biases_[i][j] = 0.0; // 偏置初始化为0
    }
    prev_size = hidden_sizes_[i];
  }

  // 最后一个隐藏层到输出层
  int last_layer = hidden_sizes_.size();
  weights_[last_layer].resize(output_size_);
  biases_[last_layer].resize(output_size_);
  for (int j = 0; j < output_size_; ++j) {
    weights_[last_layer][j].resize(prev_size);
    biases_[last_layer][j] = 0.0; // 偏置初始化为0
  }

  // 使用Xavier初始化权重
  initialize_weights();

  // 初始化Adam优化器参数
  initialize_adam();
}

void MLP::initialize_weights() {
  // Xavier初始化（适用于Sigmoid/Tanh激活函数）
  for (size_t layer = 0; layer < weights_.size(); ++layer) {
    int fan_in = (layer == 0) ? input_size_ : hidden_sizes_[layer - 1];
    int fan_out =
        (layer == weights_.size() - 1) ? output_size_ : hidden_sizes_[layer];
    double limit = std::sqrt(6.0 / (fan_in + fan_out));

    for (size_t neuron = 0; neuron < weights_[layer].size(); ++neuron) {
      for (size_t weight = 0; weight < weights_[layer][neuron].size();
           ++weight) {
        weights_[layer][neuron][weight] =
            uniform_dist_(rng_) * 2 * limit - limit;
      }
      // 偏置已经在构造函数中初始化为0，这里不需要重复设置
    }
  }

  // He初始化（适用于ReLU激活函数）
  // for (size_t layer = 0; layer < weights_.size(); ++layer) {
  //   int fan_in = (layer == 0) ? input_size_ : hidden_sizes_[layer - 1];
  //   int fan_out =
  //       (layer == weights_.size() - 1) ? output_size_ : hidden_sizes_[layer];

  //   // He初始化：标准差为sqrt(2/fan_in)，但针对小范围数据调整
  //   double std_dev = std::sqrt(2.0 / fan_in); // 增加权重初始化

  //   for (size_t neuron = 0; neuron < weights_[layer].size(); ++neuron) {
  //     for (size_t weight = 0; weight < weights_[layer][neuron].size();
  //          ++weight) {
  //       // 使用正态分布，均值为0，标准差为std_dev
  //       // 注意：normal_dist_的标准差是0.1，需要调整缩放因子
  //       weights_[layer][neuron][weight] = normal_dist_(rng_) * (std_dev /
  //       0.1);
  //     }
  //     // 偏置已经在构造函数中初始化为0，这里不需要重复设置
  //   }
  // }
}

double MLP::random_weight() { return normal_dist_(rng_); }

double MLP::relu(double x) {
  // 添加数值稳定性检查
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
  // 添加数值稳定性检查
  if (std::isnan(x)) {
    std::cerr << "警告: Sigmoid输入包含NaN: " << x << std::endl;
    return 0.0;
  }
  if (x < -709) { // 避免exp(-x)溢出
    return 0.0;
  } else if (x > 709) { // 避免exp(-x)下溢
    return 1.0;
  }
  return 1.0 / (1.0 + std::exp(-x));
}

double MLP::sigmoid_derivative(double x) {
  // 使用sigmoid函数的输出计算导数以提高数值稳定性
  if (std::isnan(x)) {
    std::cerr << "警告: Sigmoid导数输入包含NaN: " << x << std::endl;
    return 0.0;
  }
  double s = sigmoid(x);
  return s * (1.0 - s);
}

//! 一个潜在的“陷阱”是代码中没有对保留的激活进行缩放
//! 通常乘以 1 / (1 - dropout_rate_)
//! 这可能导致训练和推理时的激活值不一致
void MLP::apply_dropout(std::vector<double> &activations) {
  for (double &activation : activations) {
    if (uniform_dist_(rng_) < dropout_rate_) {
      activation = 0.0;
    }
  }
}

std::vector<double> MLP::forward(const std::vector<double> &input) {
  if (input.size() != input_size_) {
    throw std::invalid_argument("输入大小不匹配");
  }

  std::vector<double> current_input = input;

  // 通过隐藏层
  for (size_t layer = 0; layer < hidden_sizes_.size(); ++layer) {
    std::vector<double> next_input(hidden_sizes_[layer]);

    for (int neuron = 0; neuron < hidden_sizes_[layer]; ++neuron) {
      double sum = biases_[layer][neuron];
      for (size_t i = 0; i < current_input.size(); ++i) {
        sum += weights_[layer][neuron][i] * current_input[i];
      }
      // 尝试不同的激活函数
      next_input[neuron] = relu(sum);
      // next_input[neuron] = sigmoid(sum);
    }

    // 应用dropout（仅在训练时）
    // apply_dropout(next_input);

    current_input = next_input;
  }

  // 输出层
  std::vector<double> output(output_size_);
  for (int neuron = 0; neuron < output_size_; ++neuron) {
    double sum = biases_[hidden_sizes_.size()][neuron];
    for (size_t i = 0; i < current_input.size(); ++i) {
      sum += weights_[hidden_sizes_.size()][neuron][i] * current_input[i];
    }
    output[neuron] = sum; // 输出层不使用激活函数
  }

  return output;
}

std::vector<std::vector<double>>
MLP::forward_batch(const std::vector<std::vector<double>> &inputs) {
  std::vector<std::vector<double>> outputs;
  outputs.reserve(inputs.size());

  //! 目前仍然是串行处理，可以考虑用多线程加速
  for (const auto &input : inputs) {
    outputs.push_back(forward(input));
  }

  return outputs;
}

std::vector<double> MLP::predict(const std::vector<double> &input) {
  return forward(input);
}

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

  // 使用标准MSE损失，不进行放大
  // 放大应该在评估时进行，而不是在训练时
  return sum / prediction.size();
}

std::vector<double>
MLP::mse_loss_derivative(const std::vector<double> &prediction,
                         const std::vector<double> &target) {
  std::vector<double> derivative(prediction.size());
  for (size_t i = 0; i < prediction.size(); ++i) {
    // 使用标准MSE导数，不进行放大
    derivative[i] = 2.0 * (prediction[i] - target[i]) / prediction.size();
  }
  return derivative;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
MLP::forward_with_cache(const std::vector<double> &input) {
  std::vector<std::vector<double>> activations;
  std::vector<std::vector<double>> linear_sums; // 新增：缓存线性和
  std::vector<double> current_input = input;

  // 输入层激活（无线性和）
  activations.push_back(input);

  // 通过隐藏层
  for (size_t layer = 0; layer < hidden_sizes_.size(); ++layer) {
    std::vector<double> next_input(hidden_sizes_[layer]);
    std::vector<double> layer_linear_sums(
        hidden_sizes_[layer]); // 当前层的线性和

    for (int neuron = 0; neuron < hidden_sizes_[layer]; ++neuron) {
      double sum = biases_[layer][neuron];
      for (size_t i = 0; i < current_input.size(); ++i) {
        sum += weights_[layer][neuron][i] * current_input[i];
      }
      layer_linear_sums[neuron] = sum; // 缓存线性和
      next_input[neuron] = relu(sum);
      // next_input[neuron] = sigmoid(sum);
    }

    linear_sums.push_back(layer_linear_sums); // 添加到线性和向量
    activations.push_back(next_input);
    current_input = next_input;
  }

  // 输出层（无激活函数，线性和即为输出）
  std::vector<double> output(output_size_);
  std::vector<double> output_linear_sums(output_size_);
  for (int neuron = 0; neuron < output_size_; ++neuron) {
    double sum = biases_[hidden_sizes_.size()][neuron];
    for (size_t i = 0; i < current_input.size(); ++i) {
      sum += weights_[hidden_sizes_.size()][neuron][i] * current_input[i];
    }
    output_linear_sums[neuron] = sum;
    output[neuron] = sum;
  }

  linear_sums.push_back(output_linear_sums); // 输出层的线性和
  activations.push_back(output);
  return {activations, linear_sums};
}

void MLP::backward(
    const std::vector<double> &input, const std::vector<double> &target,
    const std::vector<std::vector<double>> &activations,
    const std::vector<std::vector<double>> &linear_sums, // 新增参数：线性和缓存
    double learning_rate) {

  // 计算输出层梯度
  std::vector<double> output_grad =
      mse_loss_derivative(activations.back(), target);

  // 梯度裁剪
  double max_grad = 2.0; // 适中的梯度裁剪阈值
  for (double &grad : output_grad) {
    if (std::abs(grad) > max_grad) {
      grad = (grad > 0) ? max_grad : -max_grad;
    }
  }

  // 反向传播
  std::vector<double> current_grad = output_grad;

  // 从输出层开始反向传播
  for (int layer = static_cast<int>(weights_.size()) - 1; layer >= 0; --layer) {
    // 获取当前层的输入激活
    int input_idx = layer; // activations[layer] 是当前层的输入
    const std::vector<double> &layer_input = activations[input_idx];
    std::vector<double> prev_grad(layer_input.size(), 0.0);

    for (size_t neuron = 0; neuron < weights_[layer].size(); ++neuron) {
      // 更新偏置
      biases_[layer][neuron] -= learning_rate * current_grad[neuron];

      // 更新权重并计算前一层梯度
      for (size_t weight = 0; weight < weights_[layer][neuron].size();
           ++weight) {
        // 更新权重
        weights_[layer][neuron][weight] -=
            learning_rate * current_grad[neuron] * layer_input[weight];

        // 计算前一层梯度
        if (layer > 0) {
          // 计算激活函数的导数
          double activation_derivative = 1.0;
          if (layer < static_cast<int>(hidden_sizes_.size())) {
            // 对于隐藏层，使用缓存的线性和计算 ReLU 导数
            activation_derivative = relu_derivative(linear_sums[layer][neuron]);
            // activation_derivative =
            // sigmoid_derivative(linear_sums[layer][neuron]);
          }

          // 累加到前一层对应神经元的梯度
          prev_grad[weight] += current_grad[neuron] *
                               weights_[layer][neuron][weight] *
                               activation_derivative;
        }
      }
    }

    // 对前一层梯度进行裁剪
    for (double &grad : prev_grad) {
      if (std::abs(grad) > max_grad) {
        grad = (grad > 0) ? max_grad : -max_grad;
      }
    }

    current_grad = prev_grad;
  }
}

void MLP::train(const std::vector<std::vector<double>> &inputs,
                const std::vector<std::vector<double>> &targets, int epochs,
                double learning_rate, int batch_size) {

  if (inputs.size() != targets.size()) {
    throw std::invalid_argument("输入和目标数量不匹配");
  }

  for (int epoch = 0; epoch < epochs; ++epoch) {
    double total_loss = 0.0;
    int batch_count = 0;

    // 随机打乱数据
    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);

    // 批次训练
    for (size_t i = 0; i < inputs.size(); i += batch_size) {
      size_t end = std::min(i + static_cast<size_t>(batch_size), inputs.size());

      for (size_t j = i; j < end; ++j) {
        size_t idx = indices[j];
        auto [activations, linear_sums] =
            forward_with_cache(inputs[idx]); // 解包返回的 pair
        // backward(inputs[idx], targets[idx], activations, linear_sums,
        //          learning_rate); // 传递线性和
        backward_with_adam(inputs[idx], targets[idx], activations, linear_sums,
                           learning_rate); // 使用Adam优化器

        total_loss += mse_loss(activations.back(), targets[idx]);
        batch_count++;
      }
    }

    double avg_loss = total_loss / batch_count;

    if (epoch % 10 == 0 || epoch == epochs - 1) {
      std::cout << "Epoch " << std::setw(3) << epoch + 1 << "/" << epochs
                << " - 平均损失: " << std::fixed << std::setprecision(6)
                << avg_loss << std::endl;
    }
  }

  // std::cout << "训练完成！" << std::endl;
}

void MLP::train_batch(const std::vector<std::vector<double>> &inputs,
                      const std::vector<std::vector<double>> &targets,
                      int epochs, double learning_rate, int batch_size) {

  if (inputs.size() != targets.size()) {
    throw std::invalid_argument("输入和目标数量不匹配");
  }

  for (int epoch = 0; epoch < epochs; ++epoch) {
    double total_loss = 0.0;
    int batch_count = 0;

    // 随机打乱数据
    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);

    // 真正的批次训练
    for (size_t i = 0; i < inputs.size(); i += batch_size) {
      size_t end = std::min(i + static_cast<size_t>(batch_size), inputs.size());
      int current_batch_size = end - i;

      // 准备批次数据
      std::vector<std::vector<double>> batch_inputs;
      std::vector<std::vector<double>> batch_targets;
      std::vector<std::vector<std::vector<double>>> batch_activations;
      std::vector<std::vector<std::vector<double>>> batch_linear_sums;

      batch_inputs.reserve(current_batch_size);
      batch_targets.reserve(current_batch_size);
      batch_activations.reserve(current_batch_size);
      batch_linear_sums.reserve(current_batch_size);

      // 批次前向传播
      for (size_t j = i; j < end; ++j) {
        size_t idx = indices[j];
        batch_inputs.push_back(inputs[idx]);
        batch_targets.push_back(targets[idx]);

        auto [activations, linear_sums] = forward_with_cache(inputs[idx]);
        batch_activations.push_back(activations);
        batch_linear_sums.push_back(linear_sums);

        total_loss += mse_loss(activations.back(), targets[idx]);
      }

      // 批次反向传播和权重更新
      backward_batch_with_adam(batch_inputs, batch_targets, batch_activations,
                               batch_linear_sums, learning_rate);

      batch_count++;
    }

    double avg_loss = total_loss / batch_count;

    if (epoch % 10 == 0 || epoch == epochs - 1) {
      std::cout << "Epoch " << std::setw(3) << epoch + 1 << "/" << epochs
                << " - 平均损失: " << std::fixed << std::setprecision(6)
                << avg_loss << std::endl;
    }
  }
}

void MLP::backward_batch_with_adam(
    const std::vector<std::vector<double>> &inputs,
    const std::vector<std::vector<double>> &targets,
    const std::vector<std::vector<std::vector<double>>> &batch_activations,
    const std::vector<std::vector<std::vector<double>>> &batch_linear_sums,
    double learning_rate, double beta1, double beta2, double epsilon) {

  int batch_size = inputs.size();

  // Adam时间步更新
  t_++;

  // 计算偏差修正
  double bias_correction1 = 1.0 - std::pow(beta1, t_);
  double bias_correction2 = 1.0 - std::pow(beta2, t_);

  // 计算批次平均梯度 - 修复：每个权重都有独立的梯度
  std::vector<std::vector<std::vector<double>>> avg_weight_gradients(
      weights_.size());
  std::vector<std::vector<double>> avg_bias_gradients(weights_.size());

  for (size_t layer = 0; layer < weights_.size(); ++layer) {
    avg_weight_gradients[layer].resize(weights_[layer].size());
    avg_bias_gradients[layer].resize(weights_[layer].size());
    for (size_t neuron = 0; neuron < weights_[layer].size(); ++neuron) {
      avg_weight_gradients[layer][neuron].resize(weights_[layer][neuron].size(),
                                                 0.0);
      avg_bias_gradients[layer][neuron] = 0.0;
    }
  }

  // 对每个样本计算梯度并累加
  for (int sample = 0; sample < batch_size; ++sample) {
    // 计算当前样本的输出层梯度
    std::vector<double> output_grad =
        mse_loss_derivative(batch_activations[sample].back(), targets[sample]);

    // 梯度裁剪
    double max_grad = 5.0; // 降低梯度裁剪阈值
    for (double &grad : output_grad) {
      if (std::abs(grad) > max_grad) {
        grad = (grad > 0) ? max_grad : -max_grad;
      }
    }

    // 反向传播计算梯度
    std::vector<double> current_grad = output_grad;

    for (int layer = static_cast<int>(weights_.size()) - 1; layer >= 0;
         --layer) {
      int input_idx = layer;
      const std::vector<double> &layer_input =
          batch_activations[sample][input_idx];
      std::vector<double> prev_grad(layer_input.size(), 0.0);

      for (size_t neuron = 0; neuron < weights_[layer].size(); ++neuron) {
        // 计算偏置梯度
        double grad_bias = current_grad[neuron];
        avg_bias_gradients[layer][neuron] += grad_bias;

        // 计算权重梯度
        for (size_t weight = 0; weight < weights_[layer][neuron].size();
             ++weight) {
          double grad_weight = current_grad[neuron] * layer_input[weight];
          avg_weight_gradients[layer][neuron][weight] += grad_weight;
        }

        // 计算前一层梯度
        if (layer > 0) {
          double activation_derivative = 1.0;
          if (layer < static_cast<int>(hidden_sizes_.size())) {
            activation_derivative =
                relu_derivative(batch_linear_sums[sample][layer][neuron]);
          }

          for (size_t weight = 0; weight < weights_[layer][neuron].size();
               ++weight) {
            prev_grad[weight] += current_grad[neuron] *
                                 weights_[layer][neuron][weight] *
                                 activation_derivative;
          }
        }
      }

      // 梯度裁剪
      for (double &grad : prev_grad) {
        if (std::abs(grad) > max_grad) {
          grad = (grad > 0) ? max_grad : -max_grad;
        }
      }

      current_grad = prev_grad;
    }
  }

  // 计算平均梯度并更新权重
  for (size_t layer = 0; layer < weights_.size(); ++layer) {
    for (size_t neuron = 0; neuron < weights_[layer].size(); ++neuron) {
      // 更新偏置
      double avg_grad_bias = avg_bias_gradients[layer][neuron] / batch_size;

      m_biases_[layer][neuron] =
          beta1 * m_biases_[layer][neuron] + (1.0 - beta1) * avg_grad_bias;
      v_biases_[layer][neuron] = beta2 * v_biases_[layer][neuron] +
                                 (1.0 - beta2) * avg_grad_bias * avg_grad_bias;

      double m_hat_bias = m_biases_[layer][neuron] / bias_correction1;
      double v_hat_bias = v_biases_[layer][neuron] / bias_correction2;

      biases_[layer][neuron] -=
          learning_rate * m_hat_bias / (std::sqrt(v_hat_bias) + epsilon);

      // 更新权重
      for (size_t weight = 0; weight < weights_[layer][neuron].size();
           ++weight) {
        double avg_grad_weight =
            avg_weight_gradients[layer][neuron][weight] / batch_size;

        m_weights_[layer][neuron][weight] =
            beta1 * m_weights_[layer][neuron][weight] +
            (1.0 - beta1) * avg_grad_weight;
        v_weights_[layer][neuron][weight] =
            beta2 * v_weights_[layer][neuron][weight] +
            (1.0 - beta2) * avg_grad_weight * avg_grad_weight;

        double m_hat_weight =
            m_weights_[layer][neuron][weight] / bias_correction1;
        double v_hat_weight =
            v_weights_[layer][neuron][weight] / bias_correction2;

        // 添加L2正则化项
        double l2_grad = l2_lambda_ * weights_[layer][neuron][weight];
        weights_[layer][neuron][weight] -=
            learning_rate *
            (m_hat_weight / (std::sqrt(v_hat_weight) + epsilon) + l2_grad);
      }
    }
  }
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

double MLP::evaluate_real_error(const std::vector<std::vector<double>> &inputs,
                                const std::vector<std::vector<double>> &targets,
                                double output_scale) {
  if (inputs.size() != targets.size()) {
    throw std::invalid_argument("输入和目标数量不匹配");
  }

  double total_error = 0.0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto prediction = forward(inputs[i]);
    // 计算真实误差（反标准化后）
    double real_error =
        std::abs(prediction[0] * output_scale - targets[i][0] * output_scale);
    total_error += real_error;
  }

  return total_error / inputs.size();
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
      count += weights_[layer][neuron].size() + 1; // 权重 + 偏置
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
  std::cout << "Dropout率: " << dropout_rate_ << std::endl;
  std::cout << "总参数数量: " << get_parameter_count() << std::endl;
}

void MLP::initialize_adam() {
  // 初始化Adam优化器的动量参数
  m_weights_.resize(weights_.size());
  m_biases_.resize(biases_.size());
  v_weights_.resize(weights_.size());
  v_biases_.resize(biases_.size());

  for (size_t layer = 0; layer < weights_.size(); ++layer) {
    m_weights_[layer].resize(weights_[layer].size());
    m_biases_[layer].resize(biases_[layer].size());
    v_weights_[layer].resize(weights_[layer].size());
    v_biases_[layer].resize(biases_[layer].size());

    for (size_t neuron = 0; neuron < weights_[layer].size(); ++neuron) {
      m_weights_[layer][neuron].resize(weights_[layer][neuron].size(), 0.0);
      m_biases_[layer][neuron] = 0.0;
      v_weights_[layer][neuron].resize(weights_[layer][neuron].size(), 0.0);
      v_biases_[layer][neuron] = 0.0;
    }
  }

  t_ = 0;
}

void MLP::backward_with_adam(
    const std::vector<double> &input, const std::vector<double> &target,
    const std::vector<std::vector<double>> &activations,
    const std::vector<std::vector<double>> &linear_sums, double learning_rate,
    double beta1, double beta2, double epsilon) {

  // 计算输出层梯度
  std::vector<double> output_grad =
      mse_loss_derivative(activations.back(), target);

  // 梯度裁剪
  double max_grad = 10.0; // 放宽梯度裁剪阈值，允许更大的梯度
  for (double &grad : output_grad) {
    if (std::abs(grad) > max_grad) {
      grad = (grad > 0) ? max_grad : -max_grad;
    }
  }

  // Adam时间步更新
  t_++;

  // 计算偏差修正
  double bias_correction1 = 1.0 - std::pow(beta1, t_);
  double bias_correction2 = 1.0 - std::pow(beta2, t_);

  // 反向传播
  std::vector<double> current_grad = output_grad;

  // 从输出层开始反向传播
  for (int layer = static_cast<int>(weights_.size()) - 1; layer >= 0; --layer) {
    // 获取当前层的输入激活
    int input_idx = layer;
    const std::vector<double> &layer_input = activations[input_idx];
    std::vector<double> prev_grad(layer_input.size(), 0.0);

    for (size_t neuron = 0; neuron < weights_[layer].size(); ++neuron) {
      // 计算偏置梯度
      double grad_bias = current_grad[neuron];

      // Adam更新偏置
      m_biases_[layer][neuron] =
          beta1 * m_biases_[layer][neuron] + (1.0 - beta1) * grad_bias;
      v_biases_[layer][neuron] = beta2 * v_biases_[layer][neuron] +
                                 (1.0 - beta2) * grad_bias * grad_bias;

      double m_hat_bias = m_biases_[layer][neuron] / bias_correction1;
      double v_hat_bias = v_biases_[layer][neuron] / bias_correction2;

      biases_[layer][neuron] -=
          learning_rate * m_hat_bias / (std::sqrt(v_hat_bias) + epsilon);

      // 更新权重并计算前一层梯度
      for (size_t weight = 0; weight < weights_[layer][neuron].size();
           ++weight) {
        // 计算权重梯度
        double grad_weight = current_grad[neuron] * layer_input[weight];

        // Adam更新权重（添加L2正则化）
        m_weights_[layer][neuron][weight] =
            beta1 * m_weights_[layer][neuron][weight] +
            (1.0 - beta1) * grad_weight;
        v_weights_[layer][neuron][weight] =
            beta2 * v_weights_[layer][neuron][weight] +
            (1.0 - beta2) * grad_weight * grad_weight;

        double m_hat_weight =
            m_weights_[layer][neuron][weight] / bias_correction1;
        double v_hat_weight =
            v_weights_[layer][neuron][weight] / bias_correction2;

        // 添加L2正则化项
        double l2_grad = l2_lambda_ * weights_[layer][neuron][weight];
        weights_[layer][neuron][weight] -=
            // learning_rate * m_hat_weight / (std::sqrt(v_hat_weight) +
            // epsilon);
            learning_rate *
            (m_hat_weight / (std::sqrt(v_hat_weight) + epsilon) + l2_grad);

        // 计算前一层梯度
        if (layer > 0) {
          // 计算激活函数的导数
          double activation_derivative = 1.0;
          if (layer < static_cast<int>(hidden_sizes_.size())) {
            activation_derivative = relu_derivative(linear_sums[layer][neuron]);
          }

          // 累加到前一层对应神经元的梯度
          prev_grad[weight] += current_grad[neuron] *
                               weights_[layer][neuron][weight] *
                               activation_derivative;
        }
      }
    }

    // 对前一层梯度进行裁剪
    for (double &grad : prev_grad) {
      if (std::abs(grad) > max_grad) {
        grad = (grad > 0) ? max_grad : -max_grad;
      }
    }

    current_grad = prev_grad;
  }
}
