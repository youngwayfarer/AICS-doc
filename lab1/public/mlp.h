#ifndef MLP_H
#define MLP_H

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

class MLP {
public:
  MLP(int input_size = 2, const std::vector<int> &hidden_sizes = {64, 32, 16},
      int output_size = 1, double dropout_rate = 0.1);

  ~MLP() = default;

  std::vector<double> forward(const std::vector<double> &input);

  std::vector<double> predict(const std::vector<double> &input);

  void train(const std::vector<std::vector<double>> &inputs,
             const std::vector<std::vector<double>> &targets,
             double learning_rate = 0.001, int batch_size = 32);

  double evaluate(const std::vector<std::vector<double>> &inputs,
                  const std::vector<std::vector<double>> &targets);

  // 保存模型
  void save_model(const std::string &filename);

  // 加载模型
  void load_model(const std::string &filename);

  // 获取参数数量
  int get_parameter_count() const;

  // 打印网络结构
  void print_architecture() const;

private:
  // 网络结构参数
  int input_size_;
  std::vector<int> hidden_sizes_;
  int output_size_;
  double dropout_rate_;

  // 网络权重和偏置
  std::vector<std::vector<std::vector<double>>>
      weights_;                             // [layer][neuron][input]
  std::vector<std::vector<double>> biases_; // [layer][neuron]

  // 随机数生成器
  std::mt19937 rng_;
  std::uniform_real_distribution<double> uniform_dist_;
  std::normal_distribution<double> normal_dist_;

  void initialize_weights();
  double relu(double x);
  double relu_derivative(double x);
  double sigmoid(double x);
  double sigmoid_derivative(double x);
  void apply_dropout(std::vector<double> &activations);
  std::vector<double> softmax(const std::vector<double> &x);

  std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
  forward_with_cache(const std::vector<double> &input);
  void backward(const std::vector<double> &input,
                const std::vector<double> &target,
                const std::vector<std::vector<double>> &activations,
                const std::vector<std::vector<double>> &linear_sums,
                double learning_rate);

  double mse_loss(const std::vector<double> &prediction,
                  const std::vector<double> &target);
  std::vector<double> mse_loss_derivative(const std::vector<double> &prediction,
                                          const std::vector<double> &target);
};

#endif // MLP_H
