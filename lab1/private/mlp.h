#ifndef MLP_H
#define MLP_H

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

class MLP {
public:
  // 构造函数
  MLP(int input_size = 2, const std::vector<int> &hidden_sizes = {64, 32, 16},
      int output_size = 1, double dropout_rate = 0.1, double l2_lambda = 0.001);

  // 析构函数
  ~MLP() = default;

  // 前向传播
  std::vector<double> forward(const std::vector<double> &input);

  // 批量前向传播
  std::vector<std::vector<double>>
  forward_batch(const std::vector<std::vector<double>> &inputs);

  // 预测方法
  std::vector<double> predict(const std::vector<double> &input);

  // 训练方法
  void train(const std::vector<std::vector<double>> &inputs,
             const std::vector<std::vector<double>> &targets, int epochs = 100,
             double learning_rate = 0.001, int batch_size = 32);

  // 真正的批次训练方法
  void train_batch(const std::vector<std::vector<double>> &inputs,
                   const std::vector<std::vector<double>> &targets,
                   int epochs = 100, double learning_rate = 0.001,
                   int batch_size = 32);

  // 评估方法
  double evaluate(const std::vector<std::vector<double>> &inputs,
                  const std::vector<std::vector<double>> &targets);
  double evaluate_real_error(const std::vector<std::vector<double>> &inputs,
                             const std::vector<std::vector<double>> &targets,
                             double output_scale = 10000.0);

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
  double l2_lambda_; // L2正则化参数

  // 网络权重和偏置
  std::vector<std::vector<std::vector<double>>>
      weights_;                             // [layer][neuron][input]
  std::vector<std::vector<double>> biases_; // [layer][neuron]

  // Adam优化器参数
  std::vector<std::vector<std::vector<double>>> m_weights_; // 一阶矩估计
  std::vector<std::vector<double>> m_biases_;
  std::vector<std::vector<std::vector<double>>> v_weights_; // 二阶矩估计
  std::vector<std::vector<double>> v_biases_;
  int t_; // 时间步

  // 随机数生成器
  std::mt19937 rng_;
  std::uniform_real_distribution<double> uniform_dist_;
  std::normal_distribution<double> normal_dist_;

  // 辅助方法
  void initialize_weights();
  double relu(double x);
  double relu_derivative(double x);
  double sigmoid(double x);
  double sigmoid_derivative(double x);
  double random_weight();
  void apply_dropout(std::vector<double> &activations);
  std::vector<double> softmax(const std::vector<double> &x);

  // 训练相关
  std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
  forward_with_cache(const std::vector<double> &input);
  void backward(const std::vector<double> &input,
                const std::vector<double> &target,
                const std::vector<std::vector<double>> &activations,
                const std::vector<std::vector<double>> &linear_sums,
                double learning_rate);

  // Adam优化器方法
  void initialize_adam();
  void backward_with_adam(const std::vector<double> &input,
                          const std::vector<double> &target,
                          const std::vector<std::vector<double>> &activations,
                          const std::vector<std::vector<double>> &linear_sums,
                          double learning_rate, double beta1 = 0.9,
                          double beta2 = 0.999, double epsilon = 1e-8);

  // 批次Adam优化器方法
  void backward_batch_with_adam(
      const std::vector<std::vector<double>> &inputs,
      const std::vector<std::vector<double>> &targets,
      const std::vector<std::vector<std::vector<double>>> &batch_activations,
      const std::vector<std::vector<std::vector<double>>> &batch_linear_sums,
      double learning_rate, double beta1 = 0.9, double beta2 = 0.999,
      double epsilon = 1e-8);

  // 损失函数
  double mse_loss(const std::vector<double> &prediction,
                  const std::vector<double> &target);
  double real_mse_loss(const std::vector<double> &prediction,
                       const std::vector<double> &target,
                       double output_scale = 10000.0);
  std::vector<double> mse_loss_derivative(const std::vector<double> &prediction,
                                          const std::vector<double> &target);
};

#endif // MLP_H
