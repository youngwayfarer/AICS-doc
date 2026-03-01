#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

class DataGenerator {
public:
  // 构造函数
  DataGenerator(double min_val = -10.0, double max_val = 10.0, int seed = 42);

  // 生成训练数据
  void generate_training_data(int train_size, int val_size,
                              std::vector<std::vector<double>> &train_inputs,
                              std::vector<std::vector<double>> &train_targets,
                              std::vector<std::vector<double>> &val_inputs,
                              std::vector<std::vector<double>> &val_targets);

  // 生成测试数据
  void generate_test_data(int test_size,
                          std::vector<std::vector<double>> &test_inputs,
                          std::vector<std::vector<double>> &test_targets);

  // 生成单个样本
  std::pair<std::vector<double>, double> generate_sample();

  // 计算两个数的乘积
  double multiply(double a, double b);

  // 设置随机种子
  void set_seed(int seed);

  // 设置数值范围
  void set_range(double min_val, double max_val);

private:
  double min_val_;
  double max_val_;
  std::mt19937 rng_;
  std::uniform_real_distribution<double> uniform_dist_;

  // 生成随机数
  double random_value();
};

#endif // DATA_GENERATOR_H
