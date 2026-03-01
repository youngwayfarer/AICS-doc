#include "data_generator.h"
#include <iomanip>
#include <iostream>

DataGenerator::DataGenerator(double min_val, double max_val, int seed)
    : min_val_(min_val), max_val_(max_val), rng_(seed),
      uniform_dist_(min_val, max_val) {}

void DataGenerator::set_seed(int seed) { rng_.seed(seed); }

void DataGenerator::set_range(double min_val, double max_val) {
  if (min_val >= max_val) {
    throw std::invalid_argument("min_val must be less than max_val");
  }
  min_val_ = min_val;
  max_val_ = max_val;
  uniform_dist_ = std::uniform_real_distribution<double>(min_val, max_val);
}

double DataGenerator::random_value() { return uniform_dist_(rng_); }

std::pair<std::vector<double>, double> DataGenerator::generate_sample() {
  double a = random_value();
  double b = random_value();
  double product = multiply(a, b);

  return {{a, b}, product};
}

double DataGenerator::multiply(double a, double b) { return a * b; }

void DataGenerator::generate_training_data(
    int train_size, int val_size,
    std::vector<std::vector<double>> &train_inputs,
    std::vector<std::vector<double>> &train_targets,
    std::vector<std::vector<double>> &val_inputs,
    std::vector<std::vector<double>> &val_targets) {

  // 清空现有数据
  train_inputs.clear();
  train_targets.clear();
  val_inputs.clear();
  val_targets.clear();

  // 预分配空间
  train_inputs.reserve(train_size);
  train_targets.reserve(train_size);
  val_inputs.reserve(val_size);
  val_targets.reserve(val_size);

  std::cout << "生成训练数据..." << std::endl;
  std::cout << "训练集大小: " << train_size << std::endl;
  std::cout << "验证集大小: " << val_size << std::endl;
  std::cout << "数值范围: [" << min_val_ << ", " << max_val_ << "]"
            << std::endl;

  // 生成训练数据
  for (int i = 0; i < train_size; ++i) {
    auto sample = generate_sample();
    train_inputs.push_back(sample.first);
    train_targets.push_back({sample.second});
  }

  // 生成验证数据
  for (int i = 0; i < val_size; ++i) {
    auto sample = generate_sample();
    val_inputs.push_back(sample.first);
    val_targets.push_back({sample.second});
  }

  std::cout << "数据生成完成！" << std::endl;

  // 显示一些样本
  std::cout << "\n训练数据样本:" << std::endl;
  for (int i = 0; i < std::min(5, train_size); ++i) {
    std::cout << "样本 " << (i + 1) << ": [" << std::fixed
              << std::setprecision(3) << train_inputs[i][0] << ", "
              << train_inputs[i][1] << "] -> " << train_targets[i][0]
              << std::endl;
  }

  std::cout << "\n验证数据样本:" << std::endl;
  for (int i = 0; i < std::min(5, val_size); ++i) {
    std::cout << "样本 " << (i + 1) << ": [" << std::fixed
              << std::setprecision(3) << val_inputs[i][0] << ", "
              << val_inputs[i][1] << "] -> " << val_targets[i][0] << std::endl;
  }
}

void DataGenerator::generate_test_data(
    int test_size, std::vector<std::vector<double>> &test_inputs,
    std::vector<std::vector<double>> &test_targets) {

  // 清空现有数据
  test_inputs.clear();
  test_targets.clear();

  // 预分配空间
  test_inputs.reserve(test_size);
  test_targets.reserve(test_size);

  std::cout << "生成测试数据..." << std::endl;
  std::cout << "测试集大小: " << test_size << std::endl;

  // 生成测试数据
  for (int i = 0; i < test_size; ++i) {
    auto sample = generate_sample();
    test_inputs.push_back(sample.first);
    test_targets.push_back({sample.second});
  }

  std::cout << "测试数据生成完成！" << std::endl;

  // 显示一些样本
  std::cout << "\n测试数据样本:" << std::endl;
  for (int i = 0; i < std::min(5, test_size); ++i) {
    std::cout << "样本 " << (i + 1) << ": [" << std::fixed
              << std::setprecision(3) << test_inputs[i][0] << ", "
              << test_inputs[i][1] << "] -> " << test_targets[i][0]
              << std::endl;
  }
}
