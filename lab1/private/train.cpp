#include "data_generator.h"
#include "mlp.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

void print_training_progress(int epoch, int total_epochs, double train_loss,
                             double val_loss) {
  std::cout << "--------------------------" << std::endl;
  std::cout << "in train" << std::endl;
  std::cout << "Epoch " << std::setw(3) << epoch + 1 << "/" << total_epochs
            << " - 训练损失: " << std::fixed << std::setprecision(6)
            << train_loss << " - 验证损失: " << std::fixed
            << std::setprecision(6) << val_loss << std::endl;
  std::cout << "--------------------------" << std::endl;
}

void save_training_history(const std::vector<double> &train_losses,
                           const std::vector<double> &val_losses,
                           const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "无法创建训练历史文件: " << filename << std::endl;
    return;
  }

  file << "epoch,train_loss,val_loss" << std::endl;
  for (size_t i = 0; i < train_losses.size(); ++i) {
    file << (i + 1) << "," << train_losses[i] << "," << val_losses[i]
         << std::endl;
  }

  file.close();
  std::cout << "训练历史已保存到: " << filename << std::endl;
}

int main() {
  std::cout << "=== MLP网络训练程序 ===" << std::endl;
  std::cout << "用于预测两个输入数字的乘积" << std::endl;
  std::cout << "========================================" << std::endl;

  // 设置随机种子
  const int seed = 42;

  // 创建数据生成器
  // DataGenerator data_gen(-100.0, 100.0, seed);
  DataGenerator data_gen(-1.0, 1.0, seed);

  // 生成训练和验证数据
  std::vector<std::vector<double>> train_inputs, train_targets;
  std::vector<std::vector<double>> val_inputs, val_targets;

  const int train_size = 8000;
  const int val_size = 2000;

  // 尝试读取已存在的数据文件
  bool data_loaded = false;

  // 检查训练数据文件是否存在
  std::ifstream train_file_check("train_data.csv");
  std::ifstream val_file_check("val_data.csv");

  if (train_file_check.good() && val_file_check.good()) {
    std::cout << "发现已存在的数据文件，正在加载..." << std::endl;

    // 加载训练数据
    train_file_check.close();
    std::ifstream train_file("train_data.csv");
    std::string line;
    std::getline(train_file, line); // 跳过标题行

    while (std::getline(train_file, line) && train_inputs.size() < train_size) {
      std::istringstream ss(line);
      std::string cell;
      std::vector<double> input(2);
      std::vector<double> target(1);

      std::getline(ss, cell, ',');
      input[0] = std::stod(cell);
      std::getline(ss, cell, ',');
      input[1] = std::stod(cell);
      std::getline(ss, cell, ',');
      target[0] = std::stod(cell);

      train_inputs.push_back(input);
      train_targets.push_back(target);
    }
    train_file.close();

    // 加载验证数据
    val_file_check.close();
    std::ifstream val_file("val_data.csv");
    std::getline(val_file, line); // 跳过标题行

    while (std::getline(val_file, line) && val_inputs.size() < val_size) {
      std::istringstream ss(line);
      std::string cell;
      std::vector<double> input(2);
      std::vector<double> target(1);

      std::getline(ss, cell, ',');
      input[0] = std::stod(cell);
      std::getline(ss, cell, ',');
      input[1] = std::stod(cell);
      std::getline(ss, cell, ',');
      target[0] = std::stod(cell);

      val_inputs.push_back(input);
      val_targets.push_back(target);
    }
    val_file.close();

    if (train_inputs.size() == train_size && val_inputs.size() == val_size) {
      // 对加载的数据进行标准化
      std::cout << "对加载的数据进行标准化..." << std::endl;
      // for (auto &input : train_inputs) {
      //   input[0] /= 100.0;
      //   input[1] /= 100.0;
      // }
      // for (auto &target : train_targets) {
      //   target[0] /= 10000.0;
      // }
      // for (auto &input : val_inputs) {
      //   input[0] /= 100.0;
      //   input[1] /= 100.0;
      // }
      // for (auto &target : val_targets) {
      //   target[0] /= 10000.0;
      // }

      data_loaded = true;
      std::cout << "数据加载成功！" << std::endl;
      std::cout << "训练集大小: " << train_inputs.size() << std::endl;
      std::cout << "验证集大小: " << val_inputs.size() << std::endl;
    } else {
      std::cout << "数据文件不完整，将重新生成..." << std::endl;
      train_inputs.clear();
      train_targets.clear();
      val_inputs.clear();
      val_targets.clear();
    }
  }

  // 如果数据文件不存在或不完整，则生成新数据
  if (!data_loaded) {
    std::cout << "生成新的训练和验证数据..." << std::endl;
    data_gen.generate_training_data(train_size, val_size, train_inputs,
                                    train_targets, val_inputs, val_targets);

    // 保存原始数据到文件（不缩放）
    {
      std::ofstream train_file("train_data.csv");
      if (!train_file.is_open()) {
        std::cerr << "无法创建训练数据文件: train_data.csv" << std::endl;
      } else {
        train_file << "input1,input2,target" << std::endl;
        for (size_t i = 0; i < train_inputs.size(); ++i) {
          train_file << train_inputs[i][0] << "," << train_inputs[i][1] << ","
                     << train_targets[i][0] << std::endl;
        }
        train_file.close();
        std::cout << "训练数据已保存到: train_data.csv" << std::endl;
      }
    }

    // 保存验证数据到文件
    {
      std::ofstream val_file("val_data.csv");
      if (!val_file.is_open()) {
        std::cerr << "无法创建验证数据文件: val_data.csv" << std::endl;
      } else {
        val_file << "input1,input2,target" << std::endl;
        for (size_t i = 0; i < val_inputs.size(); ++i) {
          val_file << val_inputs[i][0] << "," << val_inputs[i][1] << ","
                   << val_targets[i][0] << std::endl;
        }
        val_file.close();
        std::cout << "验证数据已保存到: val_data.csv" << std::endl;
      }
    }

    // ========== 原有的Min-Max标准化方法（已注释，用于对比） ==========
    // 数据标准化：使用Min-Max标准化（简单除法）
    // std::cout << "对数据进行Min-Max标准化..." << std::endl;

    // // 对于已知范围的均匀分布数据，直接除以最大值
    // const double input_scale = 100.0;    // 输入范围[-100, 100]
    // const double output_scale = 10000.0; // 输出范围[-10000, 10000]

    // std::cout << "输入缩放因子: " << input_scale << std::endl;
    // std::cout << "输出缩放因子: " << output_scale << std::endl;

    // // Min-Max标准化
    // for (auto &input : train_inputs) {
    //   input[0] /= input_scale; // 范围变为[-1, 1]
    //   input[1] /= input_scale;
    // }
    // for (auto &target : train_targets) {
    //   target[0] /= output_scale; // 范围变为[-1, 1]
    // }
    // for (auto &input : val_inputs) {
    //   input[0] /= input_scale;
    //   input[1] /= input_scale;
    // }
    // for (auto &target : val_targets) {
    //   target[0] /= output_scale;
    // }

    // 保存标准化参数
    // std::ofstream norm_file("normalization_params.txt");
    // if (norm_file.is_open()) {
    //   norm_file << input_scale << std::endl;
    //   norm_file << output_scale << std::endl;
    //   norm_file.close();
    //   std::cout << "标准化参数已保存到: normalization_params.txt" << std::endl;
    // }
    // ========== 原有的Min-Max标准化方法结束 ==========

    // ========== 新的对数标准化方法 ==========
    // 数据标准化：使用改进的对数标准化方法
    // std::cout << "对数据进行改进的对数标准化..." << std::endl;

    // // 对于乘法任务，使用对数标准化更合适
    // auto log_normalize = [](double x) -> double {
    //   if (x == 0)
    //     return 0;
    //   // 添加数值稳定性检查
    //   double abs_x = std::abs(x);
    //   if (abs_x < 1e-8)
    //     abs_x = 1e-8; // 避免log(0)
    //   if (abs_x > 1e6)
    //     abs_x = 1e6; // 避免log过大
    //   return std::log(abs_x) * (x >= 0 ? 1 : -1);
    // };

    // auto log_denormalize = [](double x) -> double {
    //   if (x == 0)
    //     return 0;
    //   return std::exp(std::abs(x)) * (x >= 0 ? 1 : -1);
    // };

    // // 对输入进行对数标准化
    // for (auto &input : train_inputs) {
    //   input[0] = log_normalize(input[0]);
    //   input[1] = log_normalize(input[1]);
    // }
    // for (auto &target : train_targets) {
    //   target[0] = log_normalize(target[0]);
    // }
    // for (auto &input : val_inputs) {
    //   input[0] = log_normalize(input[0]);
    //   input[1] = log_normalize(input[1]);
    // }
    // for (auto &target : val_targets) {
    //   target[0] = log_normalize(target[0]);
    // }

    // // 保存标准化方法标识
    // std::ofstream norm_file("normalization_params.txt");
    // if (norm_file.is_open()) {
    //   norm_file << "log_normalization" << std::endl;
    //   norm_file.close();
    //   std::cout << "对数标准化参数已保存到: normalization_params.txt"
    //             << std::endl;
    // }
  }

  // 生成测试数据
  std::vector<std::vector<double>> test_inputs, test_targets;
  // std::vector<std::vector<double>> dummy_val_inputs, dummy_val_targets;
  const int test_size = 10;
  // data_gen.generate_training_data(test_size, 0, test_inputs, test_targets,
                                  // dummy_val_inputs, dummy_val_targets);
  data_gen.generate_test_data(test_size, test_inputs, test_targets);

  // ========== 原有的测试数据标准化方法（已注释，用于对比） ==========
  // 对测试数据也进行稳定标准化
  // for (auto &input : test_inputs) {
  //   input[0] /= 100.0;
  //   input[1] /= 100.0;
  // }
  // for (auto &target : test_targets) {
  //   target[0] /= 10000.0;
  // }
  // ========== 原有的测试数据标准化方法结束 ==========

  // ========== 新的测试数据对数标准化方法 ==========
  // 对测试数据也进行对数标准化
  // auto log_normalize = [](double x) -> double {
  //   if (x == 0)
  //     return 0;
  //   // 添加数值稳定性检查
  //   double abs_x = std::abs(x);
  //   if (abs_x < 1e-8)
  //     abs_x = 1e-8; // 避免log(0)
  //   if (abs_x > 1e6)
  //     abs_x = 1e6; // 避免log过大
  //   return std::log(abs_x) * (x >= 0 ? 1 : -1);
  // };

  // for (auto &input : test_inputs) {
  //   input[0] = log_normalize(input[0]);
  //   input[1] = log_normalize(input[1]);
  // }
  // for (auto &target : test_targets) {
  //   target[0] = log_normalize(target[0]);
  // }

  // 创建MLP网络 - 使用更简单有效的架构
  std::vector<int> hidden_sizes = {256, 128, 64, 32};
  MLP model(2, hidden_sizes, 1, 0.1, 0.0001);

  // 打印网络架构
  model.print_architecture();
  std::cout << std::endl;

  // 训练参数 - 优化的超参数
  const int epochs = 2000;
  double learning_rate = 0.0005;
  const int batch_size = 64;

  // 记录训练历史
  std::vector<double> train_losses, val_losses;

  // 早停机制
  double best_val_loss = std::numeric_limits<double>::max();
  int patience = 300;
  int patience_counter = 0;

  // 损失监控机制
  double prev_val_loss = std::numeric_limits<double>::max();
  int loss_spike_counter = 0;

  auto start_time = std::chrono::high_resolution_clock::now();

  // 训练循环
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // 训练阶段
    model.train_batch(train_inputs, train_targets, 1, learning_rate,
                      batch_size);

    // 计算训练损失
    double train_loss = model.evaluate(train_inputs, train_targets);
    double val_loss = model.evaluate(val_inputs, val_targets);

    train_losses.push_back(train_loss);
    val_losses.push_back(val_loss);

    // 损失突增检测和处理
    if (epoch > 0 && val_loss > prev_val_loss * 5.0) {
      loss_spike_counter++;
      std::cout << "警告: 检测到损失突增！当前损失: " << val_loss
                << ", 前一轮损失: " << prev_val_loss << std::endl;

      if (loss_spike_counter >= 3) {
        // 如果连续3次突增，降低学习率
        learning_rate *= 0.7;
        loss_spike_counter = 0;
        std::cout << "损失突增，学习率降低至: " << learning_rate << std::endl;
      }
    } else {
      loss_spike_counter = 0;
    }

    prev_val_loss = val_loss;

    // 早停检查
    if (val_loss < best_val_loss) {
      best_val_loss = val_loss;
      patience_counter = 0;
      // 保存最佳模型
      model.save_model("best_model.txt");
    } else {
      patience_counter++;
    }

    // 打印进度
    if (epoch % 10 == 0 || epoch == epochs - 1) {
      print_training_progress(epoch, epochs, train_loss, val_loss);
    }

    // 早停
    if (patience_counter >= patience) {
      std::cout << "早停触发！验证损失在" << patience << "轮内没有改善。"
                << std::endl;
      break;
    }

    // 学习率调度：更温和的策略
    if (epoch > 0 && epoch % 200 == 0) {
      learning_rate *= 0.9; // 每200轮衰减10%，更温和
      std::cout << "学习率衰减至: " << learning_rate << std::endl;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

  std::cout << "----------------------------------------" << std::endl;
  std::cout << "训练完成！" << std::endl;
  std::cout << "总用时: " << duration.count() << " 秒" << std::endl;
  std::cout << "最终训练损失: " << std::fixed << std::setprecision(6)
            << train_losses.back() << std::endl;
  std::cout << "最终验证损失: " << std::fixed << std::setprecision(6)
            << val_losses.back() << std::endl;
  std::cout << "最佳验证损失: " << std::fixed << std::setprecision(6)
            << best_val_loss << std::endl;

  // 保存训练历史
  save_training_history(train_losses, val_losses, "training_history.csv");

  // 加载最佳模型进行测试
  std::cout << "加载最佳模型进行测试..." << std::endl;
  model.load_model("best_model.txt");

  // 测试一些特定案例（反缩放预测输出）
  std::cout << "\n测试特定案例:" << std::endl;
  std::cout << "输入1    输入2    真实乘积    预测值      误差" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  // ========== 原有的测试反标准化方法（已注释，用于对比） ==========
  for (size_t i = 0; i < test_inputs.size(); ++i) {
    // double a = test_inputs[i][0] * 100.0;               // 反标准化
    // double b = test_inputs[i][1] * 100.0;               // 反标准化
    // double true_product = test_targets[i][0] * 10000.0; // 反标准化
    double a = test_inputs[i][0];
    double b = test_inputs[i][1];
    double true_product = test_targets[i][0];

    std::vector<double> input = {
        test_inputs[i][0], test_inputs[i][1]};
    auto prediction = model.predict(input);

    // double pred_value = prediction[0] * 10000.0; // 反标准化预测结果
    double pred_value = prediction[0];
    double error = std::abs(pred_value - true_product);

    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << a
              << std::setw(8) << std::fixed << std::setprecision(2) << b
              << std::setw(12) << std::fixed << std::setprecision(4)
              << true_product << std::setw(12) << std::fixed
              << std::setprecision(4) << pred_value << std::setw(12)
              << std::fixed << std::setprecision(6) << error << std::endl;
  }
  // ========== 原有的测试反标准化方法结束 ==========

  // ========== 新的测试对数反标准化方法 ==========
  // 对数反标准化函数
  // auto log_denormalize = [](double x) -> double {
  //   if (x == 0)
  //     return 0;
  //   return std::exp(std::abs(x)) * (x >= 0 ? 1 : -1);
  // };

  // for (size_t i = 0; i < test_inputs.size(); ++i) {
  //   // 反标准化输入和输出（从对数空间）
  //   double a = log_denormalize(test_inputs[i][0]);
  //   double b = log_denormalize(test_inputs[i][1]);
  //   double true_product = log_denormalize(test_targets[i][0]);

  //   std::vector<double> input = {
  //       test_inputs[i][0], test_inputs[i][1]}; // 使用标准化后的输入进行预测
  //   auto prediction = model.predict(input);

  //   double pred_value = log_denormalize(prediction[0]); // 反标准化预测结果
  //   double error = std::abs(pred_value - true_product);

  //   std::cout << std::setw(8) << std::fixed << std::setprecision(2) << a
  //             << std::setw(8) << std::fixed << std::setprecision(2) << b
  //             << std::setw(12) << std::fixed << std::setprecision(4)
  //             << true_product << std::setw(12) << std::fixed
  //             << std::setprecision(4) << pred_value << std::setw(12)
  //             << std::fixed << std::setprecision(6) << error << std::endl;
  // }

  std::cout << "\n模型已保存为: best_model.txt" << std::endl;
  std::cout << "训练历史已保存为: training_history.csv" << std::endl;
  std::cout << "可以使用 validate.cpp 进行验证测试。" << std::endl;

  return 0;
}
