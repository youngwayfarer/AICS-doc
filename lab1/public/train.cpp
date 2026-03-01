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
  std::cout << "Epoch " << std::setw(3) << epoch + 1 << "/" << total_epochs
            << " - 训练损失: " << std::fixed << std::setprecision(6)
            << train_loss << " - 验证损失: " << std::fixed
            << std::setprecision(6) << val_loss << std::endl;
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

// TODO: 实现完整的训练程序，你可以添加任意机制，如早停、学习率衰减等
// 注意！当前训练程序并未保存训练好的模型
// 请根据你的训练策略来决定何时保存最佳模型
// Hint：使用 model.save_model("best_model.txt") 来保存模型
int main() {
  std::cout << "=== MLP网络训练程序 ===" << std::endl;
  std::cout << "用于预测两个输入数字的乘积" << std::endl;
  std::cout << "========================================" << std::endl;

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
    std::getline(val_file, line);

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
      data_loaded = true;
      std::cout << "数据加载成功！" << std::endl;
      std::cout << "训练集大小: " << train_inputs.size() << std::endl;
      std::cout << "验证集大小: " << val_inputs.size() << std::endl;
    } else {
      std::cout << "数据集文件不完整，程序退出。" << std::endl;
      return 1;
    }
  }

  if (!data_loaded) {
    std::cout << "数据集文件不存在，程序退出。" << std::endl;
    return 1;
  }

  // TODO: 创建合适的网络结构，并选择合适的超参数
  std::vector<int> hidden_sizes = {};
  MLP model(2, hidden_sizes, 1, );

  model.print_architecture();
  std::cout << std::endl;

  // TODO: 选择合适的训练参数
  const int epochs = ;
  double learning_rate = ;
  const int batch_size = ;

  // 记录训练历史
  std::vector<double> train_losses, val_losses;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int epoch = 0; epoch < epochs; ++epoch) {
    model.train(train_inputs, train_targets, learning_rate, batch_size);

    // 计算损失
    double train_loss = model.evaluate(train_inputs, train_targets);
    double val_loss = model.evaluate(val_inputs, val_targets);

    train_losses.push_back(train_loss);
    val_losses.push_back(val_loss);

    // 打印进度
    if (epoch % 10 == 0 || epoch == epochs - 1) {
      print_training_progress(epoch, epochs, train_loss, val_loss);
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

  // 保存训练历史
  save_training_history(train_losses, val_losses, "training_history.csv");

  std::cout << "\n模型已保存为: best_model.txt" << std::endl;
  std::cout << "训练历史已保存为: training_history.csv" << std::endl;
  std::cout << "可以使用 validate.cpp 进行验证测试。" << std::endl;

  return 0;
}
