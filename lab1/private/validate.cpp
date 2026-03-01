#include "data_generator.h"
#include "mlp.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

struct EvaluationResults {
  double mse;
  double rmse;
  double mae;
  double r2_score;
  std::vector<double> predictions;
  std::vector<double> targets;
};

EvaluationResults
evaluate_model(MLP &model, const std::vector<std::vector<double>> &inputs,
               const std::vector<std::vector<double>> &targets) {

  EvaluationResults results;
  results.predictions.reserve(inputs.size());
  results.targets.reserve(targets.size());

  double total_error = 0.0;
  double total_abs_error = 0.0;
  double sum_squared_error = 0.0;

  // 计算预测值和各种误差
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto prediction = model.predict(inputs[i]);
    double pred_value = prediction[0];
    double target_value = targets[i][0];

    results.predictions.push_back(pred_value);
    results.targets.push_back(target_value);

    double error = pred_value - target_value;
    total_error += error;
    total_abs_error += std::abs(error);
    sum_squared_error += error * error;
  }

  int n = inputs.size();

  // 计算MSE
  results.mse = sum_squared_error / n;

  // 计算RMSE
  results.rmse = std::sqrt(results.mse);

  // 计算MAE
  results.mae = total_abs_error / n;

  // 计算R²分数
  double target_mean = 0.0;
  for (double target : results.targets) {
    target_mean += target;
  }
  target_mean /= n;

  double ss_tot = 0.0;
  for (double target : results.targets) {
    double diff = target - target_mean;
    ss_tot += diff * diff;
  }

  double ss_res = sum_squared_error;
  results.r2_score = 1.0 - (ss_res / ss_tot);

  return results;
}

void print_evaluation_results(const EvaluationResults &results,
                              const std::string &dataset_name) {
  std::cout << dataset_name << "评估结果:" << std::endl;
  std::cout << "  均方误差 (MSE): " << std::fixed << std::setprecision(6)
            << results.mse << std::endl;
  std::cout << "  均方根误差 (RMSE): " << std::fixed << std::setprecision(6)
            << results.rmse << std::endl;
  std::cout << "  平均绝对误差 (MAE): " << std::fixed << std::setprecision(6)
            << results.mae << std::endl;
  std::cout << "  R² 分数: " << std::fixed << std::setprecision(6)
            << results.r2_score << std::endl;
  std::cout << std::endl;
}

void save_predictions_csv(const EvaluationResults &results,
                          const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "无法创建预测结果文件: " << filename << std::endl;
    return;
  }

  file << "target,prediction,error,abs_error" << std::endl;
  for (size_t i = 0; i < results.targets.size(); ++i) {
    double error = results.predictions[i] - results.targets[i];
    double abs_error = std::abs(error);

    file << std::fixed << std::setprecision(6) << results.targets[i] << ","
         << results.predictions[i] << "," << error << "," << abs_error
         << std::endl;
  }

  file.close();
  std::cout << "预测结果已保存到: " << filename << std::endl;
}

void print_error_statistics(const EvaluationResults &results) {
  std::vector<double> errors;
  for (size_t i = 0; i < results.predictions.size(); ++i) {
    errors.push_back(std::abs(results.predictions[i] - results.targets[i]));
  }

  std::sort(errors.begin(), errors.end());

  double min_error = errors[0];
  double max_error = errors.back();
  double median_error = errors[errors.size() / 2];
  double q1_error = errors[errors.size() / 4];
  double q3_error = errors[3 * errors.size() / 4];

  std::cout << "误差统计:" << std::endl;
  std::cout << "  最小误差: " << std::fixed << std::setprecision(6) << min_error
            << std::endl;
  std::cout << "  最大误差: " << std::fixed << std::setprecision(6) << max_error
            << std::endl;
  std::cout << "  中位数误差: " << std::fixed << std::setprecision(6)
            << median_error << std::endl;
  std::cout << "  第一四分位数: " << std::fixed << std::setprecision(6)
            << q1_error << std::endl;
  std::cout << "  第三四分位数: " << std::fixed << std::setprecision(6)
            << q3_error << std::endl;
  std::cout << std::endl;
}

int main() {
  std::cout << "=== MLP网络验证程序 ===" << std::endl;
  std::cout << "用于验证训练好的MLP模型" << std::endl;
  std::cout << "========================================" << std::endl;

  // 检查模型文件是否存在
  std::ifstream model_file("best_model.txt");
  if (!model_file.good()) {
    std::cerr << "错误: 找不到模型文件 'best_model.txt'" << std::endl;
    std::cerr << "请先运行 train.cpp 训练模型" << std::endl;
    return 1;
  }
  model_file.close();

  // 创建MLP网络并加载模型
  std::vector<int> hidden_sizes = {64, 32, 16};
  MLP model(2, hidden_sizes, 1, 0.1);

  try {
    model.load_model("best_model.txt");
  } catch (const std::exception &e) {
    std::cerr << "加载模型失败: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "模型加载成功！" << std::endl;
  model.print_architecture();
  std::cout << std::endl;

  // 创建数据生成器
  DataGenerator data_gen(-1.0, 1.0, 123); // 使用不同的种子

  // 生成测试数据
  std::vector<std::vector<double>> test_inputs, test_targets;
  const int test_size = 1000;

  // 尝试读取已存在的测试数据文件
  bool test_data_loaded = false;
  std::ifstream test_file_check("test_data.csv");
  if (test_file_check.good()) {
    std::cout << "发现已存在的测试数据文件，正在加载..." << std::endl;
    test_file_check.close();
    std::ifstream test_file("test_data.csv");
    std::string line;
    std::getline(test_file, line); // 跳过标题行

    while (std::getline(test_file, line) && test_inputs.size() < test_size) {
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

      test_inputs.push_back(input);
      test_targets.push_back(target);
    }
    test_file.close();

    if (test_inputs.size() == test_size) {
      test_data_loaded = true;
      std::cout << "测试数据加载成功！" << std::endl;
      std::cout << "测试集大小: " << test_inputs.size() << std::endl;
    } else {
      std::cout << "测试数据文件不完整，将重新生成..." << std::endl;
      test_inputs.clear();
      test_targets.clear();
    }
  }

  // 如果测试数据文件不存在或不完整，则生成新数据
  if (!test_data_loaded) {
    std::cout << "生成新的测试数据..." << std::endl;
    data_gen.generate_test_data(test_size, test_inputs, test_targets);

    // 保存测试数据到文件
    std::ofstream test_file("test_data.csv");
    if (!test_file.is_open()) {
      std::cerr << "无法创建测试数据文件: test_data.csv" << std::endl;
    } else {
      test_file << "input1,input2,target" << std::endl;
      for (size_t i = 0; i < test_inputs.size(); ++i) {
        test_file << test_inputs[i][0] << "," << test_inputs[i][1] << ","
                  << test_targets[i][0] << std::endl;
      }
      test_file.close();
      std::cout << "测试数据已保存到: test_data.csv" << std::endl;
    }
  }

  // 在测试集上评估
  std::cout << "在测试集上评估模型..." << std::endl;
  auto test_results = evaluate_model(model, test_inputs, test_targets);
  print_evaluation_results(test_results, "测试集");

  // 打印误差统计
  print_error_statistics(test_results);

  // 保存预测结果
  save_predictions_csv(test_results, "test_predictions.csv");

  // 保存评估报告
  std::ofstream report("evaluation_report.txt");
  if (report.is_open()) {
    report << "MLP网络评估报告" << std::endl;
    report << "=================" << std::endl;
    report << std::endl;

    report << "  MSE: " << std::fixed << std::setprecision(6)
           << test_results.mse << std::endl;
    report << "  RMSE: " << std::fixed << std::setprecision(6)
           << test_results.rmse << std::endl;
    report << "  MAE: " << std::fixed << std::setprecision(6)
           << test_results.mae << std::endl;
    report << "  R²: " << std::fixed << std::setprecision(6)
           << test_results.r2_score << std::endl;

    report.close();
    std::cout << "评估报告已保存到: evaluation_report.txt" << std::endl;
  }

  std::cout << "生成的相关文件:" << std::endl;
  std::cout << "  - test_predictions.csv: 测试集预测结果" << std::endl;
  std::cout << "  - evaluation_report.txt: 评估报告" << std::endl;

  return 0;
}
