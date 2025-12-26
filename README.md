# machine-learning-exam

机器学习与神经网络导论 - 课程期末考核代码库

## 目录

- [项目简介](#项目简介)
- [环境准备](#环境准备)
- [目录结构](#目录结构)
- [任务说明](#任务说明)
  - [任务一：分类任务](#任务一分类任务)
  - [任务二：回归任务](#任务二回归任务)
- [运行指南](#运行指南)
- [可视化输出](#可视化输出)
- [模型性能](#模型性能)

## 项目简介

本项目包含机器学习与神经网络课程期末考核的两个核心任务：
- **任务一**：乳腺癌分类（分类任务）- 对比 Neural Network 与 SVM 两种模型
- **任务二**：回归预测任务 - 使用 Random Forest 进行预测

## 环境准备

本项目基于 Python 3.9+ 开发。

### 安装依赖

请确保终端位于项目根目录下，运行以下命令一键安装所需库：

```bash
pip install -r requirements.txt
```

### 主要依赖

| 包名 | 版本要求 | 说明 |
|------|----------|------|
| numpy | - | 数值计算 |
| pandas | - | 数据处理 |
| matplotlib | - | 数据可视化 |
| seaborn | - | 统计图表 |
| scikit-learn | - | 机器学习工具 |
| torch | - | 深度学习框架 |
| openpyxl | - | Excel文件读写 |

## 目录结构

```
Project_Root/
├── data/
│   ├── breast_cancer.csv           # Task 1: Breast Cancer Dataset
│   ├── 回归预测.xlsx                # Task 2: Regression Dataset
│   ├── best_nn_model.pth           # Trained Neural Network Model
│   ├── prediction_results.csv      # Regression Prediction Results
│   │
│   ├── Task 1 图表 (01~07):
│   │   ├── 01_class_distribution.png     # 类别分布
│   │   ├── 02_feature_correlation.png    # 特征相关性
│   │   ├── 03_training_loss_curve.png    # 训练损失曲线
│   │   ├── 04_model_comparison.png       # 模型对比
│   │   ├── 05_roc_curves.png             # ROC曲线
│   │   ├── 06_confusion_matrices.png     # 混淆矩阵
│   │   └── 07_prediction_distribution.png # 预测分布
│   │
│   └── Task 2 图表 (11~17):
│       ├── 11_target_distribution.png    # 目标分布
│       ├── 12_feature_distributions.png  # 特征分布
│       ├── 13_correlation_heatmap.png    # 相关性热力图
│       ├── 14_actual_vs_predicted.png    # 真实值vs预测值
│       ├── 15_residual_distribution.png  # 残差分布
│       ├── 16_sre_distribution.png       # SRE分布
│       └── 17_feature_importance.png     # 特征重要性
│
├── task1_classification.py         # Task 1: Classification (NN vs SVM)
├── task2_regression.py             # Task 2: Regression (Gradient Boosting)
├── task2_regression_RF.py          # Task 2: Regression (Random Forest)
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

## 任务说明

### 任务一：分类任务

**数据集**：Wisconsin Breast Cancer Dataset

**任务目标**：根据乳腺癌细胞的特征数据，预测肿瘤是恶性(Malignant)还是良性(Benign)

**模型对比**：
- **Neural Network** (PyTorch实现)
  - 3层隐藏层 (64 -> 32 -> 16)
  - BatchNorm + Dropout 正则化
  - Adam优化器 + 学习率调度
- **SVM** (sklearn实现)
  - RBF核函数
  - 支持概率输出

**评估指标**：
- 准确率 (Accuracy)
- AUC (Area Under Curve)
- 混淆矩阵
- 分类报告 (Precision, Recall, F1-score)

### 任务二：回归任务

**数据集**：回归预测.xlsx（包含训练集和测试集两个Sheet）

**任务目标**：根据给定的特征数据，预测目标连续值

**模型**：Random Forest Regressor

**超参数搜索**：
- n_estimators: [100, 200, 300]
- max_depth: [3, 5, 7]
- min_samples_split: [5, 10]
- max_features: ['sqrt', 'log2']

**评估指标**：
- R² Score
- MSE (均方误差)
- RMSE (均方根误差)
- MAE (平均绝对误差)
- Mean SRE (平方相对误差均值)
- Variance SRE (平方相对误差方差)

## 运行指南

### 运行任务一（分类）

```bash
python task1_classification.py
```

### 运行任务二（回归）

```bash
# Gradient Boosting 版本
python task2_regression.py

# Random Forest 版本
python task2_regression_RF.py
```

## 可视化输出

### 任务一可视化 (01~07)

| 图表 | 说明 |
|------|------|
| 01_class_distribution.png | 类别分布饼图 |
| 02_feature_correlation.png | 特征相关性热力图 |
| 03_training_loss_curve.png | 神经网络训练损失曲线 |
| 04_model_comparison.png | 模型准确率与AUC对比 |
| 05_roc_curves.png | ROC曲线对比 |
| 06_confusion_matrices.png | 混淆矩阵（两个模型） |
| 07_prediction_distribution.png | 预测概率分布 |

### 任务二可视化 (11~17)

| 图表 | 说明 |
|------|------|
| 11_target_distribution.png | 训练集/测试集目标分布 |
| 12_feature_distributions.png | 特征分布 |
| 13_correlation_heatmap.png | 特征相关性热力图 |
| 14_actual_vs_predicted.png | 真实值vs预测值 + 残差图 |
| 15_residual_distribution.png | 残差直方图 + 箱线图 |
| 16_sre_distribution.png | 平方相对误差分布 |
| 17_feature_importance.png | 特征重要性排名 |

## 模型性能

### 任务一预期性能

| 模型 | 准确率 | AUC |
|------|--------|-----|
| Neural Network | >95% | >0.98 |
| SVM | >95% | >0.98 |

### 任务二预期性能

| 指标 | 预期范围 |
|------|----------|
| R² Score | >0.9 |
| RMSE | 较低 |
| Mean SRE | <0.01 |

---

