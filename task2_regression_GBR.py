"""
回归预测任务 - Gradient Boosting 模型 (修复版 v3)
修复内容：
1. 解决 TypeError: Feature names must be all strings
2. 保持 One-Hot 编码和特征工程逻辑
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置全局样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def process_features(df_features):
    """
    特征工程函数
    """
    # 假设前30列是数值型量表数据 (索引 0-29)
    # 注意：此时列名是整数
    numeric_cols = list(range(30))
    
    # 1. 行均值
    df_features['row_mean'] = df_features[numeric_cols].mean(axis=1)
    # 2. 行标准差
    df_features['row_std'] = df_features[numeric_cols].std(axis=1)
    # 3. 行求和
    df_features['row_sum'] = df_features[numeric_cols].sum(axis=1)
    
    return df_features

def train_and_evaluate():
    print("=" * 60)
    print("           回归预测任务 - Gradient Boosting (修复版)")
    print("=" * 60)

    # ==========================================
    # 第一部分：数据加载
    # ==========================================
    file_path = 'data/回归预测.xlsx'

    try:
        # 读取数据
        train_df = pd.read_excel(file_path, sheet_name='训练集', header=None)
        test_df = pd.read_excel(file_path, sheet_name='测试集', header=None)
    except FileNotFoundError:
        print("[ERROR] 找不到文件，请确保 data/回归预测.xlsx 存在")
        return

    # 分离特征和目标
    X_train_raw = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    
    X_test_raw = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    print(f"\n[INFO] 原始训练集特征形状: {X_train_raw.shape}")

    # ==========================================
    # 第二部分：特征工程与编码
    # ==========================================
    # 1. 合并数据
    X_train_raw['dataset_type'] = 'train'
    X_test_raw['dataset_type'] = 'test'
    all_data = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

    # 2. 特征工程 (添加统计特征)
    all_data = process_features(all_data)

    # 3. One-Hot 编码 (针对第30列)
    cat_col_index = 30
    print("\n[INFO] 正在应用 One-Hot 编码...")
    all_data = pd.get_dummies(all_data, columns=[cat_col_index], drop_first=False)

    # ==========================================
    # 【关键修复】将所有列名转换为字符串
    # ==========================================
    all_data.columns = all_data.columns.astype(str)
    print("[INFO] 已将所有特征列名转换为字符串格式。")

    # 4. 拆分回训练集和测试集
    X_train = all_data[all_data['dataset_type'] == 'train'].drop(columns=['dataset_type'])
    X_test = all_data[all_data['dataset_type'] == 'test'].drop(columns=['dataset_type'])

    print(f"[INFO] 编码后训练集特征形状: {X_train.shape}")

    # ==========================================
    # 第三部分：标准化
    # ==========================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 第四部分：模型训练 (网格搜索)
    # ==========================================
    gb_model = GradientBoostingRegressor(random_state=42)

    # 稍微调整了参数范围，兼顾速度和效果
    param_grid = {
        'n_estimators': [100, 200, 300],    
        'learning_rate': [0.05, 0.1], 
        'max_depth': [3, 4, 5],             
        'subsample': [0.8, 1.0]             
    }

    print(f"\n[INFO] 开始训练 (网格搜索中，请稍候)...")
    grid_search = GridSearchCV(
        estimator=gb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n[RESULT] 最佳参数: {grid_search.best_params_}")

    # ==========================================
    # 第五部分：特征重要性可视化
    # ==========================================
    feature_names = X_train.columns
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    top_n = 20
    plt.title("Feature Importance (Top 20)", fontsize=16)
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), feature_names[indices[:top_n]], rotation=45, ha='right')
    plt.xlim([-1, top_n])
    plt.tight_layout()
    plt.savefig('data/17_feature_importance.png', dpi=300)
    print("[OK] 已保存: 17_feature_importance.png")

    # ==========================================
    # 第六部分：模型评估与可视化
    # ==========================================
    y_pred = best_model.predict(X_test_scaled)

    # 计算指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    sre = ((y_test - y_pred) / y_test) ** 2
    mean_sre = np.mean(sre)
    var_sre = np.var(sre)

    # 真实值 vs 预测值 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
    
    # 绘制完美预测线
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted (R2: {r2:.3f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/14_actual_vs_predicted.png', dpi=300)
    print("[OK] 已保存: 14_actual_vs_predicted.png")

    # SRE 分布直方图
    plt.figure(figsize=(8, 6))
    sns.histplot(sre, bins=30, kde=True, color='purple')
    plt.title('Distribution of Squared Relative Error (SRE)')
    plt.xlabel('SRE')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('data/16_sre_distribution.png', dpi=300)
    print("[OK] 已保存: 16_sre_distribution.png")

    # 保存详细预测结果
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'SRE': sre.values
    })
    results_df.to_csv('data/prediction_results.csv', index=False)

    # ==========================================
    # 第七部分：最终结果输出
    # ==========================================
    print("\n" + "=" * 60)
    print("           [RESULTS] 最终结果汇总")
    print("=" * 60)
    print(f"{'评估指标':<25} {'值':<15}")
    print("-" * 40)
    print(f"{'R2 Score':<25} {r2:.6f}")
    print(f"{'Mean SRE':<25} {mean_sre:.6f}")
    print(f"{'Variance SRE':<25} {var_sre:.6f}")
    print(f"{'RMSE':<25} {rmse:.6f}")

if __name__ == '__main__':
    train_and_evaluate()