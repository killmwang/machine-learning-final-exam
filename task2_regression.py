"""
回归预测任务 - Gradient Boosting 模型
数据集: 回归预测.xlsx
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


def train_and_evaluate():
    print("=" * 60)
    print("           回归预测任务 - Gradient Boosting")
    print("=" * 60)

    # ==========================================
    # 第一部分：数据加载与探索
    # ==========================================
    file_path = 'data/回归预测.xlsx'

    # 读取训练集和测试集
    train_df = pd.read_excel(file_path, sheet_name='训练集', header=None)
    test_df = pd.read_excel(file_path, sheet_name='测试集', header=None)

    print(f"\n[INFO] 数据集信息:")
    print(f"       训练集形状: {train_df.shape}")
    print(f"       测试集形状: {test_df.shape}")
    print(f"       特征数量: {train_df.shape[1] - 1}")
    print(f"       目标列: {train_df.columns[-1]}")

    # 显示列名
    print(f"\n[INFO] 训练集列名: {train_df.columns.tolist()}")

    # 检测分类变量
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    print(f"[INFO] 检测到分类变量: {cat_cols}")

    # 数据基本信息
    print("\n[INFO] 训练集前5行:")
    print(train_df.head())

    # ==========================================
    # 可视化1: 目标变量分布
    # ==========================================
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 训练集目标分布
    axes[0].hist(train_df.iloc[:, -1], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Target Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Train Set Target Distribution', fontsize=14, fontweight='bold')

    # 测试集目标分布
    axes[1].hist(test_df.iloc[:, -1], bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Target Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Test Set Target Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('data/11_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 已保存: 11_target_distribution.png")

    # ==========================================
    # 可视化2: 数值特征分布
    # ==========================================
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1:
        n_features = min(len(numerical_cols) - 1, 6)  # 最多显示6个特征
        fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols[:-1][:n_features]):
            axes[i].hist(train_df[col], bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
            axes[i].set_xlabel(col, fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].set_title(f'{col} Distribution', fontsize=11, fontweight='bold')

        # 隐藏多余的子图
        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/12_feature_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[OK] 已保存: 12_feature_distributions.png")

    # ==========================================
    # 可视化3: 数值特征相关性热力图
    # ==========================================
    numerical_train = train_df.select_dtypes(include=[np.number])
    if numerical_train.shape[1] > 1:
        fig3, ax3 = plt.subplots(figsize=(12, 10))
        corr_matrix = numerical_train.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    fmt='.2f', square=True, linewidths=0.5, ax=ax3, annot_kws={'size': 8})
        ax3.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/13_correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[OK] 已保存: 13_correlation_heatmap.png")

    # ==========================================
    # 第二部分：数据预处理
    # ==========================================
    print("\n" + "=" * 60)
    print("           数据预处理")
    print("=" * 60)

    X_train = train_df.iloc[:, :-1].copy()
    y_train = train_df.iloc[:, -1].copy()
    X_test = test_df.iloc[:, :-1].copy()
    y_test = test_df.iloc[:, -1].copy()

    # 对分类变量进行 Label Encoding
    from sklearn.preprocessing import LabelEncoder

    for col in cat_cols:
        le = LabelEncoder()
        # 合并训练集和测试集的唯一值进行fit
        all_values = pd.concat([X_train[col], X_test[col]]).unique()
        le.fit(all_values)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        print(f"[INFO] 已编码分类变量 '{col}': {len(le.classes_)} 个类别")

    # 列名转字符串
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    print(f"\n[INFO] 预处理后特征数量: {X_train.shape[1]}")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 第三部分：模型训练
    # ==========================================
    print("\n" + "=" * 60)
    print("           模型训练 - Gradient Boosting")
    print("=" * 60)

    print("\n[INFO] 开始训练 (使用 GridSearchCV 进行超参数搜索)...")
    gb_model = GradientBoostingRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4, 5]
    }

    grid_search = GridSearchCV(
        gb_model, param_grid, cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n[RESULT] 最佳参数: {grid_search.best_params_}")
    print(f"[RESULT] 最佳交叉验证分数 (neg_MSE): {grid_search.best_score_:.4f}")

    # ==========================================
    # 第四部分：预测与评估
    # ==========================================
    print("\n" + "=" * 60)
    print("           模型评估")
    print("=" * 60)

    y_pred = best_model.predict(X_test_scaled)

    # 计算各种评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 计算平方相对误差 (SRE)
    epsilon = 1e-8
    squared_relative_errors = ((y_test - y_pred) / (y_test + epsilon)) ** 2
    mean_sre = np.mean(squared_relative_errors)
    var_sre = np.var(squared_relative_errors)

    print(f"\n[RESULT] 评估指标:")
    print(f"       MSE: {mse:.6f}")
    print(f"       RMSE: {rmse:.6f}")
    print(f"       MAE: {mae:.6f}")
    print(f"       R2 Score: {r2:.6f}")
    print(f"       Mean SRE: {mean_sre:.6f}")
    print(f"       Variance SRE: {var_sre:.6f}")

    # ==========================================
    # 第五部分：可视化评估
    # ==========================================
    print("\n" + "=" * 60)
    print("           可视化评估")
    print("=" * 60)

    # 可视化4: 真实值 vs 预测值
    fig4, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 散点图
    axes[0].scatter(y_test, y_pred, color='steelblue', alpha=0.6, edgecolors='black', linewidth=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Value', fontsize=12)
    axes[0].set_ylabel('Predicted Value', fontsize=12)
    axes[0].set_title(f'Actual vs Predicted (R2 = {r2:.4f})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 残差图
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, color='coral', alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Value', fontsize=12)
    axes[1].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/14_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 已保存: 14_actual_vs_predicted.png")

    # 可视化5: 残差分布
    fig5, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 残差直方图
    axes[0].hist(residuals, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residual', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Residual Distribution (Mean: {residuals.mean():.4f}, Std: {residuals.std():.4f})',
                      fontsize=14, fontweight='bold')

    # 残差箱线图
    axes[1].boxplot(residuals, vert=True)
    axes[1].set_ylabel('Residual', fontsize=12)
    axes[1].set_title('Residual Boxplot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/15_residual_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 已保存: 15_residual_distribution.png")

    # 可视化6: 平方相对误差分布
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    sns.histplot(squared_relative_errors, bins=30, kde=True, color='purple', ax=ax6)
    ax6.set_xlabel('Squared Relative Error (SRE)', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title(f'Squared Relative Error Distribution (Mean SRE: {mean_sre:.6f})',
                  fontsize=14, fontweight='bold')
    ax6.axvline(x=mean_sre, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sre:.6f}')
    ax6.legend()
    plt.tight_layout()
    plt.savefig('data/16_sre_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 已保存: 16_sre_distribution.png")

    # 可视化7: 特征重要性
    fig7, ax7 = plt.subplots(figsize=(12, 8))
    feature_names = [str(c) for c in X_train.columns]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True).tail(20)  # 显示前20个重要特征

    ax7.barh(feature_importance['feature'], feature_importance['importance'],
             color='steelblue', edgecolor='black')
    ax7.set_xlabel('Importance', fontsize=12)
    ax7.set_ylabel('Feature', fontsize=12)
    ax7.set_title('Feature Importance (Top 20)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/17_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 已保存: 17_feature_importance.png")

    # ==========================================
    # 第六部分：预测结果对比表
    # ==========================================
    print("\n" + "=" * 60)
    print("           预测结果展示 (前10个样本)")
    print("=" * 60)

    results_df = pd.DataFrame({
        '真实值': y_test.values[:10],
        '预测值': y_pred[:10],
        '绝对误差': np.abs(y_test.values[:10] - y_pred[:10]),
        '相对误差 (%)': np.abs((y_test.values[:10] - y_pred[:10]) / y_test.values[:10] * 100)
    })
    print(results_df.to_string(index=False))

    # 保存完整预测结果
    results_full = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Absolute Error': np.abs(y_test.values - y_pred),
        'Relative Error (%)': np.abs((y_test.values - y_pred) / y_test.values * 100)
    })
    results_full.to_csv('data/prediction_results.csv', index=False, encoding='utf-8-sig')
    print("\n[OK] 已保存完整预测结果: prediction_results.csv")

    # ==========================================
    # 第七部分：最终汇总
    # ==========================================
    print("\n" + "=" * 60)
    print("           [RESULTS] 最终结果汇总")
    print("=" * 60)
    print(f"\n{'评估指标':<25} {'值':<15}")
    print("-" * 40)
    print(f"{'R2 Score':<25} {r2:.6f}")
    print(f"{'MSE':<25} {mse:.6f}")
    print(f"{'RMSE':<25} {rmse:.6f}")
    print(f"{'MAE':<25} {mae:.6f}")
    print(f"{'Mean SRE':<25} {mean_sre:.6f}")
    print(f"{'Variance SRE':<25} {var_sre:.6f}")

    print(f"\n[INFO] 最佳模型参数:")
    for param, value in grid_search.best_params_.items():
        print(f"       {param}: {value}")

    print("\n[INFO] 所有可视化图表已保存至 data/ 目录:")
    print("   - 11_target_distribution.png")
    print("   - 12_feature_distributions.png")
    print("   - 13_correlation_heatmap.png")
    print("   - 14_actual_vs_predicted.png")
    print("   - 15_residual_distribution.png")
    print("   - 16_sre_distribution.png")
    print("   - 17_feature_importance.png")
    print("   - prediction_results.csv")

    print("\n" + "=" * 60)
    print("                    任务完成!")
    print("=" * 60)

    return best_model, y_pred, y_test


if __name__ == "__main__":
    train_and_evaluate()
