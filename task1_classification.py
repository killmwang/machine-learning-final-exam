"""
乳腺癌分类任务 - Neural Network vs SVM 对比
数据集: Wisconsin Breast Cancer Dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# 设置全局样式
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 第一部分：数据加载与预处理
# ==========================================
print("=" * 60)
print("           乳腺癌分类任务 - Neural Network vs SVM")
print("=" * 60)

# 1. 读取数据
df = pd.read_csv('data/breast_cancer.csv')
print(f"\n[INFO] 数据集形状: {df.shape}")
print(f"       特征数量: {df.shape[1] - 1}")
print(f"       样本数量: {df.shape[0]}")

# 数据基本信息
print("\n[INFO] 数据集信息:")
print(df.info())

# 可视化1: 类别分布饼图
fig1, ax1 = plt.subplots(figsize=(8, 6))
labels = ['Malignant', 'Benign']
sizes = [df[df['target'] == 0].shape[0], df[df['target'] == 1].shape[0]]
colors = ['#ff6b6b', '#4ecdc4']
explode = (0.05, 0)
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12})
ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 01_class_distribution.png")

# 可视化2: 特征相关性热力图
fig2, ax2 = plt.subplots(figsize=(14, 10))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax2)
ax2.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/02_feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 02_feature_correlation.png")

# 准备特征和标签
X = df.drop('target', axis=1).values
y = df['target'].values

# 2. 划分训练集和测试集 (80% / 20%)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[INFO] 数据集划分:")
print(f"       训练集样本数: {len(X_train_raw)}")
print(f"       测试集样本数: {len(X_test_raw)}")

# 3. 标准化
scaler = StandardScaler()
X_train_numpy = scaler.fit_transform(X_train_raw)
X_test_numpy = scaler.transform(X_test_raw)

# 4. 转换为 Tensor
X_train_tensor = torch.tensor(X_train_numpy, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_numpy, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# ==========================================
# 第二部分：PyTorch 神经网络训练
# ==========================================
print("\n" + "=" * 60)
print("           开始训练神经网络 (Neural Network)")
print("=" * 60)

class BreastCancerClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BreastCancerClassifier, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layer_stack(x)

# 初始化模型
model = BreastCancerClassifier(X_train_numpy.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# 训练循环
epochs = 100
train_losses = []
best_loss = float('inf')

print(f"\n[INFO] 开始训练 (最大轮数: {epochs})")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'data/best_nn_model.pth')

    if (epoch + 1) % 20 == 0:
        print(f"   Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# 可视化3: 训练损失曲线
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(range(1, epochs + 1), train_losses, 'b-', linewidth=2, label='Training Loss')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/03_training_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 03_training_loss_curve.png")

# 评估神经网络
model.load_state_dict(torch.load('data/best_nn_model.pth'))
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted_probs = torch.sigmoid(test_outputs).numpy().flatten()
    nn_predicted = (predicted_probs > 0.5).astype(int)

nn_acc = accuracy_score(y_test, nn_predicted)
nn_auc = roc_auc_score(y_test, predicted_probs)

print(f"\n[RESULT] Neural Network 准确率: {nn_acc * 100:.2f}%")
print(f"[RESULT] Neural Network AUC: {nn_auc:.4f}")


# ==========================================
# 第三部分：SVM 训练
# ==========================================
print("\n" + "=" * 60)
print("           开始训练 SVM (支持向量机)")
print("=" * 60)

# 初始化并训练
svm_model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
svm_model.fit(X_train_numpy, y_train)

# 预测与评估
svm_predicted = svm_model.predict(X_test_numpy)
svm_probs = svm_model.predict_proba(X_test_numpy)[:, 1]
svm_acc = accuracy_score(y_test, svm_predicted)
svm_auc = roc_auc_score(y_test, svm_probs)

print(f"\n[RESULT] SVM 准确率: {svm_acc * 100:.2f}%")
print(f"[RESULT] SVM AUC: {svm_auc:.4f}")


# ==========================================
# 第四部分：模型对比可视化
# ==========================================
print("\n" + "=" * 60)
print("           模型性能对比")
print("=" * 60)

# 可视化4: 模型准确率和AUC对比柱状图
fig4, axes = plt.subplots(1, 2, figsize=(14, 5))

# 准确率对比
models = ['Neural Network', 'SVM']
accuracies = [nn_acc * 100, svm_acc * 100]
colors = ['#3498db', '#2ecc71']
bars = axes[0].bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 105)
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# AUC对比
aucs = [nn_auc, svm_auc]
bars2 = axes[1].bar(models, aucs, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('AUC Score', fontsize=12)
axes[1].set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 1.1)
for bar, auc_val in zip(bars2, aucs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{auc_val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('data/04_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 04_model_comparison.png")

# 可视化5: ROC曲线对比
fig5, ax5 = plt.subplots(figsize=(8, 8))
fpr_nn, tpr_nn, _ = roc_curve(y_test, predicted_probs)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)

ax5.plot(fpr_nn, tpr_nn, 'b-', linewidth=2, label=f'Neural Network (AUC = {nn_auc:.4f})')
ax5.plot(fpr_svm, tpr_svm, 'g-', linewidth=2, label=f'SVM (AUC = {svm_auc:.4f})')
ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax5.set_xlabel('False Positive Rate', fontsize=12)
ax5.set_ylabel('True Positive Rate', fontsize=12)
ax5.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax5.legend(loc='lower right', fontsize=11)
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/05_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 05_roc_curves.png")

# 可视化6: 混淆矩阵
fig6, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_nn = confusion_matrix(y_test, nn_predicted)
cm_svm = confusion_matrix(y_test, svm_predicted)

disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=['Malignant', 'Benign'])
disp_nn.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title(f'Neural Network Confusion Matrix\nAccuracy: {nn_acc*100:.2f}%', fontsize=12, fontweight='bold')

disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Malignant', 'Benign'])
disp_svm.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title(f'SVM Confusion Matrix\nAccuracy: {svm_acc*100:.2f}%', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('data/06_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 06_confusion_matrices.png")

# 可视化7: 预测概率分布
fig7, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(predicted_probs[y_test == 0], bins=20, alpha=0.7, color='red', label='Malignant')
axes[0].hist(predicted_probs[y_test == 1], bins=20, alpha=0.7, color='green', label='Benign')
axes[0].set_xlabel('Predicted Probability', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Neural Network Prediction Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].axvline(x=0.5, color='black', linestyle='--')

axes[1].hist(svm_probs[y_test == 0], bins=20, alpha=0.7, color='red', label='Malignant')
axes[1].hist(svm_probs[y_test == 1], bins=20, alpha=0.7, color='green', label='Benign')
axes[1].set_xlabel('Predicted Probability', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('SVM Prediction Distribution', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].axvline(x=0.5, color='black', linestyle='--')

plt.tight_layout()
plt.savefig('data/07_prediction_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 07_prediction_distribution.png")

# ==========================================
# 第五部分：最终结果汇总
# ==========================================
print("\n" + "=" * 60)
print("           [RESULTS] 最终结果对比")
print("=" * 60)
print(f"\n{'模型':<20} {'准确率':<15} {'AUC':<15}")
print("-" * 50)
print(f"{'Neural Network':<20} {nn_acc*100:.2f}%{'':<8} {nn_auc:.4f}")
print(f"{'SVM':<20} {svm_acc*100:.2f}%{'':<8} {svm_auc:.4f}")

# 打印详细分类报告
print("\n[INFO] Neural Network 分类报告:")
print(classification_report(y_test, nn_predicted, target_names=['Malignant', 'Benign']))

print("[INFO] SVM 分类报告:")
print(classification_report(y_test, svm_predicted, target_names=['Malignant', 'Benign']))

# 结论
print("\n" + "=" * 60)
print("           [CONCLUSION] 结论")
print("=" * 60)
if svm_acc > nn_acc:
    print("=> SVM 在该数据集上表现更好")
elif nn_acc > svm_acc:
    print("=> Neural Network 在该数据集上表现更好")
else:
    print("=> 两个模型表现一致")

print("\n[INFO] 所有可视化图表已保存至 data/ 目录:")
print("   - 01_class_distribution.png")
print("   - 02_feature_correlation.png")
print("   - 03_training_loss_curve.png")
print("   - 04_model_comparison.png")
print("   - 05_roc_curves.png")
print("   - 06_confusion_matrices.png")
print("   - 07_prediction_distribution.png")

print("\n" + "=" * 60)
print("                    任务完成!")
print("=" * 60)
