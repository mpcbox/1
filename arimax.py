import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取数据
file_path = 'hour(1).csv'
data = pd.read_csv(file_path)

# 定义特征和目标变量
X_full = data[['temp', 'hum', 'windspeed', 'atemp']]
y_full = data['cnt']

# 保留20%数据用于测试
train_size = int(len(X_full) * 0.8)
X_train_full, X_test_full = X_full[:train_size], X_full[train_size:]
y_train_full, y_test_full = y_full[:train_size], y_full[train_size:]

# 在训练集上拟合ARIMAX模型
exog_train_full = sm.add_constant(X_train_full)
model_full = ARIMA(endog=y_train_full, exog=exog_train_full, order=(1, 1, 1))
model_fit_full = model_full.fit()

# 打印模型摘要
model_summary_full = model_fit_full.summary()
print(model_summary_full)

# 分析特征对出行量的影响
coefficients = model_fit_full.params
for feature in ['temp', 'hum', 'windspeed', 'atemp']:
    coef = coefficients[feature]
    direction = "正向" if coef > 0 else "负向"
    print(f"{feature} 对出行量的影响系数为 {coef:.4f}，影响为{direction}。")

# 对测试集进行预测
exog_test_full = sm.add_constant(X_test_full)
y_pred_full = model_fit_full.predict(start=len(y_train_full), end=len(y_full) - 1, exog=exog_test_full)

# 创建包含测试集特征数据和预测结果的表格
results_with_features_df = X_test_full.copy()
results_with_features_df['Predicted_cnt'] = y_pred_full.values

# 输出预测结果表格
print(results_with_features_df.head())

# 保存预测结果到本地文件
output_file_path = 'prediction_results.csv'  # 输出文件路径
results_with_features_df.to_csv(output_file_path, index=False)
print(f"预测结果已保存到 {output_file_path}")


