import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_val_score,
    KFold,
    StratifiedKFold
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss,
    classification_report,
    explained_variance_score
)
import shap
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    ElasticNet,
    Ridge,
    BayesianRidge
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    StackingRegressor
)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
import optuna
import re
from scipy.stats import pearsonr


Trainning_data = pd.read_csv(r'/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/train_date/training_date_LC-ωPBE.csv')
Trainning_data.drop_duplicates(inplace=True)
Trainning_data.dropna(inplace=True)

def convert_numbers_in_string(s):
    # 使用正则表达式找到字符串中的所有数字
    numbers = re.findall(r'\d+\.\d+|\d+', s)
    for number in numbers:
        # 将找到的数字替换为浮点数
        s = s.replace(number, str(float(number)))
    return s

# 遍历DataFrame中的每一列
for column in Trainning_data.columns:
    # 将当前列转换为列表，以便修改
    column_data = Trainning_data[column].tolist()
    # 遍历当前列中的每个值
    for i, value in enumerate(column_data):
        # 检查值是否为字符串
        if isinstance(value, str):
            # 尝试将字符串中的数字转换为浮点数
            column_data[i] = convert_numbers_in_string(value)
    # 将修改后的列数据更新回DataFrame中
    Trainning_data[column] = column_data

# 输出修改后的DataFrame
print(Trainning_data)

for col in Trainning_data.columns:
    if Trainning_data[col].dtype == 'object':  # 如果列是字符串类型
        try:
            Trainning_data[col] = pd.to_numeric(Trainning_data[col], errors='coerce')
        except ValueError:
            print(f"无法将列 '{col}' 转换为数值类型。")

# 删除包含缺失值的行（如果在转换过程中产生了NaN）
Trainning_data.dropna(inplace=True)

# 定义检测异常值的函数
def detect_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return outliers

# 检查每个特征的异常值数量
outliers_count = {}
for col in Trainning_data.columns:
    if pd.api.types.is_numeric_dtype(Trainning_data[col]):  # 确保列是数值类型
        outliers = detect_outliers(Trainning_data[col])
        outliers_count[col] = len(outliers)
    else:
        print(f"列 '{col}' 不是数值类型，跳过异常值检测。")



# 删除 'best_omega' 的异常值
if 'best_omega' in Trainning_data.columns:
    outliers = detect_outliers(Trainning_data['best_omega'])
    Trainning_data = Trainning_data[~Trainning_data['best_omega'].isin(outliers)]

# 数据集划分（用于最终模型训练）
X = Trainning_data.drop(columns=['best_omega'], errors='ignore')
y = Trainning_data['best_omega']

# 数据标准化（仅对数值特征）
numeric_features = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()

# 重新划分数据集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

def interpolate_samples(X, y, n_samples=300, k=20):
    """
    通过插值生成新样本
    :param X: 特征矩阵 (n_samples, n_features)
    :param y: 目标变量 (n_samples,)
    :param n_samples: 需生成的样本数
    :param k: 近邻数
    :return: 插值后的 X_new, y_new
    """
    # 确保 X 是 Pandas DataFrame
    
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    synthetic_X = []
    synthetic_y = []
    
    for _ in range(n_samples):
        # 随机选择一个样本
        idx = np.random.randint(8, len(X))
        X_center = X.iloc[idx]
        y_center = y.iloc[idx]
        
        # 找到 k 个最近邻
        _, indices = nbrs.kneighbors([X_center])
        nn_idx = np.random.choice(indices[0])
        X_nn = X.iloc[nn_idx]
        y_nn = y.iloc[nn_idx]
        
        # 生成插值样本
        alpha = np.random.random()  # 插值系数
        X_new = X_center + alpha * (X_nn - X_center)
        y_new = y_center + alpha * (y_nn - y_center)
        
        synthetic_X.append(X_new)
        synthetic_y.append(y_new)
    
    X_new = pd.DataFrame(synthetic_X, columns=X.columns)
    y_new = pd.Series(synthetic_y, name=y.name)
    return X_new, y_new

# 生成插值样本
x_train_interpolated, y_train_interpolated = interpolate_samples(x_train, y_train, n_samples=16080-9000, k=2800)

# 合并原始训练数据和插值数据
x_train_final = pd.concat([x_train, x_train_interpolated], axis=0).reset_index(drop=True)
y_train_final = pd.concat([y_train, y_train_interpolated], axis=0).reset_index(drop=True)


scaler = joblib.load('/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-wPBE/Stacking_model/scaler_LC.pkl')
x_train[numeric_features] = scaler.transform(x_train[numeric_features])
x_test[numeric_features] = scaler.transform(x_test[numeric_features])

xgb_regressor = joblib.load('/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-wPBE/Stacking_model/final_xgb_model_LC.pkl')
lgbm_regressor = joblib.load('/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-wPBE/Stacking_model/final_lgb_model_LC.pkl')
cat_regressor = joblib.load('/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-wPBE/Stacking_model/final_catboost_model_LC.pkl')
gbr_regressor = joblib.load('/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-wPBE/Stacking_model/final_gbr_model_LC.pkl')
rf_regressor = joblib.load('/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-wPBE/Stacking_model/final_rf_model_LC.pkl')
ada_regressor = joblib.load('/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-wPBE/Stacking_model/final_adaboost_model_LC.pkl')
lasso_regressor = joblib.load('/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-wPBE/Stacking_model/final_lasso_model_LC.pkl')
ridge_regressor = joblib.load('/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-wPBE/Stacking_model/final_ridge_model_LC.pkl')
elastic_regressor = joblib.load('/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-wPBE/Stacking_model/final_elasticnet_model_LC.pkl')

base_learners = [
    ("XGBoost", xgb_regressor),
    ("LightGBM", lgbm_regressor),
    ("GBDT", gbr_regressor),
    ("RandomForest", rf_regressor),
    ("Lasso", lasso_regressor),
    ("CatBoost", cat_regressor),
    ("Ridge", ridge_regressor),
    ("ElasticNet", elastic_regressor),
    ("AdaBoost",ada_regressor)
]

def objective(trial):
    # 定义超参数搜索空间
    X_sampled, y_sampled = x_train[:800], y_train[:800]
    
    # BayesianRidge 的超参数
    alpha_1 = trial.suggest_float('alpha_1', 1e-99, 1e-5, log=True)
    lambda_1 = trial.suggest_float('lambda_1', 1e-99, 1e-5, log=True)
    alpha_2 = trial.suggest_float('alpha_2', 1e-99, 1e-5, log=True)
    lambda_2 = trial.suggest_float('lambda_2', 1e-99, 1e-5, log=True)
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    
    # 更新元模型的超参数
    meta_model = BayesianRidge(
        alpha_1=alpha_1,
        lambda_1=lambda_1,
        alpha_2=alpha_2,
        lambda_2=lambda_2,
        fit_intercept=fit_intercept,

    )
    
    # 创建堆叠回归器
    stacking_regressor = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_model,
        cv=trial.suggest_int('cv', 3, 10),  # 调整交叉验证的折数
        n_jobs=trial.suggest_categorical('n_jobs', [None, -1])  # 调整并行计算的参数
    )
    
    # 使用交叉验证评估堆叠回归器的性能
    kfold = KFold(
        n_splits=trial.suggest_int('n_splits', 2, 8),  # 调整 KFold 的折数
        shuffle=True,
        random_state=42
    )
    score = cross_val_score(
        stacking_regressor,
        x_train,
        y_train,
        cv=kfold,
        scoring='neg_mean_squared_error'
    ).mean()
    
    # 返回负的均方误差（因为 Optuna 默认是最小化目标函数）
    return -score

# 打印基学习器信息
base_learners_info = [
    ("XGB", xgb_regressor, "regressor"),
    ("LGB", lgbm_regressor, "regressor"),
    ("GBM", gbr_regressor, "regressor"),
    ("RF", rf_regressor, "regressor"),
    ("LASSO", lasso_regressor, "regressor"),
    ("CatBoost", cat_regressor, "regressor"),
    ("Ridge", ridge_regressor, "regressor"),
    ("ElasticNet", elastic_regressor, "regressor")
]

for name, estimator, estimator_type in base_learners_info:
    print(f"{name} estimator type: {estimator_type}")

# 创建 Optuna 研究
study = optuna.create_study(direction='minimize', 
                            pruner=optuna.pruners.MedianPruner(), 
                         )
study.optimize(objective, n_trials=20, show_progress_bar=True, n_jobs=-1)

# 输出最佳超参数
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print(f'  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# 获取最佳超参数
best_params = study.best_params

# 使用最佳超参数更新元模型
best_meta_model = BayesianRidge(
    alpha_1=best_params['alpha_1'],
    lambda_1=best_params['lambda_1'],
    alpha_2=best_params['alpha_2'],
    lambda_2=best_params['lambda_2'],
    fit_intercept=best_params['fit_intercept'],
)

# 创建最终的堆叠回归器
final_stacking_regressor = StackingRegressor(
    estimators=base_learners,
    final_estimator=best_meta_model,
    cv=best_params['cv'],
    n_jobs=best_params['n_jobs']
)

# 训练最终的堆叠回归器
final_stacking_regressor.fit(x_train, y_train)
joblib.dump(final_stacking_regressor, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-wPBE/Stacking_model/final_stacking_model_LC.pkl')
results = []

# 定义函数用于计算指标
def evaluate_model(model_name, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) if np.all(y_test != 0) else np.nan
    ev = explained_variance_score(y_test, y_pred)
    R2,p=pearsonr(y_test, y_pred)
    results.append({
        "R²": r2,

        "MSE/10^-4": mse*10000,
        "MAE/10^-2": mae*100,
        "MAPE/10^-1": mape*10,
        "EV": ev,
        "R-Pearso": R2,
        "Model": model_name
    })

# 评估最终的堆叠回归器
y_pred = final_stacking_regressor.predict(x_test)
evaluate_model("Stacking Regressor", y_test, y_pred)

# 评估基学习器
for name, model in base_learners:
    y_pred = model.predict(x_test)
    evaluate_model(name, y_test, y_pred)

# 输出结果
results_df = pd.DataFrame(results)

# 打印表格
print(results_df)