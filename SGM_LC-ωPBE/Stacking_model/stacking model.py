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

logging.getLogger('lightgbm').setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


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

print(f"原始训练数据量: {len(x_train)}, 插值后训练数据量: {len(x_train_final)}")

# 更新变量
x_train = x_train_final
y_train = y_train_final

# 标准化数据

scaler = StandardScaler()
x_train[numeric_features] = scaler.fit_transform(x_train[numeric_features])
# 假设 x_test 也已经定义
x_test[numeric_features] = scaler.transform(x_test[numeric_features])
joblib.dump(scaler, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/Stacking_model/scaler_LC.pkl')

def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-9, 0.3, log=True),  # Adjusted range and log scale
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),  # Adjusted range
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),  # Adjusted range
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-30, 1.0, log=True),  # Adjusted range and log scale
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-30, 100.0, log=True),  # Adjusted range and log scale
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-30, 100.0, log=True),  # Adjusted range and log scale
        'n_jobs': -1  # Use all available cores
    }

    # Create a pipeline with StandardScaler and XGBRegressor
    model = make_pipeline(
        StandardScaler(),
        XGBRegressor(**params)
    )
    
    # Use 5-fold cross-validation to calculate MAE
    scores = cross_val_score(model, x_train, y_train, 
                            cv=5, 
                            scoring='neg_mean_squared_error',
                            n_jobs=-1)  # Parallel processing
    mse = -scores.mean()  # Convert to positive MAE
    return mse

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective_xgb, n_trials=200, show_progress_bar=True)

# Get the best parameters
best_params_xgb = study.best_params

# Train the final model with the best parameters
final_model = make_pipeline(
    StandardScaler(),
    XGBRegressor(**best_params_xgb)
)
final_model.fit(x_train, y_train)

# Evaluate on the test set
y_pred = final_model.predict(x_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE with Best Model: {test_mse}")
joblib.dump(final_model, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/Stacking_model/final_xgb_model_LC.pkl')




def objective_lgb(trial):
    params = {
        'verbose': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'n_jobs': -1
    }

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for train_idx, valid_idx in kf.split(x_train):
        # 数据标准化（每个fold独立进行）
        scaler = StandardScaler()
        
        # 划分原始数据
        X_train_fold = x_train.iloc[train_idx]
        X_valid_fold = x_train.iloc[valid_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        # 标准化处理（仅用训练fold数据拟合）
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_valid_scaled = scaler.transform(X_valid_fold)

        # 模型训练
        model = LGBMRegressor(**params)
        model.fit(X_train_scaled, y_train_fold
                  )        
        # 验证预测
        y_pred = model.predict(X_valid_scaled)
        mse_scores.append(mean_squared_error(y_valid_fold, y_pred))

    return np.mean(mse_scores)

# 优化过程
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=30, show_progress_bar=True)

# Get the best parameters
best_params_lgb = study_lgb.best_params
final_model = make_pipeline(
    StandardScaler(),
    LGBMRegressor(**best_params_lgb)
)
final_model.fit(x_train, y_train)

# 评估测试集
y_pred = final_model.predict(x_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE with Best Model: {test_mse}")
joblib.dump(final_model, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/Stacking_model/final_lgb_model_LC.pkl')



def objective_gbr(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # 树的数量
        'max_depth': trial.suggest_int('max_depth', 1, 12),  # 每棵树的最大深度
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),  # 分裂内部节点所需的最小样本数
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),  # 叶节点所需的最小样本数
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),  # 学习率
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),  # 子采样率
        'max_features': trial.suggest_int('max_features', 1, 50),  # 最大特征数
        'alpha': trial.suggest_uniform('alpha', 1e-8, 0.9),  # Huber损失或Quantile损失的正则化参数
    }
    
    # 创建包含标准化的Pipeline
    model = make_pipeline(
        StandardScaler(),
        GradientBoostingRegressor(**params)
    )
    
    # 使用5折交叉验证计算MSE
    scores = cross_val_score(model, x_train, y_train, 
                            cv=4, 
                            scoring='neg_mean_squared_error',
                            n_jobs=-1)  # 并行加速
    mse = -scores.mean()
    return mse

# Optuna优化
study = optuna.create_study(direction='minimize')
study.optimize(objective_gbr, n_trials=20, show_progress_bar=True, n_jobs=-1)

# 输出最佳参数
best_params_gbr = study.best_params
print(f"Best Params: {best_params_gbr}")

# 使用最佳参数训练最终模型
final_model = make_pipeline(
    StandardScaler(),
    GradientBoostingRegressor(**best_params_gbr, random_state=42),
)
final_model.fit(x_train, y_train)                       

# 评估测试集
y_pred = final_model.predict(x_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE with Best Model: {test_mse}")
joblib.dump(final_model, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/Stacking_model/final_gbr_model_LC.pkl')

# 定义目标函数
def objective_lasso(trial):
    # 定义超参数搜索空间
    params = {
        'alpha': trial.suggest_loguniform('alpha', 1e-6, 1.0),  # Lasso 的正则化强度
        'max_iter': trial.suggest_int('max_iter', 100, 1000),  # 最大迭代次数
        'random_state': trial.suggest_int('random_state', 1, 100),  # 随机种子
    }

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for train_idx, valid_idx in kf.split(x_train):
        # 数据标准化（每个fold独立进行）
        scaler = StandardScaler()

        # 划分原始数据
        X_train_fold = x_train.iloc[train_idx]
        X_valid_fold = x_train.iloc[valid_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        # 标准化处理（仅用训练fold数据拟合）
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_valid_scaled = scaler.transform(X_valid_fold)

        # 模型训练
        model = Lasso(**params)
        model.fit(X_train_scaled, y_train_fold)

        # 验证预测
        y_pred = model.predict(X_valid_scaled)
        mse_scores.append(mean_squared_error(y_valid_fold, y_pred))  # 使用MSE作为评估指标

    return np.mean(mse_scores)

# 优化过程
study_lasso = optuna.create_study(direction='minimize')
study_lasso.optimize(objective_lasso, n_trials=40, show_progress_bar=True)

# 获取最佳参数
best_params_lasso = study_lasso.best_params

# 训练最终模型
final_model = make_pipeline(
    StandardScaler(),
    Lasso(**best_params_lasso)
)
final_model.fit(x_train, y_train)

# 评估测试集
y_pred = final_model.predict(x_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE with Best Lasso Model: {test_mse}")
joblib.dump(final_model, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/Stacking_model/final_lasso_model_LC.pkl')

# 定义目标函数
def objective_catboost(trial):
    params = {
        'verbose': False,  # 关闭训练日志
        'iterations': trial.suggest_int('iterations', 10, 500),  # 树的数量
        'depth': trial.suggest_int('depth', 1, 12),  # 树的深度
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 0.1, log=True),  # 学习率
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-20, 100.0, log=True),  # L2 正则化
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),  # 随机强度
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),  # 样本采样温度
        'border_count': trial.suggest_int('border_count', 1, 500),  # 特征分箱数
        'random_state': trial.suggest_int('random_state', 1, 100),  # 随机种子
    }

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for train_idx, valid_idx in kf.split(x_train):
        # 数据标准化（每个fold独立进行）
        scaler = StandardScaler()
        
        # 划分原始数据
        X_train_fold = x_train.iloc[train_idx]
        X_valid_fold = x_train.iloc[valid_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        # 标准化处理（仅用训练fold数据拟合）
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_valid_scaled = scaler.transform(X_valid_fold)

        # 模型训练
        model = CatBoostRegressor(**params)
        model.fit(X_train_scaled, y_train_fold)
        
        # 验证预测
        y_pred = model.predict(X_valid_scaled)
        mse_scores.append(mean_squared_error(y_valid_fold, y_pred))

    return np.mean(mse_scores)

# 优化过程
study_catboost = optuna.create_study(direction='minimize')
study_catboost.optimize(objective_catboost, n_trials=20, show_progress_bar=True)

# 获取最佳参数
best_params_catboost = study_catboost.best_params

# 训练最终模型
final_model = make_pipeline(
    StandardScaler(),
    CatBoostRegressor(**best_params_catboost, verbose=False)
)
final_model.fit(x_train, y_train)

# 评估测试集
y_pred = final_model.predict(x_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE with Best CatBoost Model: {test_mse}")
joblib.dump(final_model, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/Stacking_model/final_catboost_model_LC.pkl')
# 定义目标函数
def objective_adaboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 300),  # 弱学习器的数量
        'learning_rate': trial.suggest_float('learning_rate', 1e-10, 1.0, log=True),  # 学习率
        'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),  # 损失函数
        'estimator': DecisionTreeRegressor(
            max_depth=trial.suggest_int('max_depth', 1, 30),  # 决策树的最大深度
            min_samples_split=trial.suggest_int('min_samples_split', 2, 50),  # 内部节点再划分所需最小样本数
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 50),  # 叶子节点最少样本数
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2'])  # 最大特征数
        ),
        'random_state': trial.suggest_int('random_state', 1, 100),  # 随机种子
    }

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for train_idx, valid_idx in kf.split(x_train):
        # 数据标准化（每个fold独立进行）
        scaler = StandardScaler()
        
        # 划分原始数据
        X_train_fold = x_train.iloc[train_idx]
        X_valid_fold = x_train.iloc[valid_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        # 标准化处理（仅用训练fold数据拟合）
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_valid_scaled = scaler.transform(X_valid_fold)

        # 模型训练
        model = AdaBoostRegressor(**params)
        model.fit(X_train_scaled, y_train_fold)
        
        # 验证预测
        y_pred = model.predict(X_valid_scaled)
        mse_scores.append(mean_squared_error(y_valid_fold, y_pred))

    return np.mean(mse_scores)

# 优化过程
study_adaboost = optuna.create_study(direction='minimize')
study_adaboost.optimize(objective_adaboost, n_trials=20, show_progress_bar=True)

# 获取最佳参数
best_params_adaboost = study_adaboost.best_params

# 训练最终模型
final_model = make_pipeline(
    StandardScaler(),
    AdaBoostRegressor(
        n_estimators=best_params_adaboost['n_estimators'],
        learning_rate=best_params_adaboost['learning_rate'],
        loss=best_params_adaboost['loss'],
        estimator=DecisionTreeRegressor(
            max_depth=best_params_adaboost['max_depth'],
            min_samples_split=best_params_adaboost['min_samples_split'],
            min_samples_leaf=best_params_adaboost['min_samples_leaf'],
            max_features=best_params_adaboost['max_features']
        ),
        random_state=best_params_adaboost['random_state']
    )
)
final_model.fit(x_train, y_train)

# 评估测试集
y_pred = final_model.predict(x_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE with Best AdaBoost Model: {test_mse}")
joblib.dump(final_model, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/Stacking_model/final_adaboost_model_LC.pkl')

# 定义目标函数
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 40),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 70),  # 增加范围
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),  # 增加范围
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'max_features': trial.suggest_int('max_features', 1, 50),
        'max_samples': trial.suggest_float('max_samples', 0.1, 1.0) if trial.params['bootstrap'] else None,
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 1),
        'oob_score': trial.suggest_categorical('oob_score', [True, False]) if trial.params['bootstrap'] else False,
        'criterion': trial.suggest_categorical('criterion', ['poisson', 'absolute_error', 'friedman_mse', 'squared_error'])
    }
    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(**params, n_jobs=-1)
    )
    
    # 使用5折交叉验证计算MSE
    scores = cross_val_score(model, x_train, y_train, 
                            cv=5, 
                            scoring='neg_mean_squared_error',
                            n_jobs=-1)  # 并行加速
    mse = -scores.mean()
    return mse

# 使用Optuna优化
study = optuna.create_study(direction='minimize')
study.optimize(objective_rf, n_trials=200, show_progress_bar=True)

# 获取最佳参数
best_params_rf = study.best_params

# 训练最终模型
final_model = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(**best_params_rf, n_jobs=-1)
)
final_model.fit(x_train, y_train)

# 评估测试集
y_pred = final_model.predict(x_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE with Best Random Forest Model: {test_mse}")
joblib.dump(final_model, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/Stacking_model/final_rf_model_LC.pkl')

# 定义目标函数（Lasso）
def objective_ridge(trial):
    params = {
        'alpha': trial.suggest_loguniform('alpha', 1e-30, 10.0),  # 正则化强度
        'max_iter': trial.suggest_int('max_iter', 100, 2000),  # 最大迭代次数
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),  # 是否拟合截距项
        'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),  # 求解器
    }

    kf = KFold(n_splits=5, shuffle=True)
    mse_scores = []

    for train_idx, valid_idx in kf.split(x_train):
        scaler = StandardScaler()
        X_train_fold = x_train.iloc[train_idx]
        X_valid_fold = x_train.iloc[valid_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_valid_scaled = scaler.transform(X_valid_fold)

        model = Ridge(**params)
        model.fit(X_train_scaled, y_train_fold)
        y_pred = model.predict(X_valid_scaled)
        mse_scores.append(mean_squared_error(y_valid_fold, y_pred))

    return np.mean(mse_scores)

# 优化 Ridge 模型
study_ridge = optuna.create_study(direction='minimize')
study_ridge.optimize(objective_ridge, n_trials=100, show_progress_bar=True)
best_params_ridge = study_ridge.best_params
print("Best Ridge parameters:", best_params_ridge)


# 训练最终的 Ridge 模型
final_ridge_model = make_pipeline(
    StandardScaler(),
    Ridge(**best_params_ridge)
)
final_ridge_model.fit(x_train, y_train)
ridge_y_pred = final_ridge_model.predict(x_test)
ridge_test_mse = mean_squared_error(y_test, ridge_y_pred)
print(f"Test MSE with Best Ridge Model: {ridge_test_mse}")
joblib.dump(final_ridge_model, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/Stacking_model/final_ridge_model_LC.pkl')
def objective_elasticnet(trial):
    params = {
        'alpha': trial.suggest_loguniform('alpha', 1e-60, 1.0),  # 正则化强度
        'l1_ratio': trial.suggest_uniform('l1_ratio', 0.0, 1.0),  # L1 和 L2 混合比例
        'max_iter': trial.suggest_int('max_iter', 100, 1000),  # 最大迭代次数
        'tol': trial.suggest_loguniform('tol', 1e-60, 1e-2),  # 收敛阈值
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),  # 是否拟合截距项
        'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),  # 特征选择策略
        'random_state': trial.suggest_int('random_state', 1, 100),  # 随机种子
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for train_idx, valid_idx in kf.split(x_train):
        scaler = StandardScaler()
        X_train_fold = x_train.iloc[train_idx]
        X_valid_fold = x_train.iloc[valid_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_valid_scaled = scaler.transform(X_valid_fold)

        model = ElasticNet(**params)
        model.fit(X_train_scaled, y_train_fold)
        y_pred = model.predict(X_valid_scaled)
        mse_scores.append(mean_squared_error(y_valid_fold, y_pred))

    return np.mean(mse_scores)

study_elasticnet = optuna.create_study(direction='minimize')
study_elasticnet.optimize(objective_elasticnet, n_trials=50, show_progress_bar=True)
best_params_elasticnet = study_elasticnet.best_params
print("Best ElasticNet parameters:", best_params_elasticnet)

final_elasticnet_model = make_pipeline(
    StandardScaler(),
    ElasticNet(**best_params_elasticnet)
)
final_elasticnet_model.fit(x_train, y_train)
elasticnet_y_pred = final_elasticnet_model.predict(x_test)
elasticnet_test_mse = mean_squared_error(y_test, elasticnet_y_pred)
print(f"Test MSE with Best ElasticNet Model: {elasticnet_test_mse}")

joblib.dump(final_elasticnet_model, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/Stacking_model/final_elasticnet_model_LC.pkl')

# 定义基学习器
xgb_reg = XGBRegressor(**best_params_xgb)
lgb_reg = LGBMRegressor(**best_params_lgb)
gbm_reg = GradientBoostingRegressor(**best_params_gbr)
rf_reg = RandomForestRegressor(**best_params_rf)
cat_reg = CatBoostRegressor(**best_params_catboost, verbose=0)  # 设置 verbose=0 避免日志输出
lasso_reg = Lasso(**best_params_lasso)
ridge_reg = Ridge(**best_params_ridge)
elasticnet_reg = ElasticNet(**best_params_elasticnet)
ada_reg=AdaBoostRegressor(
        n_estimators=best_params_adaboost['n_estimators'],
        learning_rate=best_params_adaboost['learning_rate'],
        loss=best_params_adaboost['loss'],
        estimator=DecisionTreeRegressor(
            max_depth=best_params_adaboost['max_depth'],
            min_samples_split=best_params_adaboost['min_samples_split'],
            min_samples_leaf=best_params_adaboost['min_samples_leaf'],
            max_features=best_params_adaboost['max_features']
        ),
        random_state=best_params_adaboost['random_state'])

# 定义基学习器列表
base_learners = [
    ("XGBoost", xgb_reg),
    ("LightGBM", lgb_reg),
    ("GBDT", gbm_reg),
    ("RandomForest", rf_reg),
    ("Lasso", lasso_reg),
    ("CatBoost", cat_reg),
    ("Ridge", ridge_reg),
    ("ElasticNet", elasticnet_reg),
    ("AdaBoost",ada_reg)
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
    ("XGB", xgb_reg, "regressor"),
    ("LGB", lgb_reg, "regressor"),
    ("GBM", gbm_reg, "regressor"),
    ("RF", rf_reg, "regressor"),
    ("LASSO", lasso_reg, "regressor"),
    ("CatBoost", cat_reg, "regressor"),
    ("Ridge", ridge_reg, "regressor"),
    ("ElasticNet", elasticnet_reg, "regressor")
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

# 保存模型（可选）

joblib.dump(final_stacking_regressor, '/Users/jiaoyuan/Documents/GitHub/code_ml_dft/SGM_LC-ωPBE/Stacking_model/final_stacking_regressor_LC.pkl')


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

lgb_reg = LGBMRegressor(**best_params_lgb)
lgb_reg.fit(x_train, y_train)
y_pred_lgb = lgb_reg.predict(x_test)
evaluate_model("LightGBMRegressor", y_test, y_pred_lgb)


gbm_reg = GradientBoostingRegressor(**best_params_gbr)
gbm_reg.fit(x_train, y_train)
y_pred_gbr = gbm_reg.predict(x_test)
evaluate_model("GBDTRegressor", y_test, y_pred_gbr)

xgb_reg = XGBRegressor(**best_params_xgb)
xgb_reg.fit(x_train, y_train)
y_pred_xg_reg= xgb_reg.predict(x_test)
evaluate_model("XGBoostRegressor", y_test, y_pred_xg_reg)

rf_reg = RandomForestRegressor(**best_params_rf)
rf_reg.fit(x_train, y_train)
y_pred_rf = rf_reg.predict(x_test)
evaluate_model("RandomForestRegressor", y_test, y_pred_rf)

lasso_reg = Lasso(**best_params_lasso)
lasso_reg.fit(x_train, y_train)
y_pred_lasso_reg = lasso_reg.predict(x_test)
evaluate_model("Lasso", y_test, y_pred_lasso_reg)

cat_reg = CatBoostRegressor(**best_params_catboost)
cat_reg.fit(x_train, y_train)
y_pred_cat = cat_reg.predict(x_test)
evaluate_model("CatBoostRegressor", y_test, y_pred_cat)

ridge_reg = Ridge(**best_params_ridge)
ridge_reg.fit(x_train, y_train)
y_pred_ridge_reg = ridge_reg.predict(x_test)
evaluate_model("Ridge", y_test, y_pred_ridge_reg)

elasticnet_reg=ElasticNet(**best_params_elasticnet)
elasticnet_reg.fit(x_train, y_train)
y_pred_elasticnet_reg=elasticnet_reg.predict(x_test)
evaluate_model("ElasticNet", y_test, y_pred_elasticnet_reg)

ada_reg=AdaBoostRegressor(
        n_estimators=best_params_adaboost['n_estimators'],
        learning_rate=best_params_adaboost['learning_rate'],
        loss=best_params_adaboost['loss'],
        estimator=DecisionTreeRegressor(
            max_depth=best_params_adaboost['max_depth'],
            min_samples_split=best_params_adaboost['min_samples_split'],
            min_samples_leaf=best_params_adaboost['min_samples_leaf'],
            max_features=best_params_adaboost['max_features']
        ),
        random_state=best_params_adaboost['random_state']
        )
ada_reg.fit(x_train, y_train)
y_pred_ada_reg=ada_reg.predict(x_test)
evaluate_model("AdaBoostRegressor", y_test, y_pred_ada_reg)
pred1=final_stacking_regressor.predict(x_test)
evaluate_model("StackingRegressor", y_test, pred1)
# # 创建DataFrame
results_df = pd.DataFrame(results)

# 打印表格
print(results_df)