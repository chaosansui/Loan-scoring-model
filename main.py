import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_absolute_error
import os
from data.data_processor import process_jsonl_to_dataframe
from features.feature_engineering import build_credit_features
from xgboost import XGBRegressor

# 导入配置文件
from config import (
    CREDIT_SCORE_MODEL_PATH,
    DEFAULT_PROB_MODEL_PATH,
    AMOUNT_RECOMMENDER_PATH,
    TERM_RECOMMENDER_PATH,
    FEATURE_COLUMNS_PATH,
    ENCODER_PATH,
    BUSINESS_RULES,
    VALIDATION_RULES
)

# 辅助函数：计算KS值
def calculate_ks_statistic(y_true, y_pred_proba):
    """计算KS统计量"""
    df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    df['bucket'] = pd.qcut(df['y_pred_proba'], 10, labels=False, duplicates='drop')
    grouped = df.groupby('bucket')['y_true'].agg(['count', 'sum']).reset_index()
    grouped.rename(columns={'sum': 'bad', 'count': 'total'}, inplace=True)
    grouped['good'] = grouped['total'] - grouped['bad']
    grouped['bad_rate'] = grouped['bad'] / grouped['bad'].sum()
    grouped['good_rate'] = grouped['good'] / grouped['good'].sum()
    grouped['cum_bad_rate'] = grouped['bad_rate'].cumsum()
    grouped['cum_good_rate'] = grouped['good_rate'].cumsum()
    grouped['ks'] = np.abs(grouped['cum_bad_rate'] - grouped['cum_good_rate'])
    return grouped['ks'].max()

def create_credit_scores_from_labels(y):
    """根据违约标签创建信用评分"""
    # 违约用户给低分(300-500)，正常用户给高分(700-900)
    np.random.seed(42)  # 确保可重复性
    credit_scores = []
    for label in y:
        if label == 1:  # 违约
            score = np.random.randint(300, 500)
        else:  # 正常
            score = np.random.randint(700, 900)
        credit_scores.append(score)
    return np.array(credit_scores)

def train_base_models(X, y, random_seed=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost基模型
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc', random_state=42,
        tree_method='gpu_hist', device='cuda:3', n_estimators=100, max_depth=4
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # LightGBM基模型
    lgb_model = lgb.LGBMClassifier(random_state=42, n_estimators=100, max_depth=4)
    lgb_model.fit(X_train, y_train)
    lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
    
    # CatBoost基模型
    cb_model = cb.CatBoostClassifier(random_state=42, iterations=100, depth=4, verbose=0)
    cb_model.fit(X_train, y_train)
    cb_pred_proba = cb_model.predict_proba(X_test)[:, 1]
    
    return {
        'xgb': xgb_pred_proba,
        'lgb': lgb_pred_proba,
        'cb': cb_pred_proba
    }, (X_test, y_test)


def train_meta_model(X_meta, y_test, model_path=DEFAULT_PROB_MODEL_PATH):
    meta_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc', random_state=42,
        tree_method='gpu_hist', device='cuda:3', n_estimators=100, max_depth=4
    )
    meta_model.fit(X_meta, y_test)
    y_pred_proba = meta_model.predict_proba(X_meta)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ks_score = calculate_ks_statistic(y_test, y_pred_proba)
    
    print(f"元模型 AUC: {auc_score:.4f}")
    print(f"元模型 KS: {ks_score:.4f}")
    joblib.dump(meta_model, model_path)
    print(f"元模型已保存至 {model_path}\n")
    return meta_model

def train_credit_score_model(X, y, model_path=CREDIT_SCORE_MODEL_PATH):
    """训练信用评分模型"""
    print("--- 开始训练信用评分模型 ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建信用评分标签
    y_train_score = create_credit_scores_from_labels(y_train)
    y_test_score = create_credit_scores_from_labels(y_test)
    
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        tree_method='gpu_hist',
        device='cuda:3',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1
    )
    
    # 简化的参数搜索
    param_dist = {
        'n_estimators': [80, 100, 120],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    
    search = RandomizedSearchCV(
        model, param_dist, n_iter=5, scoring='neg_mean_absolute_error',
        cv=3, verbose=1, n_jobs=-1, random_state=42
    )
    
    print("进行参数搜索...")
    search.fit(X_train, y_train_score)
    
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    mae_score = mean_absolute_error(y_test_score, y_pred)
    
    print(f"最佳参数: {search.best_params_}")
    print(f"信用评分模型 MAE: {mae_score:.2f}")
    print(f"评分范围: {y_pred.min():.0f} - {y_pred.max():.0f}")
    
    joblib.dump(best_model, model_path)
    print(f"信用评分模型已保存至 {model_path}\n")
    return best_model

def train_default_probability_model(X, y, model_path=DEFAULT_PROB_MODEL_PATH):
    """训练违约概率模型"""
    print("--- 开始训练违约概率模型 ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        tree_method='gpu_hist',
        device='cuda:3'
    )
    
    param_distributions = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
    }

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=8,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    print("开始参数搜索...")
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ks_score = calculate_ks_statistic(y_test, y_pred_proba)
    
    print(f"最佳参数: {random_search.best_params_}")
    print(f"违约概率模型 AUC: {auc_score:.4f}")
    print(f"违约概率模型 KS: {ks_score:.4f}")
    
    joblib.dump(best_model, model_path)
    print(f"违约概率模型已保存至 {model_path}\n")
    return best_model

def train_recommendation_models(features_df, raw_df, credit_scores, default_probs):
    """训练推荐模型"""
    print("--- 开始训练推荐模型 ---")
    
    approved_codes = [3600, 3650, 3700]
    approved_mask = raw_df['stateCode'].isin(approved_codes)
    
    # 确保索引对齐
    common_idx = features_df.index.intersection(raw_df.index)
    features_df = features_df.loc[common_idx]
    raw_df = raw_df.loc[common_idx]
    credit_scores = credit_scores.loc[common_idx]
    default_probs = default_probs.loc[common_idx]
    
    # 获取批准数据
    y_amount = pd.to_numeric(raw_df.loc[approved_mask, 'loanAmountApproved'], errors='coerce').fillna(0)
    y_term = pd.to_numeric(raw_df.loc[approved_mask, 'totalTenorApproved'], errors='coerce').fillna(0)
    X_approved = features_df.loc[approved_mask]
    
    if X_approved.empty or len(y_amount) < 10 or len(y_term) < 10:
        print("批准数据不足，使用简化推荐模型")
        return train_simple_recommendation_models()
    
    # 增强特征：添加信用信息
    X_enhanced = X_approved.copy()
    X_enhanced['credit_score'] = credit_scores.loc[approved_mask].values
    X_enhanced['default_prob'] = default_probs.loc[approved_mask].values
    
    print(f"用于推荐模型训练的样本数: {len(X_enhanced)}")
    
    # 训练金额推荐模型
    print("训练金额推荐模型...")
    amount_model = XGBRegressor(
        objective='reg:squarederror', 
        random_state=42, 
        tree_method='gpu_hist',
        device='cuda:3',
        n_estimators=100,
        max_depth=4
    )
    amount_model.fit(X_enhanced, y_amount)
    
    # 训练期限推荐模型
    print("训练期限推荐模型...")
    term_model = XGBRegressor(
        objective='reg:squarederror', 
        random_state=42, 
        tree_method='gpu_hist',
        device='cuda:3',
        n_estimators=100,
        max_depth=4
    )
    term_model.fit(X_enhanced, y_term)
    
    # 评估模型
    amount_pred = amount_model.predict(X_enhanced)
    term_pred = term_model.predict(X_enhanced)
    mae_amount = mean_absolute_error(y_amount, amount_pred)
    mae_term = mean_absolute_error(y_term, term_pred)
    
    print(f"金额推荐模型 MAE: {mae_amount:.2f}")
    print(f"期限推荐模型 MAE: {mae_term:.2f}")
    
    # 保存模型
    joblib.dump(amount_model, AMOUNT_RECOMMENDER_PATH)
    joblib.dump(term_model, TERM_RECOMMENDER_PATH)
    print(f"推荐模型已保存\n")
    
    return amount_model, term_model

def train_simple_recommendation_models():
    """训练简化推荐模型（当数据不足时使用）"""
    print("使用简化推荐模型...")
    
    # 创建简单的基于规则的模型
    class SimpleRecommender:
        def predict(self, X):
            # 返回固定值，在实际应用中可以根据业务规则调整
            return np.array([50000] * len(X))
    
    amount_model = SimpleRecommender()
    term_model = SimpleRecommender()
    
    joblib.dump(amount_model, AMOUNT_RECOMMENDER_PATH)
    joblib.dump(term_model, TERM_RECOMMENDER_PATH)
    
    return amount_model, term_model

def load_and_preprocess_data():
    print("加载和处理数据...")
    raw_df = process_jsonl_to_dataframe('data/loan.jsonl')
    if raw_df.empty:
        raise ValueError("数据加载失败")
    
    approved_codes = [3600, 3650, 3700]
    default_codes = [3750, 3800, 4200, 3680]
    rejected_codes = [1700, 1900, 2900]
    
    raw_df = raw_df[~raw_df['stateCode'].isin(rejected_codes)]
    if raw_df.empty:
        raise ValueError("过滤后无有效数据")
    
    y = raw_df['stateCode'].apply(lambda x: 1 if x in default_codes else 0)
    features_df, encoder = build_credit_features(raw_df)
    if features_df.empty:
        raise ValueError("特征工程失败")
    
    common_idx = raw_df.index.intersection(features_df.index)
    raw_df = raw_df.loc[common_idx]
    features_df = features_df.loc[common_idx]
    y = y.loc[common_idx]
    
    features_df = features_df.fillna(0)
    joblib.dump(features_df.columns.tolist(), FEATURE_COLUMNS_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"最终训练数据: {len(features_df)} 个样本")
    return raw_df, features_df, y

def main():
    try:
        os.makedirs('models/train_models', exist_ok=True)
        print("=== 开始集成学习训练 ===")
        
        # 1. 加载和预处理数据
        raw_df, features_df, y = load_and_preprocess_data()
        
        # 2. 训练第一层基模型
        base_preds, (X_test, y_test) = train_base_models(features_df, y)
        X_meta = pd.DataFrame(base_preds)
        
        # 3. 训练第二层元模型
        train_meta_model(X_meta, y_test)
        
        # 4. 训练信用评分模型（可选保持）
        credit_model = train_credit_score_model(features_df, y)
        credit_scores = pd.Series(credit_model.predict(features_df), index=features_df.index)
        
        # 5. 训练推荐模型（使用元模型输出作为default_probs）
        default_model = joblib.load(DEFAULT_PROB_MODEL_PATH)
        default_probs = pd.Series(default_model.predict_proba(features_df)[:, 1], index=features_df.index)
        amount_model, term_model = train_recommendation_models(features_df, raw_df, credit_scores, default_probs)
        
        print("=== 训练完成 ===")
        print("模型架构: 基模型(XGBoost/LightGBM/CatBoost) → 元模型(XGBoost) → 推荐模型")
        
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()