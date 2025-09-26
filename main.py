import pandas as pd
import numpy as np
import joblib
import os
import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, mean_absolute_error, confusion_matrix, average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# 导入配置文件
from config import (
    CREDIT_SCORE_MODEL_PATH,
    DEFAULT_PROB_MODEL_PATH,
    AMOUNT_RECOMMENDER_PATH,
    TERM_RECOMMENDER_PATH,
    FEATURE_COLUMNS_PATH,
    ENCODER_PATH,
    BASE_MODELS_PATH,
    META_MODEL_PATH,
    TRAINING_METADATA_PATH,
    BUSINESS_RULES,
    MODELS_DIR
)

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class HighStandardEnsemble:
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models = {}
        self.meta_model = None
        self.evaluation_results = {}
        self.fitted_base_models = {}
        self.feature_encoder = {}
        self.feature_columns = []
        
    def calculate_ks_statistic(self, y_true, y_pred_proba):
        """计算KS统计量"""
        df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
        df = df.sort_values('y_pred_proba', ascending=False)
        df['bucket'] = pd.qcut(df['y_pred_proba'], 10, labels=False, duplicates='drop')
        
        grouped = df.groupby('bucket').agg({
            'y_true': ['count', 'sum']
        }).reset_index()
        
        grouped.columns = ['bucket', 'total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']
        grouped['bad_rate'] = grouped['bad'] / grouped['bad'].sum()
        grouped['good_rate'] = grouped['good'] / grouped['good'].sum()
        grouped['cum_bad_rate'] = grouped['bad_rate'].cumsum()
        grouped['cum_good_rate'] = grouped['good_rate'].cumsum()
        grouped['ks'] = np.abs(grouped['cum_bad_rate'] - grouped['cum_good_rate'])
        
        return grouped['ks'].max()
    
    def create_credit_scores_from_labels(self, y):
        """根据标签生成信用评分"""
        np.random.seed(self.random_state)
        credit_scores = []
        for label in y:
            if label == 1:  # 违约
                score = np.random.normal(400, 50)
            else:  # 正常
                score = np.random.normal(700, 50)
            score = max(300, min(900, score))
            credit_scores.append(int(score))
        return np.array(credit_scores)
    
    def build_credit_features(self, raw_df):
        """修复版：构建信用特征"""
        print("构建信用特征...")
        
        # 复制原始数据
        features_df = raw_df.copy()
        
        # 定义违约状态码
        default_codes = [3750, 3800, 4200, 3680]
        
        # 创建目标变量
        features_df['is_default'] = features_df['stateCode'].apply(
            lambda x: 1 if x in default_codes else 0
        )
        
        # 1. 数值特征处理
        numerical_features = [
            'age', 'workDurationYear', 'salaryAmount', 
            'preDsr', 'postDsr', 'scorePl01', 'scoreCm01',
            'loanAmountApplied', 'totalTenorApplied', 'flatRateApplied'
        ]
        
        # 处理缺失值
        for feature in numerical_features:
            if feature in features_df.columns:
                features_df[feature] = pd.to_numeric(features_df[feature], errors='coerce')
                # 使用中位数填充，避免标准化错误
                median_val = features_df[feature].median()
                if pd.isna(median_val):
                    median_val = 0
                features_df[feature] = features_df[feature].fillna(median_val)
        
        # 2. 分类特征编码
        categorical_features = ['gender', 'propertyType', 'industry', 'employmentType']
        
        for feature in categorical_features:
            if feature in features_df.columns:
                # 处理缺失值
                features_df[feature] = features_df[feature].fillna('Unknown')
                
                # 编码低频类别
                value_counts = features_df[feature].value_counts()
                low_freq_categories = value_counts[value_counts < 10].index
                features_df[feature] = features_df[feature].apply(
                    lambda x: 'Other' if x in low_freq_categories else x
                )
                
                # Label Encoding
                encoder = LabelEncoder()
                features_df[feature] = encoder.fit_transform(features_df[feature].astype(str))
                self.feature_encoder[feature] = encoder
        
        # 3. 创建衍生特征（确保有有效数据）
        if 'salaryAmount' in features_df.columns and 'loanAmountApplied' in features_df.columns:
            # 避免除零错误
            loan_amount_safe = features_df['loanAmountApplied'].replace(0, 1)
            features_df['income_loan_ratio'] = features_df['salaryAmount'] / loan_amount_safe
        
        if 'preDsr' in features_df.columns and 'postDsr' in features_df.columns:
            features_df['dsr_change'] = features_df['postDsr'] - features_df['preDsr']
        
        if 'totalTenorApplied' in features_df.columns and 'loanAmountApplied' in features_df.columns:
            tenor_safe = features_df['totalTenorApplied'].replace(0, 1)
            features_df['monthly_payment'] = features_df['loanAmountApplied'] / tenor_safe
        
        # 4. 选择最终特征
        feature_columns = [
            'age', 'gender', 'workDurationYear', 'salaryAmount',
            'preDsr', 'postDsr', 'scorePl01', 'scoreCm01',
            'loanAmountApplied', 'totalTenorApplied', 'flatRateApplied',
            'income_loan_ratio', 'dsr_change', 'monthly_payment'
        ]
        
        # 只保留存在的特征
        available_features = [f for f in feature_columns if f in features_df.columns]
        features_df = features_df[available_features]
        self.feature_columns = available_features
        
        # 5. 最终数据清理（修复关键错误）
        # 确保所有值都是数值型
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            features_df[col] = features_df[col].fillna(0)
        
        print(f"构建了 {len(available_features)} 个特征")
        return features_df, self.feature_encoder
    
    def comprehensive_model_evaluation(self, model, X_test, y_test, model_name=""):
        """模型评估"""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        ks = self.calculate_ks_statistic(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        
        thresholds = [0.3, 0.5, 0.7]
        threshold_results = {}
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            threshold_results[threshold] = {
                'precision': precision, 'recall': recall, 'f1': f1
            }
        
        print(f"\n=== {model_name} 评估结果 ===")
        print(f"AUC: {auc:.4f}")
        print(f"KS: {ks:.4f}")
        print(f"Average Precision: {ap:.4f}")
        
        return {'auc': auc, 'ks': ks, 'ap': ap, 'threshold_results': threshold_results}
    
    def train_stacking_ensemble(self, X, y, X_test=None, y_test=None):
        """Stacking集成训练"""
        print("=== 开始Stacking训练 ===")
        
        base_models = {
            'xgb1': xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc',
                tree_method='hist', device='cuda',
                n_estimators=150, max_depth=6, random_state=self.random_state
            ),
            'xgb2': xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc',
                tree_method='hist', device='cuda', 
                n_estimators=200, max_depth=4, random_state=self.random_state + 1
            ),
            'lgb': lgb.LGBMClassifier(
                device='gpu', n_estimators=150, random_state=self.random_state, verbose=-1
            ),
            'catboost': cb.CatBoostClassifier(
                task_type='GPU', iterations=150, random_seed=self.random_state, verbose=0
            )
        }
        
        n_samples = X.shape[0]
        meta_features = np.zeros((n_samples, len(base_models)))
        
        if X_test is not None:
            n_test = X_test.shape[0]
            test_meta_features = np.zeros((n_test, len(base_models)))
        
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        print("训练基模型...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"  第 {fold+1}/{self.n_folds} 折")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            for i, (name, model) in enumerate(base_models.items()):
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                val_pred = model_clone.predict_proba(X_val)[:, 1]
                meta_features[val_idx, i] = val_pred
                
                if fold == self.n_folds - 1:
                    self.fitted_base_models[name] = clone(model)
                    self.fitted_base_models[name].fit(X, y)
                
                if X_test is not None:
                    test_pred = model_clone.predict_proba(X_test)[:, 1]
                    test_meta_features[:, i] += test_pred / self.n_folds
        
        meta_train = pd.DataFrame(meta_features, columns=base_models.keys())
        meta_test = pd.DataFrame(test_meta_features, columns=base_models.keys()) if X_test is not None else None
        
        print("基模型训练完成")
        return meta_train, meta_test
    
    def train_meta_model(self, X_meta, y, X_meta_test=None, y_test=None):
        """元模型训练"""
        print("\n=== 训练元模型 ===")
        
        meta_candidates = {
            'xgb_meta': xgb.XGBClassifier(n_estimators=100, random_state=self.random_state),
            'logistic': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'mlp': MLPClassifier(random_state=self.random_state, max_iter=500)
        }
        
        best_score = 0
        best_model = None
        
        for name, model in meta_candidates.items():
            try:
                scores = cross_val_score(model, X_meta, y, cv=5, scoring='roc_auc')
                mean_score = scores.mean()
                print(f"{name}: AUC = {mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
            except Exception as e:
                print(f"{name} 训练失败: {e}")
                continue
        
        print(f"选择最佳元模型: {type(best_model).__name__}")
        best_model.fit(X_meta, y)
        self.meta_model = best_model
        
        if X_meta_test is not None and y_test is not None:
            eval_results = self.comprehensive_model_evaluation(best_model, X_meta_test, y_test, "元模型")
            self.evaluation_results['meta_model'] = eval_results
        
        return best_model

def save_models_fixed_names(ensemble, credit_model, default_model, amount_model, term_model, 
                          feature_columns, encoder, evaluation_results=None):
    """保存模型"""
    print("\n=== 保存模型 ===")
    
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # 保存主要模型
        joblib.dump(credit_model, CREDIT_SCORE_MODEL_PATH)
        joblib.dump(default_model, DEFAULT_PROB_MODEL_PATH)
        joblib.dump(amount_model, AMOUNT_RECOMMENDER_PATH)
        joblib.dump(term_model, TERM_RECOMMENDER_PATH)
        
        # 保存Stacking模型
        if ensemble.fitted_base_models:
            joblib.dump(ensemble.fitted_base_models, BASE_MODELS_PATH)
        if ensemble.meta_model:
            joblib.dump(ensemble.meta_model, META_MODEL_PATH)
        
        # 保存特征配置
        joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        
        # 保存元数据
        meta_data = {
            'training_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'feature_count': len(feature_columns),
            'performance': evaluation_results or {}
        }
        joblib.dump(meta_data, TRAINING_METADATA_PATH)
        
        print("✓ 所有模型保存完成")
        return True
        
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return False

def load_and_preprocess_data():
    """加载数据"""
    try:
        from data_process import process_jsonl_to_dataframe
        raw_df = process_jsonl_to_dataframe('data/loan.jsonl')
        
        if raw_df.empty:
            raise ValueError("数据加载失败")
        
        # 过滤数据
        rejected_codes = [1700, 1900, 2900]
        raw_df = raw_df[~raw_df['stateCode'].isin(rejected_codes)]
        
        # 创建目标变量
        default_codes = [3750, 3800, 4200, 3680]
        y = raw_df['stateCode'].apply(lambda x: 1 if x in default_codes else 0)
        
        return raw_df, y
        
    except ImportError:
        print("使用示例数据...")
        n_samples = 1000
        raw_df = pd.DataFrame({
            'stateCode': np.random.choice([3600, 3650, 3700, 3750], n_samples),
            'age': np.random.randint(20, 60, n_samples),
            'salaryAmount': np.random.randint(20000, 100000, n_samples),
            'preDsr': np.random.uniform(0, 1, n_samples)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        return raw_df, y

def main():
    """主训练流程"""
    try:
        ensemble = HighStandardEnsemble(n_folds=5, random_state=42)
        print("=== 开始训练 ===")
        
        # 加载数据
        raw_df, y = load_and_preprocess_data()
        print(f"数据加载完成: {len(raw_df)} 样本")
        
        # 构建特征
        features_df, encoder = ensemble.build_credit_features(raw_df)
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # Stacking训练
        meta_train, meta_test = ensemble.train_stacking_ensemble(X_train, y_train, X_test, y_test)
        meta_model = ensemble.train_meta_model(meta_train, y_train, meta_test, y_test)
        
        # 信用评分模型
        print("\n=== 训练信用评分模型 ===")
        credit_scores = ensemble.create_credit_scores_from_labels(y)
        credit_model = xgb.XGBRegressor(random_state=42)
        credit_model.fit(features_df, credit_scores)
        
        # 推荐模型
        print("=== 训练推荐模型 ===")
        approved_mask = raw_df['stateCode'].isin([3600, 3650, 3700])
        X_approved = features_df[approved_mask]
        
        if len(X_approved) > 0:
            # 增强特征
            X_enhanced = X_approved.copy()
            X_enhanced['credit_score'] = credit_model.predict(X_approved)
            X_enhanced['default_prob'] = meta_model.predict_proba(X_approved)[:, 1]
            
            # 模拟目标变量
            y_amount = np.random.randint(10000, 200000, len(X_approved))
            y_term = np.random.randint(6, 36, len(X_approved))
            
            amount_model = xgb.XGBRegressor(random_state=42)
            term_model = xgb.XGBRegressor(random_state=42)
            amount_model.fit(X_enhanced, y_amount)
            term_model.fit(X_enhanced, y_term)
        else:
            # 简化模型
            class SimpleModel:
                def predict(self, X): return np.full(len(X), 50000)
            amount_model = term_model = SimpleModel()
        
        # 保存模型
        save_success = save_models_fixed_names(
            ensemble, credit_model, meta_model, amount_model, term_model,
            ensemble.feature_columns, encoder, ensemble.evaluation_results
        )
        
        if save_success:
            print("\n=== 训练完成 ===")
            print("模型文件:")
            for file in os.listdir(MODELS_DIR):
                if file.endswith('.joblib'):
                    print(f"  {file}")
        
    except Exception as e:
        print(f"训练错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()