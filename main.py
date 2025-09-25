import pandas as pd
import numpy as np
import joblib
import os
import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (roc_auc_score, mean_absolute_error, classification_report, 
                           confusion_matrix, average_precision_score, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# 导入配置文件和您的数据处理函数
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

from data.data_processor import process_jsonl_to_dataframe,process_api_data
# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class HighStandardEnsemble:
    """
    高标准集成学习训练系统 - 适配您的数据格式
    """
    
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models = {}
        self.meta_model = None
        self.evaluation_results = {}
        self.meta_feature_names = None
        self.fitted_base_models = {}
        self.feature_encoder = None
        self.scaler = None
        
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
        """基于您的数据构建信用特征"""
        print("构建信用特征...")
        
        # 复制原始数据
        features_df = raw_df.copy()
        
        # 定义违约状态码（根据您的业务逻辑调整）
        default_codes = [3750, 3800, 4200, 3680]  # 违约状态码
        approved_codes = [3600, 3650, 3700]  # 批准状态码
        
        # 创建目标变量
        features_df['is_default'] = features_df['stateCode'].apply(
            lambda x: 1 if x in default_codes else 0
        )
        
        # 特征工程
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
                features_df[feature] = features_df[feature].fillna(features_df[feature].median())
        
        # 2. 分类特征编码
        categorical_features = ['gender', 'propertyType', 'industry', 'employmentType']
        
        self.feature_encoder = {}
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
        
        # 3. 创建衍生特征
        # 收入负债比
        if 'salaryAmount' in features_df.columns and 'loanAmountApplied' in features_df.columns:
            features_df['income_loan_ratio'] = features_df['salaryAmount'] / (features_df['loanAmountApplied'] + 1)
        
        # DSR变化
        if 'preDsr' in features_df.columns and 'postDsr' in features_df.columns:
            features_df['dsr_change'] = features_df['postDsr'] - features_df['preDsr']
        
        # 贷款期限特征
        if 'totalTenorApplied' in features_df.columns and 'loanAmountApplied' in features_df.columns:
            features_df['monthly_payment'] = features_df['loanAmountApplied'] / (features_df['totalTenorApplied'] + 1)
        
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
        
        # 5. 特征标准化
        self.scaler = StandardScaler()
        features_df[available_features] = self.scaler.fit_transform(features_df[available_features])
        
        print(f"构建了 {len(available_features)} 个特征")
        return features_df, self.feature_encoder
    
    def comprehensive_model_evaluation(self, model, X_test, y_test, model_name=""):
        """全面的模型评估"""
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
                'precision': precision, 'recall': recall, 'f1': f1,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            }
        
        print(f"\n=== {model_name} 综合评估结果 ===")
        print(f"AUC: {auc:.4f}")
        print(f"KS: {ks:.4f}")
        print(f"Average Precision: {ap:.4f}")
        
        for threshold, metrics in threshold_results.items():
            print(f"阈值 {threshold}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        return {
            'auc': auc, 'ks': ks, 'ap': ap,
            'threshold_results': threshold_results, 'y_pred_proba': y_pred_proba
        }
    
    def train_stacking_ensemble(self, X, y, X_test=None, y_test=None):
        """严谨的Stacking集成训练"""
        print("=== 开始严谨Stacking训练 ===")
        
        # 第一层基模型定义
        base_models = {
            'xgb1': xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc',
                tree_method='hist', device='cuda',
                n_estimators=150, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=self.random_state
            ),
            'xgb2': xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc',
                tree_method='hist', device='cuda', 
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.7, random_state=self.random_state + 1
            ),
            'lgb': lgb.LGBMClassifier(
                device='gpu', n_estimators=150, max_depth=6,
                learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, verbose=-1
            ),
            'catboost': cb.CatBoostClassifier(
                task_type='GPU', iterations=150, depth=6,
                learning_rate=0.1, random_seed=self.random_state, verbose=0
            )
        }
        
        self.base_models = base_models
        
        n_samples = X.shape[0]
        meta_features = np.zeros((n_samples, len(base_models)))
        
        if X_test is not None:
            n_test = X_test.shape[0]
            test_meta_features = np.zeros((n_test, len(base_models)))
        else:
            test_meta_features = None
        
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        print("训练第一层基模型...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"  第 {fold+1}/{self.n_folds} 折")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            for i, (name, model) in enumerate(base_models.items()):
                from sklearn.base import clone
                model_clone = clone(model)
                
                model_clone.fit(X_train, y_train)
                val_pred = model_clone.predict_proba(X_val)[:, 1]
                meta_features[val_idx, i] = val_pred
                
                # 保存最后一折的模型用于后续预测
                if fold == self.n_folds - 1:
                    self.fitted_base_models[name] = clone(model)
                    self.fitted_base_models[name].fit(X, y)
                
                # 测试集预测
                if X_test is not None:
                    test_pred = model_clone.predict_proba(X_test)[:, 1]
                    test_meta_features[:, i] += test_pred / self.n_folds
        
        # 创建元特征DataFrame
        meta_train = pd.DataFrame(meta_features, columns=base_models.keys())
        self.meta_feature_names = list(base_models.keys())
        
        if X_test is not None:
            meta_test = pd.DataFrame(test_meta_features, columns=base_models.keys())
        else:
            meta_test = None
        
        print("第一层训练完成")
        return meta_train, meta_test
    
    def transform_to_meta_features(self, X):
        """将原始特征转换为元特征"""
        if not self.fitted_base_models:
            raise ValueError("请先训练Stacking模型")
        
        meta_features = []
        for name in self.meta_feature_names:
            model = self.fitted_base_models[name]
            pred_proba = model.predict_proba(X)[:, 1]
            meta_features.append(pred_proba)
        
        meta_df = pd.DataFrame(np.column_stack(meta_features), 
                             columns=self.meta_feature_names,
                             index=X.index)
        return meta_df
    
    def train_advanced_meta_model(self, X_meta, y, X_meta_test=None, y_test=None):
        """高级元模型训练与选择"""
        print("\n=== 训练高级元模型 ===")
        
        meta_candidates = {
            'xgb_meta': xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc',
                tree_method='hist', device='cuda',
                n_estimators=100, max_depth=4, learning_rate=0.05,
                random_state=self.random_state
            ),
            'logistic': LogisticRegression(
                random_state=self.random_state, max_iter=1000, C=0.1
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(64, 32), random_state=self.random_state,
                max_iter=500, early_stopping=True
            )
        }
        
        best_score = 0
        best_model = None
        best_name = ""
        
        for name, model in meta_candidates.items():
            try:
                scores = cross_val_score(model, X_meta, y, cv=5, scoring='roc_auc')
                mean_score = scores.mean()
                std_score = scores.std()
                
                print(f"{name:15} AUC: {mean_score:.4f} (±{std_score:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_name = name
            except Exception as e:
                print(f"{name} 训练失败: {e}")
                continue
        
        print(f"\n选择最佳元模型: {best_name}")
        best_model.fit(X_meta, y)
        self.meta_model = best_model
        
        if X_meta_test is not None and y_test is not None:
            eval_results = self.comprehensive_model_evaluation(
                best_model, X_meta_test, y_test, f"Stacking元模型 ({best_name})"
            )
            self.evaluation_results['meta_model'] = eval_results
        
        return best_model
    
    def train_credit_score_model(self, X, y, model_path=CREDIT_SCORE_MODEL_PATH):
        """训练信用评分模型"""
        print("\n=== 训练信用评分模型 ===")
        
        credit_scores = self.create_credit_scores_from_labels(y)
        
        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror', tree_method='hist',
            device='cuda', random_state=self.random_state
        )
        
        search = RandomizedSearchCV(
            model, param_dist, n_iter=10, scoring='neg_mean_absolute_error',
            cv=3, verbose=1, n_jobs=1, random_state=self.random_state
        )
        
        search.fit(X, credit_scores)
        best_model = search.best_estimator_
        
        y_pred = best_model.predict(X)
        mae = mean_absolute_error(credit_scores, y_pred)
        
        print(f"最佳参数: {search.best_params_}")
        print(f"信用评分模型 MAE: {mae:.2f}")
        print(f"评分范围: {y_pred.min():.0f} - {y_pred.max():.0f}")
        
        self.save_model_with_version(best_model, model_path, "credit_score")
        return best_model
    
    def train_recommendation_models(self, features_df, raw_df, credit_scores, default_probs):
        """训练推荐模型"""
        print("\n=== 训练推荐模型 ===")
        
        approved_codes = [3600, 3650, 3700]
        approved_mask = raw_df['stateCode'].isin(approved_codes)
        
        # 检查必要的列是否存在
        required_columns = ['loanAmountApproved', 'totalTenorApproved']
        missing_columns = [col for col in required_columns if col not in raw_df.columns]
        
        if missing_columns:
            print(f"缺少必要列 {missing_columns}，使用简化推荐模型")
            return self.train_simple_recommendation_models()
        
        common_idx = features_df.index.intersection(raw_df.index)
        features_df = features_df.loc[common_idx]
        raw_df = raw_df.loc[common_idx]
        
        y_amount = pd.to_numeric(raw_df.loc[approved_mask, 'loanAmountApproved'], errors='coerce')
        y_term = pd.to_numeric(raw_df.loc[approved_mask, 'totalTenorApproved'], errors='coerce')
        
        valid_mask = (~y_amount.isna()) & (~y_term.isna()) & (y_amount > 0) & (y_term > 0)
        y_amount = y_amount[valid_mask]
        y_term = y_term[valid_mask]
        X_approved = features_df.loc[approved_mask].loc[valid_mask]
        
        if len(X_approved) < 10:
            print("批准数据不足，使用简化推荐模型")
            return self.train_simple_recommendation_models()
        
        X_enhanced = X_approved.copy()
        X_enhanced['credit_score'] = credit_scores.loc[X_enhanced.index].values
        X_enhanced['default_prob'] = default_probs.loc[X_enhanced.index].values
        
        print(f"推荐模型训练样本数: {len(X_enhanced)}")
        
        amount_model = xgb.XGBRegressor(
            objective='reg:squarederror', tree_method='hist',
            device='cuda', n_estimators=100, max_depth=4,
            random_state=self.random_state
        )
        amount_model.fit(X_enhanced, y_amount)
        
        term_model = xgb.XGBRegressor(
            objective='reg:squarederror', tree_method='hist',
            device='cuda', n_estimators=100, max_depth=4,
            random_state=self.random_state
        )
        term_model.fit(X_enhanced, y_term)
        
        amount_pred = amount_model.predict(X_enhanced)
        term_pred = term_model.predict(X_enhanced)
        mae_amount = mean_absolute_error(y_amount, amount_pred)
        mae_term = mean_absolute_error(y_term, term_pred)
        
        print(f"金额推荐模型 MAE: {mae_amount:.2f}")
        print(f"期限推荐模型 MAE: {mae_term:.2f}")
        
        self.save_model_with_version(amount_model, AMOUNT_RECOMMENDER_PATH, "amount_recommender")
        self.save_model_with_version(term_model, TERM_RECOMMENDER_PATH, "term_recommender")
        
        return amount_model, term_model
    
    def train_simple_recommendation_models(self):
        """简化推荐模型"""
        print("使用简化推荐模型...")
        
        class SimpleRecommender:
            def predict(self, X):
                return np.full(len(X), 50000)  # 默认推荐5万
        
        amount_model = SimpleRecommender()
        term_model = SimpleRecommender()
        
        joblib.dump(amount_model, AMOUNT_RECOMMENDER_PATH)
        joblib.dump(term_model, TERM_RECOMMENDER_PATH)
        print("简化推荐模型已保存")
        
        return amount_model, term_model
    
    def save_model_with_version(self, model, base_path, model_type):
        """带版本号的模型保存"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_path = f"{base_path.split('.')[0]}_{model_type}_{timestamp}.joblib"
        joblib.dump(model, versioned_path)
        joblib.dump(model, base_path)
        print(f"模型已保存: {versioned_path}")
        return versioned_path
    
    def analyze_feature_importance(self, meta_train):
        """分析特征重要性"""
        if hasattr(self.meta_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': meta_train.columns,
                'importance': self.meta_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n=== 元模型特征重要性 ===")
            for i, row in importance_df.head(10).iterrows():
                print(f"{row['feature']:15} {row['importance']:.4f}")
            
            return importance_df
        return None
    
    def load_and_preprocess_data(self, data_path='data/loan.jsonl'):
        """加载和预处理数据 - 使用您的数据处理函数"""
        print("加载和处理数据...")
        
        # 使用您的函数加载数据
        raw_df = process_jsonl_to_dataframe(data_path)
        
        if raw_df.empty:
            raise ValueError("数据加载失败，DataFrame为空")
        
        print(f"原始数据加载完成: {len(raw_df)} 行")
        
        # 过滤数据（移除拒绝状态）
        rejected_codes = [1700, 1900, 2900]
        raw_df = raw_df[~raw_df['stateCode'].isin(rejected_codes)]
        
        print(f"过滤后数据: {len(raw_df)} 行")
        
        if raw_df.empty:
            raise ValueError("过滤后无有效数据")
        
        # 构建特征
        features_df, encoder = self.build_credit_features(raw_df)
        
        # 创建目标变量
        default_codes = [3750, 3800, 4200, 3680]
        y = raw_df['stateCode'].apply(lambda x: 1 if x in default_codes else 0)
        
        # 对齐索引
        common_idx = raw_df.index.intersection(features_df.index)
        raw_df = raw_df.loc[common_idx]
        features_df = features_df.loc[common_idx]
        y = y.loc[common_idx]
        
        # 保存特征列和编码器
        joblib.dump(features_df.columns.tolist(), FEATURE_COLUMNS_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        
        print(f"最终训练数据: {len(features_df)} 个样本")
        print(f"违约样本比例: {y.mean():.2%}")
        
        return raw_df, features_df, y

def main():
    """主训练流程"""
    try:
        os.makedirs('models/train_models', exist_ok=True)
        
        ensemble = HighStandardEnsemble(n_folds=5, random_state=42)
        
        print("=== 开始高标准集成学习训练 ===")
        
        # 1. 加载数据（使用您的数据路径）
        raw_df, features_df, y = ensemble.load_and_preprocess_data('data/loan.jsonl')
        
        # 2. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"测试集: {X_test.shape[0]} 样本")
        print(f"特征维度: {X_train.shape[1]}")
        
        # 3. 训练Stacking集成
        meta_train, meta_test = ensemble.train_stacking_ensemble(X_train, y_train, X_test, y_test)
        
        # 4. 训练元模型
        meta_model = ensemble.train_advanced_meta_model(meta_train, y_train, meta_test, y_test)
        
        # 5. 特征重要性分析
        importance_df = ensemble.analyze_feature_importance(meta_train)
        
        # 6. 训练信用评分模型
        credit_model = ensemble.train_credit_score_model(features_df, y)
        credit_scores = pd.Series(credit_model.predict(features_df), index=features_df.index)
        
        # 7. 生成违约概率
        print("\n=== 生成违约概率 ===")
        meta_features_full = ensemble.transform_to_meta_features(features_df)
        default_probs = pd.Series(meta_model.predict_proba(meta_features_full)[:, 1], 
                                index=features_df.index)
        
        print(f"违约概率范围: {default_probs.min():.3f} - {default_probs.max():.3f}")
        
        # 8. 训练推荐模型
        amount_model, term_model = ensemble.train_recommendation_models(
            features_df, raw_df, credit_scores, default_probs
        )
        
        # 9. 最终总结
        print("\n" + "="*50)
        print("=== 高标准集成学习训练完成 ===")
        print("="*50)
        print("模型架构: 4种基模型 → 最优元模型 → 推荐系统")
        print(f"基模型: {list(ensemble.base_models.keys())}")
        print(f"元模型类型: {type(ensemble.meta_model).__name__}")
        
        if 'meta_model' in ensemble.evaluation_results:
            results = ensemble.evaluation_results['meta_model']
            print(f"最终模型 AUC: {results['auc']:.4f}")
            print(f"最终模型 KS: {results['ks']:.4f}")
        
        print("\n所有模型已保存到 models/train_models/ 目录")
        
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()