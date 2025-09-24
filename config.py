import os

# 定义模型保存目录
MODELS_DIR = 'models/train_models'

# 确保目录存在
os.makedirs(MODELS_DIR, exist_ok=True)

# 串联决策流模型路径
CREDIT_SCORE_MODEL_PATH = os.path.join(MODELS_DIR, 'credit_score_model.joblib')          # 信用评分模型
DEFAULT_PROB_MODEL_PATH = os.path.join(MODELS_DIR, 'default_probability_model.joblib')   # 违约概率模型
AMOUNT_RECOMMENDER_PATH = os.path.join(MODELS_DIR, 'amount_recommender.joblib')          # 金额推荐模型
TERM_RECOMMENDER_PATH = os.path.join(MODELS_DIR, 'term_recommender.joblib')              # 期限推荐模型

# 特征工程配置路径
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, 'feature_columns.joblib')                # 特征列顺序
ENCODER_PATH = os.path.join(MODELS_DIR, 'encoder.joblib')                                # 编码器

# 向后兼容的路径（可选，如果你有其他代码还在引用这些路径）
LOGISTIC_MODEL_PATH = DEFAULT_PROB_MODEL_PATH  # 指向违约概率模型
XGBOOST_MODEL_PATH = DEFAULT_PROB_MODEL_PATH   # 指向违约概率模型

# API配置
API_HOST = '0.0.0.0'
API_PORT = 8888
DEBUG_MODE = True

# 业务规则配置
BUSINESS_RULES = {
    'risk_levels': {
        'high': {'max_amount_multiplier': 0.3, 'max_term': 6},
        'medium': {'max_amount_multiplier': 0.7, 'max_term': 18},
        'low': {'max_amount_multiplier': 1.0, 'max_term': 36}
    },
    'risk_thresholds': {
        'high_risk': {'default_prob': 0.6, 'credit_score': 500},
        'medium_risk': {'default_prob': 0.3, 'credit_score': 650}
    }
}

# 数据验证配置
VALIDATION_RULES = {
    'min_loan_amount': 1000,
    'max_loan_amount': 200000,
    'min_loan_term': 1,
    'max_loan_term': 60,
    'min_credit_score': 300,
    'max_credit_score': 900
}

# 模型配置
MODEL_CONFIG = {
    'credit_score': {
        'model_type': 'XGBRegressor',
        'objective': 'reg:squarederror'
    },
    'default_prob': {
        'model_type': 'XGBClassifier', 
        'objective': 'binary:logistic'
    },
    'recommendation': {
        'model_type': 'XGBRegressor',
        'objective': 'reg:squarederror'
    }
}