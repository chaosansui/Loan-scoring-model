import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib
import os
from features.feature_engineering import build_credit_features
from data.data_processor import process_api_data
from config import (
    CREDIT_SCORE_MODEL_PATH,
    DEFAULT_PROB_MODEL_PATH,  
    FEATURE_COLUMNS_PATH,
    ENCODER_PATH,
    AMOUNT_RECOMMENDER_PATH,
    TERM_RECOMMENDER_PATH
)

# 初始化 Flask
app = Flask(__name__)

# 业务规则引擎类
class BusinessRuleEngine:
    def __init__(self):
        self.risk_rules = {
            'high': {'max_amount_multiplier': 0.3, 'max_term': 6},
            'medium': {'max_amount_multiplier': 0.7, 'max_term': 18},
            'low': {'max_amount_multiplier': 1.0, 'max_term': 36}
        }
    
    def apply_rules(self, credit_score, default_prob, raw_amount, raw_term):
        risk_level = self._determine_risk_level(credit_score, default_prob)
        rules = self.risk_rules[risk_level]
        
        final_amount = raw_amount * rules['max_amount_multiplier']
        final_term = min(raw_term, rules['max_term'])
        
        return final_amount, final_term, risk_level
    
    def _determine_risk_level(self, credit_score, default_prob):
        if default_prob > 0.6 or credit_score < 500:
            return 'high'
        elif default_prob > 0.3 or credit_score < 650:
            return 'medium'
        else:
            return 'low'

class GPUModelWrapper:
    """GPU模型包装器，用于指定显卡运行"""
    def __init__(self, model, device_id=0):
        self.model = model
        self.device_id = device_id
    
    def predict(self, X, **kwargs):
        """预测方法，指定GPU设备"""
        if hasattr(self.model, 'set_param'):
            self.model.set_param({'device': f'cuda:{self.device_id}'})
        return self.model.predict(X, **kwargs)
    
    def predict_proba(self, X, **kwargs):
        """预测概率方法，指定GPU设备"""
        if hasattr(self.model, 'set_param'):
            self.model.set_param({'device': f'cuda:{self.device_id}'})
        return self.model.predict_proba(X, **kwargs)

# 全局加载模型和特征工程配置
try:
    # 加载原始模型
    credit_model_raw = joblib.load(CREDIT_SCORE_MODEL_PATH)
    default_model_raw = joblib.load(DEFAULT_PROB_MODEL_PATH)
    amount_model_raw = joblib.load(AMOUNT_RECOMMENDER_PATH)
    term_model_raw = joblib.load(TERM_RECOMMENDER_PATH)
    
    # 使用GPU包装器包装模型，指定不同的显卡
    # 可以根据你的GPU数量分配不同的设备ID
    credit_model = GPUModelWrapper(credit_model_raw, device_id=3)
    default_model = GPUModelWrapper(default_model_raw, device_id=3)  
    amount_model = GPUModelWrapper(amount_model_raw, device_id=3) 
    term_model = GPUModelWrapper(term_model_raw, device_id=3)  
    
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    encoder = joblib.load(ENCODER_PATH)
    
    # 初始化业务规则引擎
    rule_engine = BusinessRuleEngine()
    
    print("所有模型、配置和规则引擎加载成功，API 已准备就绪。")
    print("模型GPU分配: 信用和违约模型 → GPU 0, 推荐模型 → GPU 1")
    
except Exception as e:
    print(f"模型或配置加载失败: {e}")
    # 尝试降级到CPU模式
    try:
        print("尝试降级到CPU模式...")
        credit_model = joblib.load(CREDIT_SCORE_MODEL_PATH)
        default_model = joblib.load(DEFAULT_PROB_MODEL_PATH)
        amount_model = joblib.load(AMOUNT_RECOMMENDER_PATH)
        term_model = joblib.load(TERM_RECOMMENDER_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        encoder = joblib.load(ENCODER_PATH)
        rule_engine = BusinessRuleEngine()
        print("降级到CPU模式成功")
    except Exception as fallback_error:
        print(f"CPU模式也失败: {fallback_error}")
        credit_model = None
        default_model = None
        amount_model = None
        term_model = None
        feature_columns = None
        encoder = None
        rule_engine = None

def preprocess_for_prediction(raw_data, encoder, feature_columns):
    """预处理新数据用于预测"""
    try:
        # 处理API数据
        raw_df = process_api_data(raw_data)
        if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
            return None, "数据处理失败"
        
        # 构建特征
        features_df, _ = build_credit_features(raw_df, encoder=encoder, is_training=False)
        if features_df.empty:
            return None, "特征工程失败"
        
        # 确保特征列一致
        features_df = features_df.reindex(columns=feature_columns, fill_value=0)
        features_df = features_df.fillna(0)
        
        return features_df, None
        
    except Exception as e:
        return None, f"预处理错误: {str(e)}"

def check_gpu_availability():
    """检查GPU可用性"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            # 检查CUDA是否可用
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    gpu_available = check_gpu_availability()
    if all([credit_model, default_model, amount_model, term_model]):
        status = {
            "status": "healthy", 
            "message": "所有模型加载正常",
            "gpu_available": gpu_available,
            "gpu_mode": isinstance(credit_model, GPUModelWrapper) if credit_model else False
        }
        return jsonify(status), 200
    else:
        return jsonify({"status": "unhealthy", "message": "部分模型未加载", "gpu_available": gpu_available}), 500

@app.route('/predict_score', methods=['POST'])
def predict_score_api():
    """串联决策流预测API"""
    if not request.json:
        return jsonify({"error": "请求体必须是 JSON 格式"}), 400
    
    # 检查模型是否加载
    if not all([credit_model, default_model, amount_model, term_model, rule_engine]):
        return jsonify({"error": "模型未加载完成，请检查训练状态"}), 503
    
    customer_data = request.json
    
    # 验证申请计划数据
    applied_plan = customer_data.get('appliedPlan', {})
    loan_amount = applied_plan.get('loanAmount')
    loan_term = applied_plan.get('totalTenor')
    
    if loan_amount is not None and (not isinstance(loan_amount, (int, float)) or loan_amount <= 0):
        return jsonify({"error": "loanAmount 必须是正数"}), 400
    
    if loan_term is not None and (not isinstance(loan_term, int) or loan_term <= 0):
        return jsonify({"error": "totalTenor 必须是正整数"}), 400

    try:
        # 1. 数据预处理
        features_df, error_msg = preprocess_for_prediction(customer_data, encoder, feature_columns)
        if error_msg:
            return jsonify({"error": error_msg}), 400
        
        print(f"预处理后的特征形状: {features_df.shape}")
        
        # 2. 串联预测流程
        # 第一步：信用评分预测
        credit_score = credit_model.predict(features_df)[0]
        credit_score = max(300, min(900, credit_score))  # 限制在300-900范围内
        
        # 第二步：违约概率预测
        default_prob = default_model.predict_proba(features_df)[:, 1][0]
        default_prob = max(0.0, min(1.0, default_prob))  # 限制在0-1范围内
        
        # 第三步：为推荐模型准备增强特征
        features_enhanced = features_df.copy()
        features_enhanced['credit_score'] = credit_score
        features_enhanced['default_prob'] = default_prob
        
        # 第四步：基础推荐预测
        raw_amount = amount_model.predict(features_enhanced)[0]
        raw_term = term_model.predict(features_enhanced)[0]
        
        # 确保基础值合理
        raw_amount = max(10000, min(200000, raw_amount))  # 限制在10000-200000范围内
        raw_term = max(1, min(40, raw_term))  # 限制在1-40个月内

        # 第五步：应用业务规则
        final_amount, final_term, risk_level = rule_engine.apply_rules(
            credit_score, default_prob, raw_amount, raw_term
        )
        
        # 最终格式化
        final_amount = round(final_amount)
        final_term = round(final_term)
        
        # 构建响应
        response = {
            "Reputation_score": float(round(credit_score, 2)),
            "Probability_of_default": float(round(default_prob, 4)),
            "Recommended_loan_amount": f"{final_amount}",
            "Recommended_issue_number": f"{final_term} M",
            "Risk_level": risk_level,
            "Message": "Success"
        }
        
        print(f"预测完成: 信用分={credit_score:.2f}, 违约概率={default_prob:.4f}, 风险等级={risk_level}")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"预测过程中发生错误: {str(e)}")
        return jsonify({"error": f"预测失败: {str(e)}"}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息端点"""
    model_info = {
        "credit_score_model_loaded": credit_model is not None,
        "default_prob_model_loaded": default_model is not None,
        "amount_model_loaded": amount_model is not None,
        "term_model_loaded": term_model is not None,
        "feature_columns_count": len(feature_columns) if feature_columns else 0,
        "encoder_loaded": encoder is not None,
        "gpu_available": check_gpu_availability(),
        "using_gpu": isinstance(credit_model, GPUModelWrapper) if credit_model else False
    }
    return jsonify(model_info), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)