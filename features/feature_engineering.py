import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_numeric_dtype

def build_credit_features(df: pd.DataFrame, encoder: Any = None, is_training: bool = True) -> (pd.DataFrame, Any):
    # 转换为数值类型并填充缺失值
    df['preDsr'] = pd.to_numeric(df['preDsr'], errors='coerce').fillna(0)
    df['postDsr'] = pd.to_numeric(df['postDsr'], errors='coerce').fillna(0)

    # 映射评分
    pl_score_mapping = {'A1': 9, 'A2': 8, 'A3': 7, 'A4': 6, 'A5': 5, 'A6': 4, 'A7': 3, 'A8': 2, 'A9': 1}
    cm_score_mapping = {'AA': 10, 'AB': 9, 'AC': 8, 'AD': 7, 'BA': 6, 'BB': 5, 'CC': 4, 'DD': 3, 'GG': 2, 'JJ': 1}
    
    df['pl_score_numeric'] = df['scorePl01'].map(pl_score_mapping).fillna(0)
    df['cm_score_numeric'] = df['scoreCm01'].map(cm_score_mapping).fillna(0)

    # 计算 DSR 变化比率
    df['dsr_change_ratio'] = (df['postDsr'] - df['preDsr']) / df['preDsr'].replace(0, 1e-6)
    
    # 转换为数值类型并填充缺失值
    df['salaryAmount'] = pd.to_numeric(df['salaryAmount'], errors='coerce').fillna(0)
    df['pre_loan_debt_amount'] = df['preDsr'] * df['salaryAmount']
    
    df['loanAmountApplied'] = pd.to_numeric(df['loanAmountApplied'], errors='coerce')
    df['totalTenorApplied'] = pd.to_numeric(df['totalTenorApplied'], errors='coerce')
    df['flatRateApplied'] = pd.to_numeric(df['flatRateApplied'], errors='coerce')

    df['loanAmountApproved'] = pd.to_numeric(df['loanAmountApproved'], errors='coerce')
    df['totalTenorApproved'] = pd.to_numeric(df['totalTenorApproved'], errors='coerce')
    df['flatRateApproved'] = pd.to_numeric(df['flatRateApproved'], errors='coerce')

    # 转换为日期格式
    df['loanDate'] = pd.to_datetime(df['loanDate'], errors='coerce')
    df['writeoffDate'] = pd.to_datetime(df['writeoffDate'], errors='coerce')
    df['ocaDate'] = pd.to_datetime(df['ocaDate'], errors='coerce')
    df['ivaDate'] = pd.to_datetime(df['ivaDate'], errors='coerce')

    # 确定违约日期
    default_codes = [3750, 3800, 4200, 3680]
    df['defaultDate'] = df[['writeoffDate', 'ocaDate', 'ivaDate']].min(axis=1, skipna=True)
    # 对于 3680 且无日期，设为 loanDate + 1 个月
    mask_3680_no_date = (df['stateCode'] == 3680) & (df['defaultDate'].isna())
    df.loc[mask_3680_no_date, 'defaultDate'] = df.loc[mask_3680_no_date, 'loanDate'] + pd.DateOffset(months=1)

    # 使用 stateCode 判断批核状态
    approved_codes = [3600, 3650, 3700]
    mask_completed = df['stateCode'].isin(approved_codes)
    df.loc[~mask_completed, ['loanAmountApplied', 'totalTenorApplied', 'flatRateApplied',
                           'loanAmountApproved', 'totalTenorApproved', 'flatRateApproved']] = None

    # 计算 debtToIncomeRatio
    df['debtToIncomeRatio'] = np.where(
        mask_completed,
        df['loanAmountApplied'] / (df['salaryAmount'] * 12).replace(0, 1e-6),
        None
    )

    # 分类变量
    categorical_cols = ['gender', 'propertyType', 'employmentType', 'industry']
    
    if is_training:
        # 处理缺失的分类列
        for col in categorical_cols:
            if col not in df.columns:
                df[col] = None
        
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical_cols].fillna('Unknown'))
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        
        # 训练时合并特征
        numeric_cols = ['age', 'workDurationYear', 'salaryAmount', 'preDsr', 'postDsr',
                        'pl_score_numeric', 'cm_score_numeric', 'dsr_change_ratio',
                        'pre_loan_debt_amount', 'debtToIncomeRatio']
        numeric_cols = [col for col in numeric_cols if col in df.columns and is_numeric_dtype(df[col])]
        features_df = pd.concat([df[numeric_cols].reset_index(drop=True), encoded_df], axis=1)

        # 修复 FutureWarning
        features_df = features_df.infer_objects(copy=False)

        return features_df, encoder
        
    else:
        # 处理预测时缺失的分类列
        for col in categorical_cols:
            if col not in df.columns:
                df[col] = None
        
        encoded_data = encoder.transform(df[categorical_cols].fillna('Unknown'))
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

        numeric_cols = ['age', 'workDurationYear', 'salaryAmount', 'preDsr', 'postDsr',
                        'pl_score_numeric', 'cm_score_numeric', 'dsr_change_ratio',
                        'pre_loan_debt_amount', 'debtToIncomeRatio']
        numeric_cols = [col for col in numeric_cols if col in df.columns and is_numeric_dtype(df[col])]
        features_df = pd.concat([df[numeric_cols].reset_index(drop=True), encoded_df], axis=1)

        # 修复 FutureWarning
        features_df = features_df.infer_objects(copy=False)

        if random_seed is not None:
            np.random.seed(random_seed)
            numeric_cols = [col for col in numeric_cols if np.random.random() > 0.2]

        return features_df, None