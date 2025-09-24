import numpy as np

def score_mapping(probability: float, score_a: float, score_b: float) -> float:
    """
    将违约概率转换为信用评分。
    公式：score = A - B * log(p / (1 - p))
    A: 基准分，当概率为50%时的分数。
    B: 尺度因子，控制分数随概率变化的速度。
    """
    if probability <= 0 or probability >= 1:
        return score_a # 处理边界情况，给一个默认分数
    
    odds = probability / (1 - probability)
    score = score_a - score_b * np.log(odds)
    return score