import catboost as cb
import numpy as np
from sklearn.model_selection import train_test_split

def train(X, y, random_seed=None):
    """训练CatBoost基模型"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = cb.CatBoostClassifier(
        random_state=42 if random_seed is None else random_seed,
        iterations=100,
        depth=4,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    return pred_proba