import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import ndcg_score

DATA_DIR = "data"
MODEL_DIR = "models"
K_MAX = 10

def calculate_ndcg(df, score_col, k=10):
    ndcg_scores = []
    grouped = df.groupby('query_id')
    
    for q_id, group in grouped:
        if len(group) <= 1 or sum(group['label']) == 0:
            continue
            
        y_true = np.asarray([group['label'].values])
        y_score = np.asarray([group[score_col].values])
        
        try:
            score = ndcg_score(y_true, y_score, k=k)
            ndcg_scores.append(score)
        except ValueError:
            pass
            
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def evaluate_ndcg():
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    model_path = os.path.join(MODEL_DIR, 'lambdamart_model.pkl')
    model = joblib.load(model_path)
    
    feature_cols = ['bm25', 'tfidf', 'jaccard', 'cosine_sim', 'doc_len', 'pagerank_sim']
    test_df['lambdamart_score'] = model.predict(test_df[feature_cols])
    
    baseline_ndcg = calculate_ndcg(test_df, 'bm25', k=10)
    model_ndcg = calculate_ndcg(test_df, 'lambdamart_score', k=10)
    
    improvement = ((model_ndcg - baseline_ndcg) / baseline_ndcg) * 100 if baseline_ndcg > 0 else 0
    
    print("--- Final Table Results (NDCG@10) ---")
    print(f"BM25 Baseline: {baseline_ndcg:.4f}")
    print(f"LambdaMART:    {model_ndcg:.4f}")
    print(f"Improvement:   +{improvement:.2f}%")

if __name__ == "__main__":
    evaluate_ndcg()
