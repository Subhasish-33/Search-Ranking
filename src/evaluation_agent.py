import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
K_MAX = 10

def calculate_ndcg(df, score_col, k=10):
    ndcg_scores = []
    grouped = df.groupby('query_id')
    
    for q_id, group in grouped:
        if len(group) <= 1 or sum(group['label']) == 0:
            continue # Skip queries with no positive docs or only 1 doc
            
        y_true = np.asarray([group['label'].values])
        y_score = np.asarray([group[score_col].values])
        
        try:
            score = ndcg_score(y_true, y_score, k=k)
            ndcg_scores.append(score)
        except ValueError:
            pass
            
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def evaluate_model():
    print("Loading test data...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    print("Loading LambdaMART model...")
    model_path = os.path.join(MODEL_DIR, 'lambdamart_model.pkl')
    model = joblib.load(model_path)
    
    feature_cols = ['bm25', 'tfidf', 'jaccard', 'cosine_sim', 'doc_len', 'pagerank_sim']
    
    print("Predicting scores...")
    test_df['lambdamart_score'] = model.predict(test_df[feature_cols])
    
    # Calculate NDCG@10
    baseline_ndcg = calculate_ndcg(test_df, 'bm25', k=10)
    model_ndcg = calculate_ndcg(test_df, 'lambdamart_score', k=10)
    
    improvement = ((model_ndcg - baseline_ndcg) / baseline_ndcg) * 100 if baseline_ndcg > 0 else 0
    
    report = f"--- Evaluation Report ---\n"
    report += f"BM25 Baseline NDCG@10: {baseline_ndcg:.4f}\n"
    report += f"LambdaMART NDCG@10: {model_ndcg:.4f}\n"
    report += f"Improvement: {improvement:.2f}%\n"
    
    print(report)
    
    with open(os.path.join(RESULTS_DIR, 'eval_report.txt'), 'w') as f:
        f.write(report)
        
    # Plot NDCG@k Curve
    ks = list(range(1, K_MAX + 1))
    baseline_ndcgs = [calculate_ndcg(test_df, 'bm25', k=k) for k in ks]
    model_ndcgs = [calculate_ndcg(test_df, 'lambdamart_score', k=k) for k in ks]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ks, baseline_ndcgs, label='BM25 Baseline', marker='o', linestyle='--')
    plt.plot(ks, model_ndcgs, label='LambdaMART', marker='s')
    plt.title('NDCG@k Curve')
    plt.xlabel('k')
    plt.ylabel('NDCG')
    plt.xticks(ks)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, 'ndcg_curve.png'))
    plt.close()
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    importance = model.feature_importance(importance_type='gain')
    sns.barplot(x=importance, y=feature_cols)
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'))
    plt.close()
    
    print("Evaluation complete. Artifacts saved to results/")

if __name__ == "__main__":
    evaluate_model()
