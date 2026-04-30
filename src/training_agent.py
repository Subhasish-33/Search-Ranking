import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib

DATA_DIR = "data"
MODEL_DIR = "models"
SEED = 42

def load_data(split_name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split_name}.csv"))
    # Group sizes: number of rows per query_id
    group = df.groupby('query_id').size().values
    
    # Features and labels
    feature_cols = ['bm25', 'tfidf', 'jaccard', 'cosine_sim', 'doc_len', 'pagerank_sim']
    X = df[feature_cols]
    y = df['label']
    
    return X, y, group

def objective(trial, X_train, y_train, group_train, X_val, y_val, group_val):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'label_gain': [0, 1], # For binary relevance
        'verbosity': -1,
        'seed': SEED
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
    
    # Train model
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    # Get the best NDCG score on validation set
    best_score = gbm.best_score['valid_0']['ndcg@10']
    return best_score

def train_model():
    print("Loading data...")
    X_train, y_train, group_train = load_data('train')
    X_val, y_val, group_val = load_data('val')
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    print("Starting hyperparameter tuning...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, group_train, X_val, y_val, group_val), n_trials=10)
    
    print("Best parameters:")
    print(study.best_params)
    
    # Train final model with best params
    best_params = study.best_params
    best_params['objective'] = 'lambdarank'
    best_params['metric'] = 'ndcg'
    best_params['ndcg_eval_at'] = [10]
    best_params['label_gain'] = [0, 1]
    best_params['verbosity'] = -1
    best_params['seed'] = SEED
    
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
    
    print("Training final model...")
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=True)]
    )
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'lambdamart_model.pkl')
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
