import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Constants
DATA_DIR = "data"
SAMPLE_SIZE = 1500  # Number of queries to process
SEED = 42

def jaccard_similarity(query_tokens, doc_tokens):
    intersection = len(set(query_tokens).intersection(doc_tokens))
    union = len(set(query_tokens).union(doc_tokens))
    return intersection / union if union > 0 else 0

def generate_features():
    print("Loading MS MARCO dataset (v1.1) subset...")
    # ms_marco v1.1 has a 'passages' column which is a dict of lists
    dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
    
    queries = []
    passages_list = []
    labels = []
    query_ids = []
    
    print(f"Collecting {SAMPLE_SIZE} queries...")
    count = 0
    for item in dataset:
        if count >= SAMPLE_SIZE:
            break
        
        q_id = item['query_id']
        query_text = item['query']
        passages = item['passages']
        
        # passages is a dict: {'is_selected': [0, 1, ...], 'passage_text': ['...', '...'], 'url': [...]}
        is_selected = passages['is_selected']
        passage_texts = passages['passage_text']
        
        # Ensure there's at least one positive document
        if 1 in is_selected:
            # Add all passages for this query
            for label, p_text in zip(is_selected, passage_texts):
                query_ids.append(q_id)
                queries.append(query_text)
                passages_list.append(p_text)
                labels.append(label)
            count += 1
            
    print(f"Collected {len(query_ids)} query-document pairs across {count} queries.")
    
    # Initialize feature extractors
    print("Initializing models...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    tfidf_vectorizer = TfidfVectorizer()
    
    # We need to compute features per query group
    df = pd.DataFrame({
        'query_id': query_ids,
        'query': queries,
        'passage': passages_list,
        'label': labels
    })
    
    features = []
    
    print("Extracting features per query group...")
    grouped = df.groupby('query_id')
    
    for q_id, group in tqdm(grouped, total=len(grouped)):
        q_text = group['query'].iloc[0]
        docs = group['passage'].tolist()
        
        # Lexical: BM25
        tokenized_docs = [doc.lower().split() for doc in docs]
        tokenized_query = q_text.lower().split()
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Lexical: TF-IDF cosine similarity
        tfidf_matrix = tfidf_vectorizer.fit_transform([q_text] + docs)
        tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Lexical: Jaccard
        jaccard_scores = [jaccard_similarity(tokenized_query, doc_tokens) for doc_tokens in tokenized_docs]
        
        # Semantic: Sentence Transformers Cosine Similarity
        q_emb = sentence_model.encode([q_text])
        d_embs = sentence_model.encode(docs)
        sem_scores = cosine_similarity(q_emb, d_embs).flatten()
        
        # Static: Document Length
        doc_lengths = [len(tokens) for tokens in tokenized_docs]
        
        # Static: Simulated PageRank (Proxy based on document length and term frequency of 'the' as a dummy graph centrality proxy)
        # In a real scenario, this would be pre-computed from a web graph
        simulated_pagerank = [np.log1p(dl) * (1.0 + np.random.rand() * 0.1) for dl in doc_lengths]
        
        for i in range(len(docs)):
            features.append({
                'query_id': q_id,
                'bm25': bm25_scores[i],
                'tfidf': tfidf_scores[i],
                'jaccard': jaccard_scores[i],
                'cosine_sim': sem_scores[i],
                'doc_len': doc_lengths[i],
                'pagerank_sim': simulated_pagerank[i],
                'label': group['label'].iloc[i]
            })

    features_df = pd.DataFrame(features)
    
    # Sort by query_id so grouping is contiguous (required by LightGBM)
    features_df = features_df.sort_values('query_id').reset_index(drop=True)
    
    print("Splitting data into Train/Val/Test (70/15/15)...")
    unique_qids = features_df['query_id'].unique()
    np.random.seed(SEED)
    np.random.shuffle(unique_qids)
    
    n_queries = len(unique_qids)
    train_end = int(n_queries * 0.7)
    val_end = int(n_queries * 0.85)
    
    train_qids = unique_qids[:train_end]
    val_qids = unique_qids[train_end:val_end]
    test_qids = unique_qids[val_end:]
    
    train_df = features_df[features_df['query_id'].isin(train_qids)]
    val_df = features_df[features_df['query_id'].isin(val_qids)]
    test_df = features_df[features_df['query_id'].isin(test_qids)]
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)
    
    # Save test queries and docs for qualitative evaluation later if needed
    test_raw = df[df['query_id'].isin(test_qids)]
    test_raw.to_csv(os.path.join(DATA_DIR, 'test_raw.csv'), index=False)

    print(f"Saved: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})")
    
if __name__ == "__main__":
    generate_features()
