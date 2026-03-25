import json, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings
from app.models.base import SessionLocal
from app.models.entities import Journal, JournalProfile, QueryRun, Recommendation
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from pathlib import Path

# Define local model path
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "all-MiniLM-L6-v2"

# Lazy loaded models
bert_general = None
tfidf = None

def ensure_models_loaded():
    global bert_general, tfidf

    if bert_general is None:
        if MODEL_DIR.exists():
            bert_general = SentenceTransformer(str(MODEL_DIR))
            print(f"✓ Loaded BERT model from local directory: {MODEL_DIR}")
        else:
            print("Downloading model (first time only)...")
            bert_general = SentenceTransformer("all-MiniLM-L6-v2")
            MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            bert_general.save(str(MODEL_DIR))
            print(f"✓ Model saved to: {MODEL_DIR}")

    if tfidf is None:
        tfidf_local = TfidfVectorizer(max_features=20000, stop_words="english")
        _session = SessionLocal()
        try:
            corpus = []
            for j in _session.query(Journal).all():
                text = j.name + " " + (j.publisher or "")
                corpus.append(text)

            if corpus:
                tfidf_local.fit(corpus)
            else:
                tfidf_local.fit(["machine learning", "data science", "computer science"])
        except Exception as e:
            print(f"TFIDF fallback: {e}")
            tfidf_local.fit(["machine learning", "data science", "computer science"])
        finally:
            _session.close()

        tfidf = tfidf_local


# CACHE
_journal_cache = None
_cache_timestamp = None
CACHE_TTL = 3600

def get_journal_cache():
    global _journal_cache, _cache_timestamp

    current_time = time.time()
    if _journal_cache is None or (current_time - _cache_timestamp) > CACHE_TTL:
        print("Loading journal cache...")
        _session = SessionLocal()
        try:
            journals = _session.query(Journal).join(JournalProfile).all()

            _journal_cache = {
                'journals': [],
                'tfidf_vectors': [],
                'bert_vectors': [],
                'names': [],
                'metadata': []
            }

            for j in journals:
                p = j.profile
                if not p or not p.tfidf_vector or not p.bert_vector:
                    continue

                try:
                    v_tfidf = np.array(json.loads(p.tfidf_vector))
                    v_bert = np.array(json.loads(p.bert_vector))

                    _journal_cache['journals'].append(j)
                    _journal_cache['tfidf_vectors'].append(v_tfidf)
                    _journal_cache['bert_vectors'].append(v_bert)
                    _journal_cache['names'].append(j.name)
                    _journal_cache['metadata'].append({
                        'id': j.id,
                        'name': j.name,
                        'display_name': j.display_name,
                        'publisher': j.publisher,
                        'impact_factor': j.impact_factor,
                        'is_open_access': j.is_open_access,
                        'issn': j.issn,
                        'eissn': j.eissn,
                        'subjects': json.loads(j.subjects) if j.subjects else []
                    })
                except:
                    continue

            _journal_cache['tfidf_matrix'] = np.array(_journal_cache['tfidf_vectors'])
            _journal_cache['bert_matrix'] = np.array(_journal_cache['bert_vectors'])

            _cache_timestamp = current_time
            print(f"✓ Loaded {len(_journal_cache['journals'])} journals into cache")
        finally:
            _session.close()

    return _journal_cache


def extract_keywords(text: str, top_n: int = 10):
    ensure_models_loaded()
    vec_tfidf_sparse = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()
    vec_tfidf = vec_tfidf_sparse.toarray()[0]

    top_indices = vec_tfidf.argsort()[-top_n:][::-1]
    keywords = [feature_names[i] for i in top_indices if vec_tfidf[i] > 0]
    return keywords


def rank_journals(abstract: str, top_k: int = settings.TOP_K):
    ensure_models_loaded()
    db = SessionLocal()

    cache = get_journal_cache()
    if not cache or len(cache['journals']) == 0:
        db.close()
        return []

    abstract_keywords = extract_keywords(abstract)

    vec_tfidf_sparse = tfidf.transform([abstract])
    vec_tfidf = np.array(vec_tfidf_sparse.todense()).flatten()
    vec_bert_general = bert_general.encode([abstract])[0]

    vec_tfidf_norm = vec_tfidf / (np.linalg.norm(vec_tfidf) + 1e-8)
    vec_bert_norm = vec_bert_general / (np.linalg.norm(vec_bert_general) + 1e-8)

    tfidf_matrix_norm = cache['tfidf_matrix'] / (
        np.linalg.norm(cache['tfidf_matrix'], axis=1, keepdims=True) + 1e-8)
    bert_matrix_norm = cache['bert_matrix'] / (
        np.linalg.norm(cache['bert_matrix'], axis=1, keepdims=True) + 1e-8)

    sim_tfidf_all = np.dot(tfidf_matrix_norm, vec_tfidf_norm)
    sim_bert_all = np.dot(bert_matrix_norm, vec_bert_norm)

    results = []
    for idx, (j, metadata) in enumerate(zip(cache['journals'], cache['metadata'])):
        sim_combined = (0.6 * sim_bert_all[idx]) + (0.4 * sim_tfidf_all[idx])
        results.append({
            "journal_name": metadata['name'],
            "display_name": metadata['display_name'] or metadata['name'],
            "similarity": float(sim_combined)
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    db.close()
    return results[:top_k]


def analyze_text_distribution(abstract: str):
    ensure_models_loaded()
    import re
    from collections import Counter
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = re.findall(r'\b[a-zA-Z]{3,}\b', abstract.lower())
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    word_freq = Counter(words)

    return {
        "word_frequency": dict(word_freq.most_common(20)),
        "total_words": len(words),
        "unique_words": len(set(words))
    }
