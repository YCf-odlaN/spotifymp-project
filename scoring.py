import os
import time

import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.sparse import load_npz

ARTIFACT_REPO = os.getenv("ARTIFACT_REPO", "odlanYCF/mpd-cooc-artifacts")  # private dataset repo
HF_TOKEN = os.getenv("HF_TOKEN")            # Space secret; can be None locally if artifacts/ exists
LOCAL_DIR = os.getenv("ARTIFACT_DIR", "artifacts")


def fetch(name: str) -> str:
    """Same code path everywhere: local folder if present, otherwise pull from the Hub."""
    local = os.path.join(LOCAL_DIR, name)
    if os.path.exists(local):
        return local
    return hf_hub_download(ARTIFACT_REPO, name, repo_type="dataset", token=HF_TOKEN)

t0 = time.time()
X = load_npz(fetch("X.npz")).tocsr()        # playlists x tracks
XT = X.T.tocsr()                            # tracks x playlists: row i = playlists containing track i
meta = pd.read_parquet(fetch("tracks.parquet"))
assert len(meta) == X.shape[1], "metadata rows must equal matrix columns (index alignment contract)"
meta["search_key"] = meta["search_key"].astype("string[pyarrow]")   # substring search in C++, not per-row Python
print(f"loaded X={X.shape} nnz={X.nnz:,} in {time.time() - t0:.1f}s")

#separating scoring and ranking
def score(seeds)-> np.ndarray:
    """Array-like of track indices -> length-T of co-occurence vector"""
    if seeds.size == 0:
        return np.zeros(X.shape[1], dtype = np.float32)
    assert (seeds >= 0).all() and (seeds < X.shape[1]).all(), "seed index out of range"
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    pids = XT[seeds].indices #all playlists for given seeds (T x P) for each seed T
    co = np.asarray(X[pids].sum(axis=0)).ravel() #we're summing down the rows; X is p x t (this is what X * XT is doing)
    co[seeds] = 0
    return co

def top_k(co , k) -> np.ndarray:
    """score the co-occurence vector -> k track indices, best first"""
    if k == 0:
        return np.array([], dtype=np.int64)
    k = min(int(k), int((co >0).sum()))
    top = np.argpartition (-co,k-1)[:k] #finds the top k without sorting, negate because arg partition does ascending only
    top = top[np.argsort(-co[top])] #negate again for ascending order
    return top 

       





    