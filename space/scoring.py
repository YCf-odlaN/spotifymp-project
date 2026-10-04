import os
import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.sparse import load_npz


def fetch(name: str) -> str:
    """Same code path everywhere: local folder if present, otherwise pull from the Hub."""
    local = os.path.join(LOCAL_DIR, name)
    if os.path.exists(local):
        return local
    return hf_hub_download(ARTIFACT_REPO, name, repo_type="dataset", token=HF_TOKEN)

#separating scoring and ranking, parameter will be track indices
def score(seeds, normalize=None)-> np.ndarray: 
    """Array-like of track indices -> length-T of co-occurence vector"""
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    if seeds.size == 0:
        return np.zeros(X.shape[1], dtype = np.float32)
    assert (seeds >= 0).all() and (seeds < X.shape[1]).all(), "seed index out of range"
    if normalize == "cosine_seed":
        co = np.zeros(X.shape[1], dtype=np.float32)
        for s in seeds:
            s_pids = XT.indices[XT.indptr[s]:XT.indptr[s+1]]
            if s_pids.size == 0:
                continue
            co += np.asarray(X[s_pids].sum(axis=0)).ravel() / np.sqrt(s_pids.size)
    else:
        pids = XT[seeds].indices #all playlist ids for given seeds (T x P) 
        co = np.asarray(X[pids].sum(axis=0)).ravel() #we're summing down the rows; X is p x t (this is what X * XT is doing)
    if normalize is not None:
        co = co / np.sqrt(np.maximum(n_playlists, 1))
    co[seeds] = 0 #seeded tracks are zeroed out, such that the same track is not recommended 
    return co


def top_k(co , k) -> np.ndarray:
    """score the co-occurence vector -> k track indices, best first"""
    if k == 0:
        return np.array([], dtype=np.int64)
    k = min(int(k), int((co >0).sum()))
    top = np.argpartition (-co,k-1)[:k] #finds the top k without sorting, negate because arg partition does ascending only
    top = top[np.argsort(-co[top])] #negate again for ascending order 
    return top 

ARTIFACT_REPO = os.getenv("ARTIFACT_REPO", "odlanYCF/mpd-cooc-artifacts")  # private dataset repo
HF_TOKEN = os.getenv("HF_TOKEN")            # Space secret; can be None locally if artifacts/ exists
LOCAL_DIR = os.getenv("ARTIFACT_DIR", "artifacts")

X = load_npz(fetch("X.npz")).tocsr()        # playlists x tracks
XT = X.T.tocsr()                            # tracks x playlists: row i = playlists containing track i
meta = pd.read_parquet(fetch("tracks.parquet"))
n_playlists = meta["n_playlists"].to_numpy().astype(np.float32)
assert len(meta) == X.shape[1], "metadata rows must equal matrix columns (index alignment contract)"





    