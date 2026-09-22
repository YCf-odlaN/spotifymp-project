"""
Spotify MPD — co-occurrence recommender (baseline serving app).

Model: for a seed track, find every playlist that contains it, then count how many of
those playlists each other track appears in. Rank by that count. No training step —
the "model" is the binary playlist x track matrix itself.

Artifacts (built in the notebook, see build step):
  X.npz          scipy CSR, shape (1_000_000, n_tracks), float32 ones, TRAIN split only
  tracks.parquet one row per column of X, in column order:
                 track_idx, track_uri, track_name, artist_name, n_playlists, search_key
"""
import os
import time

import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.sparse import load_npz

# ZeroGPU hardware (free tier for Gradio Spaces) requires at least one @spaces.GPU function.
# The recommender is CPU-only; this placeholder is never called. Guarded so local runs don't need `spaces`.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        pass
except ImportError:
    pass

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


def search(query: str):
    q = (query or "").strip().lower()
    if len(q) < 2:
        return gr.Dropdown(choices=[], value=None)
    hits = meta[meta["search_key"].str.contains(q, regex=False)].nlargest(15, "n_playlists")
    choices = [
        (f"{r.track_name} — {r.artist_name}  ({r.n_playlists:,} playlists)", int(r.track_idx))
        for r in hits.itertuples(index=False)
    ]
    return gr.Dropdown(choices=choices, value=None)


def recommend(track_idx, k):
    if track_idx is None:
        return pd.DataFrame()
    i = int(track_idx)
    pids = XT.indices[XT.indptr[i]:XT.indptr[i + 1]]      # playlists containing the seed: O(1) CSR row lookup
    if pids.size == 0:
        return pd.DataFrame({"note": ["seed track has no playlists in the training matrix"]})
    co = np.asarray(X[pids].sum(axis=0)).ravel()           # per-track count over just those playlists
    co[i] = 0                                              # never recommend the seed to itself
    k = min(int(k), int((co > 0).sum()))
    top = np.argpartition(-co, k - 1)[:k]
    top = top[np.argsort(-co[top])]
    out = meta.iloc[top][["track_name", "artist_name", "n_playlists"]].reset_index(drop=True)
    out.insert(0, "co_occurrence", co[top].astype(int))
    out.insert(1, "share_of_seed_playlists", np.round(co[top] / pids.size, 3))
    return out


with gr.Blocks(title="MPD co-occurrence recommender") as demo:
    gr.Markdown(
        "# Spotify MPD — co-occurrence recommender\n"
        "Search for a track, pick it, and get the tracks that most often share a playlist with it. "
        "`co_occurrence` = playlists containing both; `n_playlists` = the track's overall popularity."
    )
    query = gr.Textbox(label="Search track / artist (press Enter)", placeholder="e.g. bohemian rhapsody")
    seed = gr.Dropdown(label="Seed track", choices=[], interactive=True)
    k = gr.Slider(5, 50, value=20, step=5, label="Number of recommendations")
    table = gr.Dataframe(label="Most co-occurring tracks", interactive=False)

    query.submit(search, inputs=query, outputs=seed)
    seed.change(recommend, inputs=[seed, k], outputs=table)
    k.release(recommend, inputs=[seed, k], outputs=table)

if __name__ == "__main__":
    demo.launch()