"""
Spotify MPD - Co-Occurrence Recommender (Baseline App)

Model: For seeded tracks (Initial Playlist),  
"""
import os
import time

import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.sparse import load_npz

ARTIFACT_REPO = os.getenv("ARTIFACT_REPO", "odlanYCF/mpd-cooc-artifacts") #storing sparse matrix we create here
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_DIR = os.getenv("ARTIFACT_DIR", "artifacts")

def fetch (name: str) -> str:
    """Ensuring the code path is the same to access X (the trained matrix stored as an npz file).
       Will be referncing it locally or via hugging face artifact repo"""
    local = os.path.join(LOCAL_DIR, name)
    if os.path.exists(local):
        return local
    return hf_hub_download(ARTIFACT_REPO,name, repo_type="dataset", token=HF_TOKEN)

t0 = time.time() #return the time in UTC format (seconds since epoch)
X = load_npz(fetch("X.npz")).tocsr() #loading in the matrix
XT = X.T.tocsr()
meta = pd.read_parquet("tracks.parquet")
assert len(meta) == X.shape[1], "metadata rows must equal matrix columns (index alignment control)"
meta["search_key"] = meta["search_key"].astype("string[pyarrow]") #converting to pyarrow string data type to efficiently search
print(f"loaded X={X.shape} nnz={X.nnz:,} in {time.time() - t0:.1f}s")

def search (query: str):
    q = (query or "").strip().lower() #no white spaces and lowercase
    if len(q) < 2:
        return gr.Dropdown(choices=[])


def recommend()
with gr.Blocks(title = "Co-Occurrence Recommender") as demo:
    gr.Markdown(
        "# Spotify Million Playlist Dataset - Co-Occurrence Recommender \n"
        "Type a track name, search for it, and get recommended tracks to have alongside it in a playlist"
        "'co_occurrence' = number of playlists suggested track has appeared alongside seeded tracks; 'n_playlists' = number of playlists track has appeared in (out of a million)" 
    )

    query = gr.Textbox(label = "Search track/artist (press Enter and proceed to next dropdown)", placeholder="e.g Billie Jean")
    seed = gr.Dropdown(label = "Select the seed for tracks in your playlist", choices = [], interactive=True)
    k = gr.Slider(5, 50, value = 15, step= 1, label = "Number of track recommendations")
    table = gr.DataFrame(label= "Most co-occurring tracks", interactive=False)

    query.submit(search, inputs=query, outputs=seed)
    seed.change(recommend, inputs = [seed,k] , outputs=table)
    k.release(recommend, inputs= [seed, k], outputs=table)

if __name__ == "__main__":
    demo.launch()