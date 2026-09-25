import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

t0 = time.time()
X = load_npz(fetch("X.npz")).tocsr()        # playlists x tracks
XT = X.T.tocsr()                            # tracks x playlists: row i = playlists containing track i
meta = pd.read_parquet(fetch("tracks.parquet"))
assert len(meta) == X.shape[1], "metadata rows must equal matrix columns (index alignment contract)"
meta["search_key"] = meta["search_key"].astype("string[pyarrow]")   # substring search in C++, not per-row Python
print(f"loaded X={X.shape} nnz={X.nnz:,} in {time.time() - t0:.1f}s")

def m_recommend (track_indices: list, k):
    if track_indices is None:
        return pd.DataFrame()
    seeds = np.asarray(track_indices, dtype=np.int64) # ensuring we receive array from user input
    np.unique(seeds) # avoid feeding same seed multiple times
    pids = XT.indices[]

    