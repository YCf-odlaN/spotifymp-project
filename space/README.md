---
title: MPD Co-occurrence Recommender
emoji: 🎧
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: "6.28.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# Spotify Million Playlist Dataset — co-occurrence recommender

Pick a seed track; the app returns the tracks that most often share a playlist with it
in the training split (980k playlists). Baseline model for the RecSys Challenge 2018 dataset.
Artifacts are pulled from a private dataset repo with a read-only token.