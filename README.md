# Spotify Playlists and Medallion Architecture

## Overview
This project implements a medallion architecture data pipeline over Spotify's Million Playlist Dataset. The motivation came from work: I watched a data engineering team build and maintain these layers in Databricks and wanted to understand the pattern by implementing it end-to-end myself, starting with minimal tooling and adding complexity only where the data forced it. Bronze holds raw ingested data with no business logic applied, silver cleans and remodels it into properly grained tables, and gold produces the aggregations and features for downstream use. The gold layer is intended to feed a music recommendation system, which is the second half of this project.
 
### Note
Unlike at work, this runs locally and several architectural choices reflect that. The layers are directories on disk rather than managed tables in a metastore, so I reference data by path instead of by table name. There is no Unity Catalog- I'm not governing access, though lineage tracking could be useful here later. Spark runs in local mode on a single machine rather than against a cluster. The tables are plain parquet instead of delta, which means no ACID transactions, schema enforcement, or time travel; Delta is a reasonable upgrade path if concurrent writes or versioning become relevant. Transformations run as notebook cells rather than orchestrated jobs for now. 

## Setup
- PySpark 4.1.1 (Bundles its own spark distribution and Hadoop 3.4.2 jars)
- JDK 17 Temurin - PySpark 4.x requires Java 17+
- Windows Only: winutils.exe + hadoop.dll from cdarlint/winutils (3.3.6 build - newest available, and is stable across 3.3.x/3.3.x)
  - place in C:\hadoop\bin
  - Set HADOOP_HOME =C:\hadoop, add C:\hadoop\bin to PATH
  - Copy hadoop.dll to C:\Windows\System32 

## Status
Bronze layer ingestion complete. Spotify Million Playlist dataset (Download at https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge ) is 5.5gb and consists of 1000 json files, containing 1 million playlist metadata- as well as download source metadata. This is stored in data/
 
2 tables seemed appropriate for bronze to correspond to the two objects "info" and "playlist" with key:value pairs in JSON format. Pandas library along with path and json modules were enough to get started, but soon reached a bottleneck when it took 17 minutes to loop, read, and write the data into bronze/ as csv files. There was also no way to use RAM to do one write invocation without crashing. 
 
Parquet files was the next choice, as it would be useful to compress file sizes and optimize query search time for analysis. Looping  over data/ in the same manner proved fruitful as the writing took 7 minutes and nicely divided into sub-directories bronze/playlist/ and bronze/sliceinfo.
 
Apache Spark stood out as the compute engine, since it integrated nicely with python.
 
## Process
One table contains the slice info, totaling 1000 rows (1000 slices) in the following schema:

'''
root
 |-- generated_on: string (nullable = true)
 |-- slice: string (nullable = true)
 |-- version: string (nullable = true)
 '''
 
The second table in bronze contains the actual playlists, totaling 1 million rows of playlist metadata. The tracks column of arrays will be adjusted in the silver layer:

'''
root
 |-- name: string (nullable = true)
 |-- collaborative: string (nullable = true)
 |-- pid: long (nullable = true)
 |-- modified_at: long (nullable = true)
 |-- num_tracks: long (nullable = true)
 |-- num_albums: long (nullable = true)
 |-- num_followers: long (nullable = true)
 |-- tracks: array (nullable = true)
 |    |-- element: struct (containsNull = true)
 |    |    |-- album_name: string (nullable = true)
 |    |    |-- album_uri: string (nullable = true)
 |    |    |-- artist_name: string (nullable = true)
 |    |    |-- artist_uri: string (nullable = true)
 |    |    |-- duration_ms: long (nullable = true)
 |    |    |-- pos: long (nullable = true)
 |    |    |-- track_name: string (nullable = true)
 |    |    |-- track_uri: string (nullable = true)
 |-- num_edits: long (nullable = true)
 |-- duration_ms: long (nullable = true)
 |-- num_artists: long (nullable = true)
 |-- description: string (nullable = true)
'''

Silver layer in progress...
 
