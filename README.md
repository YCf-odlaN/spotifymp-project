# Spotify Playlists and Medallion Architecture

## Overview
This project implements a medallion architecture data pipeline over Spotify's Million Playlist Dataset. The motivation came from my workplace: I watched a data engineering team build and maintain these layers in Databricks and wanted to understand the pattern by implementing it end-to-end myself, starting with minimal tooling and adding complexity only where the data forced it. Bronze holds raw ingested data with no business logic applied, silver cleans and remodels it into properly grained tables, and gold produces the aggregations and features for downstream use. The gold layer is intended to feed a music recommendation system, which is the second half of this project.
 
### Note
Unlike at work, this runs locally and several architectural choices reflect that. The layers are directories on disk rather than managed tables in a metastore. There is no Unity Catalog- I'm not governing access, though lineage tracking could be useful here later. Spark runs in local mode on a single machine rather than against a cluster. The tables are plain parquet instead of delta, which means no ACID transactions, schema enforcement, or time travel; Delta is a reasonable upgrade if concurrent writes or versioning become relevant. Transformations run as notebook cells rather than orchestrated jobs for now... 

## Setup
- PySpark 4.1.1 (Bundles its own spark distribution and Hadoop 3.4.2 jars)
- JDK 17 Temurin - PySpark 4.x requires Java 17+
- Windows Only: winutils.exe + hadoop.dll from cdarlint/winutils (3.3.6 build - newest available, and is stable across 3.3.x/3.4.x)
  - place in C:\hadoop\bin
  - Set HADOOP_HOME =C:\hadoop, add C:\hadoop\bin to PATH
  - Copy hadoop.dll to C:\Windows\System32 

## Bronze Layer
Spotify Million Playlist dataset [Download at https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge]  is 5.5gb and consists of 1000 json files, containing 1 million playlist metadata and dataset generation metadata. This is stored in data/
 
2 tables seemed appropriate for bronze to correspond to the two objects "info" and "playlist" with key:value pairs in the JSON files. Pandas library along with path and json modules were enough to get started, but soon reached a bottleneck when it took 17 minutes to loop, read, and write the data into bronze/ as csv files. There was also no way to use RAM to do one write invocation without crashing. 
 
Parquet files were the next choice. It would be useful to compress file sizes and optimize query search time for dadta analysis. Looping  over data/ in the same manner proved fruitful as the writing took 7 minutes and nicely divided into sub-directories bronze/playlist/ and bronze/sliceinfo.
 
Apache Spark stood out as the compute engine, since it integrated nicely with python.
 
### Bronze Process
One table contains the slice info, totaling 1000 rows (1000 slices) in the following schema:

~~~
root
 |-- generated_on: string (nullable = true)
 |-- slice: string (nullable = true)
 |-- version: string (nullable = true)
~~~
 
The second table in bronze contains the actual playlists, totaling 1 million rows of playlist metadata. The tracks column of arrays will be adjusted in the silver layer:

~~~
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
~~~

## Silver Layer
Before settling on a relational model, a data quality pass (analysis.ipynb) was run to understand what the transformations actually needed to handle: unique key counts, consistency between distinct-id counts and expected totals, null handling, and flattening the nested tracks array. A few findings changed the design outright.


### Data Quality Findings

<details>
<summary>DQ Summary</summary>

Empty strings
```python
dfp.filter(F.col("description") == "").select(F.col("pid"), F.col("description"), F.when(F.col("description").isNotNull(), "NOT NULL").otherwise("NULL")).show()
```
Output:
|    pid | description   | CASE WHEN (description IS NOT NULL) THEN NOT NULL ELSE NULL END   |
|-------:|:--------------|:------------------------------------------------------------------|
| 620536 |               | NOT NULL                                                          |
| 101318 |               | NOT NULL                                                          |

Verdict: Defined emptystringconv() to handle casting NULL values to empty strings to true nulls
```python
def emptystringconv (df, columns = None):
    """
    This function will read string columns in a dataframe and convert any whitespace string values to proper null type.
    If no columns are passed, function will loop through all string columns
    """
    if columns is None:
        for x, y in df.dtypes:
            if y == "string":
                df = df.withColumn(x, F.when(F.trim(F.col(x)) == "", F.lit(None)).otherwise(F.col(x)))

    else:
        for x in columns:
            df = df.withColumn(x, F.when(F.col(x) == "", F.lit(None)).otherwise(F.col(x)))

    return df
```

Artists with same name
```python
flat.groupBy("artist_name") \
    .agg (F.countDistinct("artist_uri").alias ("distinct_uris")) \
    .filter (F.col ("distinct_uris") > 1) \
    .orderBy (F.col("distinct_uris").desc()) \
    .show()
```
Output:
| artist_name   |   distinct_uris |
|:--------------|----------------:|
| Ghost         |              12 |
| Kim           |              11 |
| Luke          |              10 |
| Ten           |              10 |
| Oliver        |              10 |
| Monty         |               9 |
| Gemini        |               9 |
| Luna          |               9 |
| Sasha         |               9 |
| Joseph        |               9 |

Verdict: Spot checked artist_uris on Spotify and confirmed to be legitamate artists. Giving way for "artist_uri" to be a primary key in the artist table.

Explicit vs Clean Albums
```python
flat.filter((F.col("artist_name") == "Drake") & ((F.col("album_name") == "Nothing Was The Same" ) | (F.col("album_name") == "More Life" ))) \
    .select("album_name", "artist_uri", "album_uri") \
    .distinct() \
    .orderBy("album_name")
```
Output:
| album_name           | artist_uri                            | album_uri                            |
|:---------------------|:--------------------------------------|:-------------------------------------|
| More Life            | spotify:artist:3TVXtAsR1Inumwj472S9r4 | spotify:album:7Ix0FS4f1lK42C3rix5rHg |
| More Life            | spotify:artist:3TVXtAsR1Inumwj472S9r4 | spotify:album:1lXY618HWkwYKJWBRYR4MK |
| Nothing Was The Same | spotify:artist:3TVXtAsR1Inumwj472S9r4 | spotify:album:2ZUFSbIkmFkGag000RWOpA |
| Nothing Was The Same | spotify:artist:3TVXtAsR1Inumwj472S9r4 | spotify:album:2gXTTQ713nCELgPOS0qWyt |

Verdict: Spotify maintains separate catalog entries (distinct album_uri's) for explicit vs clean versions of the same release.
</details>

### Schema Design
Anything carrying its own "*_uri" became its own table with that URI as primary key - track_uri, album_uri, artist_uri. Playlist is keyed on pid. This produced five tables: playlist, track, artist, album, pid_pos (bridge).

Type casting- collaborative from string to boolean. 

### Bridge Table Grain
('pid', 'track_uri') were the first candidates for a bridge key (to connect a track to its position in a playlist), but instances where a track repeats in a playlist are likely, breaking uniqueness on that pair. ('pid', 'pos') was used instead as a given position in a playlist maps to exactly one track. The resulting table holds 66 million rows.

### Tables
Playlist
~~~
root
 |-- pid: long (nullable = true)
 |-- name: string (nullable = true)
 |-- collaborative: boolean (nullable = true)
 |-- modified_at: long (nullable = true)
 |-- num_tracks: long (nullable = true)
 |-- num_albums: long (nullable = true)
 |-- num_followers: long (nullable = true)
 |-- num_edits: long (nullable = true)
 |-- num_artists: long (nullable = true)
 |-- duration_ms: long (nullable = true)
 |-- description: string (nullable = true)
 ~~~

Track
~~~
root
 |-- track_uri: string (nullable = true)
 |-- track_name: string (nullable = true)
 |-- artist_uri: string (nullable = true)
 |-- duration_ms: long (nullable = true)
~~~

Artist
~~~
root
 |-- artist_uri: string (nullable = true)
 |-- artist_name: string (nullable = true)
~~~

Album
~~~
root
 |-- album_uri: string (nullable = true)
 |-- album_name: string (nullable = true)
~~~

pos_bridge
~~~
root
 |-- pid: long (nullable = true)
 |-- pos: long (nullable = true)
 |-- track_uri: string (nullable = true)
~~~

## Next Steps
With silver complete, the next step is to select a recommendation model architecture before designing gold, since feature requirements for the model will drive aggregation and join logic that gold has to produce.
 
