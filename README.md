# Spotify Playlists and Medallion Architecture
The purpose of this project is to use Spotfiy's Million Playlist dataset to implement
industry standard medallion architecture. We looked to implement that logic with simple libraries and upgrading as needed and described.

## Status
Bronze layer ingestion complete. Spotify Million Playlist dataset (Download at ____ ) is 5.5gb and consists of 1000 json files, containing 1 million playlist metadata- as well as download source metadata.

We began by attempting to implement the medallion architecture framework with minimal libraries, and expanding as needed. In order to keep the data as close as can be to its raw source, we thought to have the bronze layer be 2 tables, flattening any JSON. 

One table contains the slice info, totaling 1000 rows (1000 slices) in the following format. These slices contain metadata related to the generation of the data set itself, including the generation date, slice #, and version:
<img width="2197" height="516" alt="image" src="https://github.com/user-attachments/assets/b9185b52-02b8-4e49-8db9-8c0ec9c3a43b" />

The second table in bronze contains the actual playlists, totaling 1 million rows of playlist metadata. The tracks column of arrays will be adjusted in the silver layer:
<img width="2193" height="521" alt="image" src="https://github.com/user-attachments/assets/9e9d1e93-0b92-4be7-8cd2-ba65c36b1862" />

We attempted to start the bronze layer with CSV files, however, it was soon clear we would need a better way of handling the data, as writing 5gb took a while:

Two attempts were made with loops -one focusing on using RAM to handle most of the processing and the other was to write dircetly to the drive on every loop iteration. 

In order to call "to_csv" once, converting data to dataframes and appending to a list would avoid the time costs of writing to the drive. 5.5gbs proved to be too much to naively process on RAM and often led to crashes:
<img width="2202" height="389" alt="image" src="https://github.com/user-attachments/assets/5cbd2c0e-1b00-4452-9654-3346d4338e21" />


Writing to drive on every iteration was successful, however, though the logic was O(n), the time costs of writing led to the 17 minute run time:
<img width="2198" height="293" alt="image" src="https://github.com/user-attachments/assets/a764a35a-8cb0-4b36-9675-490f2ec1a04e" />


Dask and Polar libraries can be used to chunk dataframes for processing, which may be considered for analysis purposes, but for this project, working with parquet files would be more efficient for handling and querying the data. Processing the files and converting to parquet was more than 50% faster compared to the CSV conversion. 
<img width="2160" height="308" alt="image" src="https://github.com/user-attachments/assets/5b67d7dc-1214-4d53-b00c-9c231d12a20c" />


For now, they will exist inside the bronze directory itself, but will likely need to be put in a subdirectory when scaling to other sources. All parquet files are named one-to-one to their JSON counterpart and the column schema is as intended.

Silver layer in progress...
