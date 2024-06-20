# Preprocessing

## Description

Preprocessing Class is designed to preprocess the data scrapped with `Promcsv`. Setting the tools needed for the development of a preprocessing pipeline to be used with the metrics collected.

The preprocessing tackles cleaning nulls,duplicates and rearranging the data while ensuring no information is lost in the process, it also ensures the existence of data when pipelines stop working, meaning a case of 0 QoS. Structure consistency is also enforced, keeping data-types and metrics consistent across time, allowing us to store preprocessed data persistently and then use it for training models.

Further use of DL tools such as AI4DL is being considered for preprocessing the data, as it allows us to efficiently encode the timeseries data into phases, thus performing feature engineering with our telemetry variables.

### Functionalities

* Take all the csv files from a given directory path, raises _Error_ if directory is empty or contains no CSV files.
* Checks the internal structure of the files, to ensure a correct merging. We have 4 options of the structure:
    1. We have all the metrics (_pipeline + node_).
    2. We are missing the _pipeline metrics_.
    3. We are missing the _node/pipeline server metrics_.
        1. We are missing the _node metrics_.
        2. We are missing the _pipeline server metrics_.
        3. We are missing both _node_ and _pipeline server metrics_.
    4. We are missing both _pipeline_ and _node/pipeline server metrics_.
  * For cases 1 and 2, we can still preprocess the data. For cases 3 and 4, we cannot, raises a _ValueError_.
* Preprocess data:
  * Ensure no Null values.
  * Ensure no duplicates.
  * Ensure correct merging of data by timestamp and cluster.
  * Show when a pipeline is dead (`pipeline_id=NaN`) by setting all the pipeline metrics to 0 when so (_QoS=0_).
  * Treat Null values on feature metrics by using linear imputation.
    * Linear imputation is specially suited for timeseries, it takes previous and future values and linearly imputes new values. Only performed in numerical feature metrics. Imputation is done grouped by cluster, to ensure statistically sound metrics are imputed.
* Stores the preprocessed data to a given archive path as PV.
* Deletes the files used for preprocessing.

We also defined some preconditions for our use-case:

* Only **one pipeline** running per cluster.
* We expect that Prometheus return at least one entry per each timestamp within the requested time range, be it through the Node performance metrics or the Pipeline server resource metrics.

## Installation

Please use the provided Dockerfile to locally compile the application and test it.

## Usage

Not defined yet. Only the definition of the class done.

Configuration will be done through a config json file. As an example:

```json
{
    "preprocessing":{
        "directory":"/dummy_directory",
        "archive":"/dummy_archive",
        "data_structure":{
            "key_values":[
                ["timestamp","datetime64[s]"],
                ["cluster","object"]
            ],
            "pipeline_values":[
                ["pipeline_id","object"],
                ["pipelines_status_avg_fps","float64"],
                ["pipelines_status_avg_pipeline_latency","float64"],
                ["pipelines_status_count_pipeline_latency","float64"],
                ["pipelines_status_elapsed_time","float64"],
                ["pipelines_status_frame_count","float64"],
                ["pipelines_status_realtime_pipeline_latency","float64"],
                ["pipelines_status_start_time","datetime64[s]"],
                ["pipelines_status_sum_pipeline_latency","float64"],
                ["pipelines_status_idelta_fps","float64"]
            ],
            "node_values":[
                ["node_cpu_usage","float64"],
                ["node_mem_usage","float64"]
            ],
            "pipeline_telemetry":[
                ["pipelines_server_cpu_usage","float64"],
                ["pipelines_server_mem_usage","float64"]
            ]
        },
        "pipeline":"pipeline_id",
        "cluster":"cluster"

    }
}
```

* `directory`: the path where the data to be preprocessed is stored. In our case, where `promcsv` data will be stored.
* `archive`: path where the data will be stored after preprocessing, acting as a persitent volume.
* `data_structure`: defines the structure that the data _to be preprocessed_ must follow.
  * `key_values`: the metrics used as indexs.
  * `pipeline_values`: the pipeline metrics.
  * `node_values`: node usage metrics.
  * `pipeline_telemetry`: pipeline server usage metrics.
* `pipeline`: the name of the metric showing the id of the pipeline.
* `cluster`: the name of the metric showing the id of the cluster.
