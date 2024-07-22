# Installing 


## Prerequisites

deploy-backend
- Kubernetes 1.12+ 
- Argo 3.0+

tracker(mlflow)
- artifact store - Minio
- backend store - PostgreSQL
- create scanflow db

## Scanflow tracker 

```bash
$ docker pull registry.gitlab.bsc.es/datacentric-computing/cloudskin-project/cloudskin-registry/scanflow-tracker
```

| Service|cluster host ip|Port|NodePort|
|----------------|-----------------|----------------|-------------|
|`Scanflow tracker(mlflow)`|  172.30.0.50 | 8080 | 46667 |

## Installing kubernetes via helm charts

To install the scanflow with chart:

create scanflow-kubernetes namespace

```bash
kubectl create namespace scanflow-system
```

```bash
helm install <specified-name> helm/chart --namespace <namespace> 

e.g :
helm install scanflow helm/chart --namespace scanflow-system
```

This command deploys scanflow in kubernetes cluster with default configuration.  The [configuration](#configuration) section lists the parameters that can be configured during installation.

## Uninstalling the Chart

```bash
$ helm delete --namespace scanflow-system scanflow
```

## Configuration

The following are the list configurable parameters of Scanflow Helm Chart and their default values.

| Parameter|Description|Default Value|
|----------------|-----------------|----------------------|
|`basic.image_tag_version`| Docker image version Tag | `latest`|
|`basic.scanflow_tracker_image_name`|server Docker Image Name|`172.30.0.49/scanflow-tracker`|
|`basic.image_pull_policy`|Image Pull Policy|`IfNotPresent`|
|`tracker.scanflow_tracker_storage_backend`|||
|`tracker.scanflow_tracker_storage_url`|||
|`tracker.scanflow_tracker_storage_username`|||
|`tracker.scanflow_tracker_storage_password`|||
|`tracker.scanflow_tracker_artifact_backend`|||
|`tracker.scanflow_tracker_artifact_url`|||
|`tracker.scanflow_tracker_artifact_username`|||
|`tracker.scanflow_tracker_artifact_password`|||

> **Tip**: You can use the default [values.yaml](helm/chart/values.yaml)