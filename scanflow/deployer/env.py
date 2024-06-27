# from typing import Optional, List, Dict

# class ScanflowSecret():
#     AWS_ACCESS_KEY_ID : Optional[str] = "scanflow"
#     AWS_SECRET_ACCESS_KEY : Optional[str] = "scanflow123"
#     MLFLOW_S3_ENDPOINT_URL : Optional[str] = "http://minio.scanflow-server.svc.cluster.local"
#     AWS_ENDPOINT_URL: Optional[str] = "http://minio.scanflow-server.svc.cluster.local"


# class ScanflowTrackerConfig():
#     TRACKER_STORAGE: Optional[str] = "postgresql://postgres:scanflow123@scanflow-postgres.scanflow-server.svc.cluster.local/scanflow-default"
#     TRACKER_ARTIFACT: Optional[str] = "s3://scanflow-default"

# class ScanflowClientConfig():
#     SCANFLOW_TRACKER_URI : Optional[str] = "http://scanflow-tracker-service.scanflow-server.svc.cluster.local"
#     SCANFLOW_SERVER_URI : Optional[str] = "http://scanflow-server-service.scanflow-server.svc.cluster.local"
#     SCANFLOW_TRACKER_LOCAL_URI : Optional[str] = "http://scanflow-tracker.scanflow-default.svc.cluster.local"


# class ScanflowEnvironment():
#     namespace: Optional[str] = "scanflow-default" 
#     #role: now we start with default
#     #secret
#     secret : Optional[ScanflowSecret] = ScanflowSecret()
#     #secret_stringData : Optional[dict] = {
#     #    "AWS_ACCESS_KEY_ID": "admin", 
#     #    "AWS_SECRET_ACCESS_KEY": "admin123", 
#     #    "MLFLOW_S3_ENDPOINT_URL": "http://minio.minio-system.svc.cluster.local:9000",
#     #    "AWS_ENDPOINT_URL": "http://minio.scanflow-server.svc.cluster.local"  
#     #}
#     image_pull_secret: Optional[ScanflowImagePullSecret] = ScanflowImagePullSecret()
#     #configmap tracker
#     tracker_config : Optional[ScanflowTrackerConfig] = ScanflowTrackerConfig()
#     #configmap_tracker_data : Optional[dict] = {
#     #    "TRACKER_STORAGE": "postgresql://scanflow:scanflow123@postgresql-service.postgresql.svc.cluster.local/scanflow-default",
#     #    "TRACKER_ARTIFACT": "s3://scanflow-default"
#     #}
#     #configmap client
#     client_config : Optional[ScanflowClientConfig] = ScanflowClientConfig()
#     #configmap_remotescanflow_data : Optional[dict] = {
#     #    "SCANFLOW_TRACKER_URI" : "http://scanflow-tracker-service.scanflow-system.svc.cluster.local",
#     #    "SCANFLOW_SERVER_URI" : "http://scanflow-server-service.scanflow-system.svc.cluster.local"
#     #}
#     #configmap_localscanflow_data : Optional[dict] = {
#     #    "SCANFLOW_TRACKER_LOCAL_URI" : "http://scanflow-tracker.scanflow-default.svc.cluster.local"
#     #}



from typing import Optional

class ScanflowSecret:
    def __init__(self,
                 AWS_ACCESS_KEY_ID: Optional[str] = "scanflow",
                 AWS_SECRET_ACCESS_KEY: Optional[str] = "scanflow123",
                 MLFLOW_S3_ENDPOINT_URL: Optional[str] = "http://minio.scanflow-server.svc.cluster.local",
                 AWS_ENDPOINT_URL: Optional[str] = "http://minio.scanflow-server.svc.cluster.local"):
        self.AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID
        self.AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY
        self.MLFLOW_S3_ENDPOINT_URL = MLFLOW_S3_ENDPOINT_URL
        self.AWS_ENDPOINT_URL = AWS_ENDPOINT_URL

    def to_dict(self):
        return {
            "AWS_ACCESS_KEY_ID": self.AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": self.AWS_SECRET_ACCESS_KEY,
            "MLFLOW_S3_ENDPOINT_URL": self.MLFLOW_S3_ENDPOINT_URL,
            "AWS_ENDPOINT_URL": self.AWS_ENDPOINT_URL
        }

class ScanflowImagePullSecret():
    registry: Optional[str] = None
    name: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    email: Optional[str] = None

class ScanflowTrackerConfig:
    def __init__(self,
                 TRACKER_STORAGE: Optional[str] = "postgresql://postgres:scanflow123@scanflow-postgres.scanflow-server.svc.cluster.local/scanflow-default",
                 TRACKER_ARTIFACT: Optional[str] = "s3://scanflow-default"):
        self.TRACKER_STORAGE = TRACKER_STORAGE
        self.TRACKER_ARTIFACT = TRACKER_ARTIFACT

    def to_dict(self):
        return {
            "TRACKER_STORAGE": self.TRACKER_STORAGE,
            "TRACKER_ARTIFACT": self.TRACKER_ARTIFACT
        }

class ScanflowClientConfig:
    def __init__(self,
                 SCANFLOW_TRACKER_URI: Optional[str] = "http://scanflow-tracker-service.scanflow-server.svc.cluster.local",
                 SCANFLOW_SERVER_URI: Optional[str] = "http://scanflow-server-service.scanflow-server.svc.cluster.local",
                 SCANFLOW_TRACKER_LOCAL_URI: Optional[str] = "http://scanflow-tracker.scanflow-default.svc.cluster.local"):
        self.SCANFLOW_TRACKER_URI = SCANFLOW_TRACKER_URI
        self.SCANFLOW_SERVER_URI = SCANFLOW_SERVER_URI
        self.SCANFLOW_TRACKER_LOCAL_URI = SCANFLOW_TRACKER_LOCAL_URI

    def to_dict(self):
        return {
            "SCANFLOW_TRACKER_URI": self.SCANFLOW_TRACKER_URI,
            "SCANFLOW_SERVER_URI": self.SCANFLOW_SERVER_URI,
            "SCANFLOW_TRACKER_LOCAL_URI": self.SCANFLOW_TRACKER_LOCAL_URI
        }

class ScanflowEnvironment:
    def __init__(self,
                 namespace: Optional[str] = "scanflow-default",
                 secret: Optional[ScanflowSecret] = ScanflowSecret(),
                 image_pull_secret: Optional[ScanflowImagePullSecret] = ScanflowImagePullSecret(),
                 tracker_config: Optional[ScanflowTrackerConfig] = ScanflowTrackerConfig(),
                 client_config: Optional[ScanflowClientConfig] = ScanflowClientConfig()):
        self.namespace = namespace
        self.secret = secret
        self.image_pull_secret = image_pull_secret
        self.tracker_config = tracker_config
        self.client_config = client_config

    def to_dict(self):
        return {
            "namespace": self.namespace,
            "secret": self.secret.to_dict(),
            "tracker_config": self.tracker_config.to_dict(),
            "client_config": self.client_config.to_dict()
        }



