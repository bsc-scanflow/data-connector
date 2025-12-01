

class Tracker:
    def __init__(self,
                 nodePort: int,
                 image: str = "registry.gitlab.bsc.es/datacentric-computing/cloudskin-project/cloudskin-registry/scanflow-tracker",
                 image_pull_secret: str = None,
                 image_pull_policy: str = "Always"
                 ):
        
        self.image = image
        self.image_pull_secret = image_pull_secret
        self.nodePort = nodePort
        self.image_pull_policy = image_pull_policy
