import logging
import requests

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

# Actuator class
class NearbyOneActuator:
    # (hardcoded) Available Site IDs
    # TODO: Retrieve Site IDs from NearbyOne API with names
    site_ids: [] = [
        ("ecd3913a-3364-4961-9f61-87e312ef1798", "cloud-vm-1"),
        ("a2826ee9-3807-4ae3-88f7-465e2bf5f65c", "edge-server")
    ]

    def get_next_site_id(self, source_site_id: str):
        """
        Return the next site_id in the site_ids array
        If site_id is the last one, return the first one
        """
        for i, (site_id, site_name) in enumerate(self.site_ids):
            if site_id == source_site_id:
                if i == len(self.site_ids) -1:
                    return self.site_ids[0]
                else:
                    return self.site_ids[i + 1]
        # Return None if there's no site_id match!
        return None

    def retrieve_services(self, cluster_id: str):
        """
        Return available services in a given cluster ID
        return: Array of ServiceChainResponse objects in JSON format
        """
        pass


    @staticmethod
    def find_service(services: [{}], service_name: str):
        """
        Return a service from the services array with the same service_name as the given one
        return: ServiceChainResponse object in JSON format
        """
        pass

    def delete_service(self, service_id: str):
        """
        Send a DELETE service request to the NearbyOne API
        return: HTTP response?
        """
        pass

    @staticmethod
    def compose_deploy_service_payload(site_id: str, service_name: str):
        """
        Compose a DeployServiceChainArgs schema object in JSON format with provided site_id and service_name
        return: DeployServiceChainArgs object in JSON format
        """
        pass

    def deploy_service(self, deploy_service_payload: {}):
        """
        Send a POST service creation request to the NearbyOne API
        return: HTTP response?
        """
        pass

    def close_session(self):
        """
        Close the session with NearbyOne Orchestrator
        """
        self.session.close()

    def migrate_service(self, service_name: str, source_cluster_id: str, dest_cluster_id: str):
        """
        Proceed to migrate a service running in source_cluster_id to dest_cluster_id.
        If dest_cluster_id is empty, take the next cluster_id after source_cluster_id from self.site_ids array
        return: migration status (TBD)
        """
        # Steps:
        # - Retrieve available services in source_cluster_id
        # - Find the service_id for the service with the given service_name
        # - Delete the service_id from the source_cluster_id
        # - Set the dest_cluster_id
        # - Compose the required DeployServiceChainArgs payload with the same service_name and the new dest_cluster_id
        # - Deploy the service_name using the DeployServiceChainArgs
        pass

    def __init__(self, api_url: str, username: str, password: str):
        """
        Initialize a NearbyOneActuator object
        """
        self.api_url: str = api_url
        self.session: requests.Session = requests.Session()
        # We expect BasicAuth for NearbyOne API server
        self.session.auth = (username, password)
        self.session.headers.update({"Accept": "application/json"})
