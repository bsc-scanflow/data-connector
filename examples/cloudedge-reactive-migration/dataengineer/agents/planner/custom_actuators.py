import logging
import requests

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

# Actuator class
class NearbyOneActuator:
    # (hardcoded) Available Site IDs
    # TODO: Retrieve Site IDs from NearbyOne API with names
    site_ids: list = [
        ("de5dad18-46c7-4d40-a09d-d929828a2189", "k8s-cnx1"),
        ("fe780ac2-4230-44ed-aea7-2fc47d4ae33c", "k8s-edge-cluster")
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

    def get_site_id_name(self, current_site_id: str) -> str:
        """
        Return the site_id name from the lookup table
        """
        for (site_id, site_name) in self.site_ids:
            if current_site_id == site_id:
                return site_name
            
        return "None"

    def retrieve_services(self, cluster_id: str):
        """
        Return available services in a given cluster ID
        return: Array of ServiceChainResponse objects in JSON format
        """
        pass


    @staticmethod
    def find_service(services: list, service_name: str):
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


    def deploy_service(self, deploy_service_payload: dict):
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


    def migrate_service(self, service_name: str, source_cluster_id: str):
        """
        Proceed to migrate a service running in source_cluster_id to dest_cluster_id.
        If dest_cluster_id is empty, take the next cluster_id after source_cluster_id from self.site_ids array
        return: migration status (TBD)
        """
        # Steps:
        # - Retrieve available services in source_cluster_id

        # - Find the service_id for the service with the given service_name

        # - Delete the service_id from the source_cluster_id

        # - Find the dest_cluster_id
        dest_cluster_id, dest_cluster_name = self.get_next_site_id(source_site_id=source_cluster_id)

        # - Compose the required DeployServiceChainArgs payload with the same service_name and the new dest_cluster_id

        # - Deploy the service_name using the DeployServiceChainArgs
        logging.info("Migrating service. Coming soon!")
        logging.info(str({
            "service_name": service_name,
            "source_cluster_id": source_cluster_id,
            "source_cluster_name": self.get_site_id_name(source_cluster_id),
            "dest_cluster_id": dest_cluster_id,
            "dest_cluster_name": dest_cluster_name,
            }
        ))
        # return str({
        #     "service_name": service_name,
        #     "source_cluster_id": source_cluster_id,
        #     "source_cluster_name": self.get_site_id_name(source_cluster_id),
        #     "dest_cluster_id": dest_cluster_id,
        #     "dest_cluster_name": dest_cluster_name,
        #     }
        # )
        return 1


    def __init__(self, api_url: str, username: str, password: str):
        """
        Initialize a NearbyOneActuator object
        """
        # TODO: test authentication!
        self.api_url: str = api_url
        self.session: requests.Session = requests.Session()
        # We expect BasicAuth for NearbyOne API server
        self.session.auth = (username, password)
        self.session.headers.update({"Accept": "application/json"})


def migrate_application(app_name: str, current_cluster_id: str, nearbyone_url: str, nearbyone_username: str, nearbyone_password: str):

    # Initialize a NearbyOneActuator
    nearby_actuator = NearbyOneActuator(
        api_url=nearbyone_url,
        username=nearbyone_username,
        password=nearbyone_password
    )
    
    # Migrate the service
    return nearby_actuator.migrate_service(
        service_name=app_name,
        source_cluster_id=current_cluster_id
    )