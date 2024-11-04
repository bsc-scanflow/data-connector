import logging
import requests
import json
import os
import yaml

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

# TODO: Import Nearby API client and KratosClient for authentication
from nbi_client import NbiClient
from inno_nbi_api.models.site_response import Site
from inno_nbi_api.models.service_chain_response import ServiceChainResponseServiceChain
from inno_nbi_api.models.deploy_service_chain_args import DeployServiceChainArgs
from inno_nbi_api.models.chart_repo_index_entry import ChartRepoIndexEntry
from inno_nbi_api.models.block_args_deploy import BlockArgsDeploy


# Actuator class
class NearbyOneActuator:
    # Available Sites
    # - Retrieve Sites from NearbyOne API
    sites: list[type[Site]]

    def update_available_sites(self, description: str = "") -> None:
        """
        Update the site_ids dictionary with the infrastructure's available sites
        :return None
        """
        # Initialize a new site_ids list
        new_sites: list[type[Site]] = []
        # Go through all of the available organizations' sites
        for org in self.nbi_client.org_api.get_organizations():
            for site_id in org.sites:
                site = self.nbi_client.infra_api.get_site_details(site_id=site_id).site
                # Only take into account 'Worker cluster' sites for migration
                if site.description.startswith(description):
                    new_sites.append(site)
                
        self.sites = new_sites

    def get_site(self, site_id: str) -> Site:
        """
        Return a Site object that matches the provided site_id
        :return Site
        """
        for site in self.sites:
            if site_id == site.id:
                return site
        
        # Return empty if not found
        return None

    def get_next_site(self, source_site: Site) -> Site:
        """
        Return the next site in the site_ids array
        If site is the last one, return the first one
        :param Source Site
        :return Next Site in array
        """
        for i, site in enumerate(self.sites):
            if site.id == source_site.id:
                if i == len(self.sites) - 1:
                    return self.sites[0]
                else:
                    return self.sites[i + 1]
                
        # Return None if there's no site.id match!
        return None

    def get_all_services(self, site: Site) -> list[type[ServiceChainResponseServiceChain]]:
        """
        Return available services in a given site
        :param Site where to get services from
        :return Array of ServiceChainResponseServiceChain objects
        """
        
        return [service_chain_response.service_chain for service_chain_response in self.nbi_client.get_all_deployed_services(site_ids=[site.id])]

    def get_service(self, site: Site, service_name: str) -> ServiceChainResponseServiceChain:
        """
        Return a service from the services array with the same service_name as the given one
        :return ServiceChainResponseServiceChain object
        """

        for service in self.get_all_services(site=site):
            if service_name == service.name:
                return service

        # Return None if there's no service.name match!
        return None

    def delete_service(self, service: ServiceChainResponseServiceChain) -> str:
        """
        Send a DELETE service request to the NearbyOne API
        :return deleted service.id if deletion has been successful
        """
        return self.nbi_client.delete_service_chain_by_id(service_id=service.id)

    def get_marketplace_chart(self, chart_name: str) -> ChartRepoIndexEntry:
        """
        Retrieve the Marketplace Chart based on its name
        :return ChartRepoIndexEntry
        """

        for chart in self.nbi_client.list_marketplace_charts().charts:
            if chart.display_name == chart_name:
                return chart
            
        return None

    def compose_deploy_service_payload(self, site: Site, app_name: str) -> DeployServiceChainArgs:
        """
        Compose a DeployServiceChainArgs schema object in JSON format with provided site_id and service_name
        :return DeployServiceChainArgs object in JSON format
        """
        
        return None

    def deploy_service(self, site: Site, app_name: str) -> ServiceChainResponseServiceChain:
        """
        Send a POST service creation request to the NearbyOne API
        :param site: The NearbyOne site where to deploy a new service_name block
        :type site: Site
        :param service_name: The Marketplace service name to deploy
        :return HTTP response?
        """

        # Retrieve the Marketplace Chart
        marketplace_chart: ChartRepoIndexEntry = self.get_marketplace_chart(chart_name=app_name)
        
        # Verify that the Block Chart version is available:
        block_chart = self.nbi_client.fetch_block_chart(
            block_name=marketplace_chart.name,
            block_version=marketplace_chart.all_versions[0]
        )

        if not block_chart:
            logging.error(f"Block Chart {marketplace_chart.name} version {marketplace_chart.all_versions[0]} doesn't exist! Can't be deployed")
            return None
        
        # TODO - Load the override values YAML
        # - These might come from an external YAML
        override_values = ""

        # Compose the BlockArgsDeploy object
        block_args: BlockArgsDeploy = BlockArgsDeploy(
            site_id=site.id,
            displayName=marketplace_chart.display_name,
            blockChartName=marketplace_chart.name,
            blockChartVersion=marketplace_chart.all_versions[0], # Retrieve latest version by default
            values=override_values # Override block chart values 
        )

        # Compose the required DeployServiceChainArgs payload with the new service_name and the new dest_cluster
        dest_service_name = f"{app_name} - {site.display_name}"
        
        # Initialize a DeployServiceChainArgs object
        deploy_args = DeployServiceChainArgs(
            name=dest_service_name,
            blocks=[block_args.model_dump()]
        )

        # Deploy the new service
        response = self.nbi_client.deploy_service(deploy_args=deploy_args.model_dump())
        
        if not response:
            return None
        
        # Return the deployed service
        return self.nbi_client.get_deployed_service(service_id=response.strip('"'))
        

    def migrate_service(self, app_name: str, source_cluster_id: str) -> str:
        """
        Proceed to migrate a service running in source_cluster_id to dest_cluster_id.
        If dest_cluster_id is empty, take the next cluster_id after source_cluster_id from self.site_ids array
        :return migration status (TBD)
        """
        # Steps:
        # - Update the available sites
        self.update_available_sites(description="Worker cluster")
        logging.debug("Available sites updated")

        # - Retrieve the source Site
        source_site: Site = self.get_site(site_id=source_cluster_id)
        logging.debug(f"Source site: {source_site.display_name} - {source_site.id}")

        # - Find the service with the given source_service_name
        source_service_name: str = f"{app_name} - {source_site.display_name}"
        source_service: ServiceChainResponseServiceChain = self.get_service(site=source_site, service_name=source_service_name)
        
        if source_service:
            logging.info(f"Source service {source_service.name} found!")
        else:
            logging.error(f"Source service {source_service_name} not found! Unable to remove it after migration")

        # - Find the dest_cluster
        dest_site = self.get_next_site(source_site=source_site)

        # - Deploy the service_name using the DeployServiceChainArgs
        logging.info("Migrating service...")
        dest_service = self.deploy_service(site=dest_site, app_name=app_name)
        
        if not dest_service:
            logging.error("Service couldn't be deployed!")
            return {
                "message": "Service couldn't be deployed"
            }
        
        # - Delete the service from the source_cluster
        if source_service:
            deleted_service =  self.delete_service(service=source_service)
        else:
            deleted_service = "Not found"
        
        # Compose and return the migration results
        migration_result = {
            "source_cluster_id": source_site.id,
            "source_cluster_name": source_site.display_name,
            "deleted_service_id": deleted_service.strip('"'),
            "dest_cluster_id": dest_site.id,
            "dest_cluster_name": dest_site.display_name,
            "deployed_service_name": dest_service.name,
        }

        logging.debug(str(migration_result))
        return migration_result

    def __init__(self):
        """
        Initialize a NearbyOneActuator object
        """
        # TODO: test authentication!
        self.nbi_client = NbiClient()


def migrate_application(app_name: str, current_cluster_id: str, nearbyone_env_name: str, nearbyone_org_id: str, nearbyone_email: str, nearbyone_password: str) -> str:

    # TODO: initialize somewhere the following environment variables, required for KratosClient. Maybe during NbiClient initialization?
    os.environ["NBY_ENV_EMAIL"] = nearbyone_email
    os.environ["NBY_ENV_PASSWORD"] = nearbyone_password
    os.environ["NBY_ORGANIZATION_ID"] = nearbyone_org_id
    os.environ["NBY_ENV_NAME"] = nearbyone_env_name

    # Initialize a NearbyOneActuator
    nearby_actuator = NearbyOneActuator(
    )
    
    # Migrate the service
    return nearby_actuator.migrate_service(
        app_name=app_name,
        source_cluster_id=current_cluster_id
    )