from __future__ import annotations

import logging
import os

# Import Nearby API client and KratosClient for authentication
from nbi_client import NbiClient
from inno_nbi_api import Site
from inno_nbi_api import ServiceChainResponseServiceChain
from inno_nbi_api import DeployServiceChainArgs
from inno_nbi_api import ChartRepoIndexEntry
from inno_nbi_api import BlockArgsDeploy
from time import sleep

logger = logging.getLogger("custom_actuator")
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)


# Actuator class
class NearbyOneActuator:
    # Available Sites
    # - Retrieve Sites from NearbyOne API
    sites: list[Site]

    def update_available_sites(self, description: str = "") -> None:
        """
        Update the site_ids dictionary with the infrastructure's available sites
        :return None
        """
        # Initialize a new site_ids list
        new_sites: list[Site] = []
        # Go through all the available organizations' sites
        for org in self.nbi_client.org_api.get_organizations():
            for site_id in org.sites:
                site = self.nbi_client.infra_api.get_site_details(site_id=site_id).site
                # Only take into account 'Worker cluster' sites for migration
                if site.description.startswith(description):
                    new_sites.append(site)
                
        self.sites = new_sites

    def get_site(self, site_id: str) -> Site | None:
        """
        Return a Site object that matches the provided site_id
        :return Site
        """
        for site in self.sites:
            if site_id == site.id:
                return site
        
        # Return empty if not found
        return None

    def get_next_site(self, source_site: Site) -> Site | None:
        """
        Return the next site in the site_ids array
        If site is the last one, return the first one
        :param source_site: Source Site
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

    def get_all_services(self, site: Site) -> list[ServiceChainResponseServiceChain]:
        """
        Return available services in a given site
        :param site: Site where to get services from
        :return Array of ServiceChainResponseServiceChain objects
        """
        
        return [service_chain_response.service_chain for service_chain_response
                in self.nbi_client.get_all_deployed_services(site_ids=[site.id])]

    def get_service(self, site: Site, service_name: str) -> ServiceChainResponseServiceChain | None:
        """
        Return a service from the services array with the same service_name as the given one
        :return ServiceChainResponseServiceChain object
        """

        for service in self.get_all_services(site=site):
            # TODO: Use regex for more flexibility?
            if service_name == service.name:
                return service

        # Return None if there's no service.name match!
        return None

    def delete_service(self, service: ServiceChainResponseServiceChain) -> str:
        """
        Send a DELETE service request to the NearbyOne API
        :param service: service chain values
        :return deleted "service.id" if deletion has been successful
        """
        return self.nbi_client.delete_service_chain_by_id(service_id=service.id)

    def get_marketplace_chart(self, chart_name: str) -> ChartRepoIndexEntry | None:
        """
        Retrieve the Marketplace Chart based on its name
        :return ChartRepoIndexEntry
        """

        for chart in self.nbi_client.list_marketplace_charts().charts:
            if chart.display_name == chart_name:
                return chart
            
        return None

    def deploy_service(self, site: Site, app_name: str) -> ServiceChainResponseServiceChain | None:
        """
        Send a POST service creation request to the NearbyOne API
        :param site: The NearbyOne site where to deploy a new service_name block
        :type site: Site
        :param app_name: The Marketplace service name to deploy
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
            logger.error(
                (
                    f"Block Chart {marketplace_chart.name} "
                    f"version {marketplace_chart.all_versions[0]} doesn't exist! Can't be deployed"
                )
            )
            return None
        
        # TODO - Load the override values YAML
        # - These might come from an external YAML
        override_values = None

        # Compose the BlockArgsDeploy object
        block_args: BlockArgsDeploy = BlockArgsDeploy(
            site_id=site.id,
            displayName=marketplace_chart.display_name,
            blockChartName=marketplace_chart.name,
            blockChartVersion=marketplace_chart.all_versions[0],  # Retrieve latest version by default
            values=override_values  # Override block chart values
        )
        # DEBUG
        logger.info(f"For debugging purposes - Destination service BlockArgsDeploy:")
        logger.info(f"{block_args.to_dict()}")

        # Compose the required DeployServiceChainArgs payload with the new service_name and the new dest_cluster
        dest_service_name = f"{app_name} - {site.display_name}"
        
        # Initialize a DeployServiceChainArgs object
        deploy_args = DeployServiceChainArgs(
            name=dest_service_name,
            blocks=[block_args.model_dump()]
        )

        # Deploy the new service
        max_retries = 5
        retry = 0
        response = None
        while retry < max_retries:
            # Sleep for 5 seconds before checking the deployed service status
            if not response:
                response = self.nbi_client.deploy_service(deploy_args=deploy_args.model_dump())
            sleep(5)
            # Retrieve the service ID of the deployed service
            deployed_service_chain = self.nbi_client.get_deployed_service(service_id=response.strip('"')).service_chain
            deployed_status = self.nbi_client.OktoStatus[deployed_service_chain.status]
            logger.info(f"Deployed service status: {deployed_status.name}")
            match deployed_status:
                case self.nbi_client.OktoStatus.OKTOSTATUS_ERROR:
                    # Delete the service and prepare o redeploy it in the next iteration
                    logger.info("Error deploying service, retrying...")
                    self.delete_service(service=deployed_service_chain)
                    response = None
                case self.nbi_client.OktoStatus.OKTOSTATUS_IN_SYNC:
                    # Break the loop
                    break
                case _:
                    # Do nothing
                    pass

            retry += 1

        if not response:
            return None
        
        # Return the deployed service
        return self.nbi_client.get_deployed_service(service_id=response.strip('"')).service_chain

    def migrate_service(self, app_name: str, source_cluster_id: str) -> dict:
        """
        Proceed to migrate a service running in source_cluster_id to dest_cluster_id.
        If dest_cluster_id is empty, take the next cluster_id after source_cluster_id from self.site_ids array
        :return migration status (TBD)
        """
        # Steps:
        # - Update the available sites
        self.update_available_sites(description="Worker cluster")
        logger.debug("Available sites updated")

        # - Retrieve the source Site
        source_site: Site = self.get_site(site_id=source_cluster_id)
        logger.debug(f"Source site: {source_site.display_name} - {source_site.id}")

        # - Find the service with the given source_service_name
        source_service_name: str = f"{app_name} - {source_site.display_name}"
        source_service: ServiceChainResponseServiceChain = self.get_service(
            site=source_site,
            service_name=source_service_name
        )
        
        if source_service:
            logger.info(f"Source service {source_service.name} found!")
        else:
            logger.error(
                (
                    f"Source service {source_service_name} not found! "
                    f"It might've been already migrated on previous checks"
                )
            )
            return {
                "message": "Service already migrated in previous executions"
            }

        # - Find the dest_cluster
        dest_site = self.get_next_site(source_site=source_site)

        # - Deploy the service_name using the DeployServiceChainArgs
        logger.info("Migrating service...")
        dest_service = self.deploy_service(site=dest_site, app_name=app_name)
        
        #logger.info(f"Deployed service: {dest_service}")

        if not dest_service:
            logger.error("Service couldn't be deployed!")
            return {
                "message": "Service couldn't be deployed"
            }
        
        # - Delete the service from the source_cluster
        if source_service:
            deleted_service = self.delete_service(service=source_service)
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

        logger.debug(str(migration_result))
        return migration_result

    def __init__(self):
        """
        Initialize a NearbyOneActuator object
        """
        # TODO: test authentication!
        self.nbi_client = NbiClient()


def migrate_application(
        app_name: str, current_cluster_id: str, nearbyone_env_name: str, nearbyone_org_id: str,
        nearbyone_email: str, nearbyone_password: str) -> dict:

    # Initialize environment variables required for KratosClient
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
