import logging
import json
from kubernetes import client, config, utils

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

#ok!
async def call_migrate_app(max_qos_index, namespace, deployment_name):
    nodeName_list=['cloudskin-k8s-control-plane-0.novalocal',
                 'cloudskin-k8s-worker-1.novalocal',
                 'cloudskin-k8s-worker-0.novalocal',
                 'cloudskin-k8s-edge-worker-2.novalocal',
                 'cloudskin-k8s-edge-worker-1.novalocal',
                 'cloudskin-k8s-edge-worker-0.novalocal']

    # Prepare the patch, which sets the nodeSelector
    patch_body = {
        "spec": {
            "template": {
                "spec": {
                    "nodeSelector": {"kubernetes.io/hostname":nodeName_list[int(max_qos_index)]}
                }
            }
        }
    }

    logging.info(f"agent is patch deployment to node - {patch_body}")

    #connect k8s
    config.load_incluster_config()

    api_instance = client.AppsV1Api()
    
    try:
        api_instance.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=patch_body
        )
        logging.info("update_deployment_with_patch succeeded")
        return True
    except client.api_client.rest.ApiException as e:
        logging.error(f"update_deployment_with_patch failed: {e}")
        return False
