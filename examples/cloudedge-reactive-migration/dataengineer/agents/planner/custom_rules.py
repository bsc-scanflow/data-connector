def qos_constraints(avg_qos: float):
    # Migration is triggered if avg QoS is higher than 100ms
    return avg_qos > 0.1

def qos_check(qos: float, cluster_type: str) -> bool:
    """
    Return true if the application's QoS should run elsewhere, false otherwise
    - If cluster_type is 'edge', then return True if QoS is above 200ms to move to the Cloud
    - If cluster_type is 'cloud', then return True if QoS is below 50ms to bring the app back to the Edge
    :param qos - QoS value of the application
    :param cluster_type - Type of cluster (edge, cloud)
    :return Boolean value
    """
    match cluster_type:
        case "edge":
            return qos > 0.2
        case "cloud":
            return qos < 0.045
        case _:
            # In case the cluster type is unknown, play it safe and don't trigger the migration
            return False