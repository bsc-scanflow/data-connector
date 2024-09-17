def qos_constraints(avg_qos: float):
    # Migration is triggered if avg QoS is higher than 100ms
    return avg_qos > 0.1
    