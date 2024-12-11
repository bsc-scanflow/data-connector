import requests
import pandas as pd
import datetime
import os 

# Prometheus server URL
prometheus_url = "http://localhost:9090/api/v1/query_range"

        #'start': '2024-07-10T11:03:00Z',
        #'end': '2024-07-10T11:27:00Z',

def query_prometheus(query):
    print(query)
    params = {
        'query': query,
        'start': '10 minutes ago',
        'step': '1m'
    }
    response = requests.get(prometheus_url, params=params)
    # print(prometheus_url, params)
    response.raise_for_status()
    data = response.json()
    # print(data)  # Log the raw response from Prometheus
    return data

def process_data(data, metric_name, namespace):
    result = []
    for item in data['data']['result']:
        for value in item['values']:
            timestamp = datetime.datetime.fromtimestamp(value[0])
            result.append({
                'timestamp': timestamp,
                'namespace': namespace,
                metric_name: float(value[1])
            })
    return result

def save_to_csv(data, filename):
    output_directory = "./data"
    os.makedirs(output_directory, exist_ok=True)
    file_path = os.path.join(output_directory, filename)
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


def main():

    namespaces=["experiment-cloudskin-edge-0","experiment-cloudskin-edge-1","experiment-cloudskin-edge-2","experiment-cloudskin-worker-0","experiment-cloudskin-worker-1"]
    instances=["cloudskin-k8s-edge-worker-0.novalocal","cloudskin-k8s-edge-worker-1.novalocal","cloudskin-k8s-edge-worker-2.novalocal","cloudskin-k8s-worker-0.novalocal","cloudskin-k8s-worker-1.novalocal"]
    
    # namespaces=["experiment-cloudskin-edge-1"]
    # instances=["cloudskin-k8s-edge-worker-1.novalocal"]
    
    
    for i in range(len(namespaces)):
        # Queries to be executed

        queries = {
            "CPUUtilizationNode": (f'100 - (avg by(instance) (irate(node_cpu_seconds_total{{mode="idle", instance="{instances[i]}"}}[1m])) * 100)'),
            "MemoryUtilizationNode": (f'100 - (avg_over_time(node_memory_MemAvailable_bytes{{instance="{instances[i]}"}}[1m]) 'f'/ avg_over_time(node_memory_MemTotal_bytes{{instance="{instances[i]}"}}[1m]) * 100)'),
            "DiskUtilizationNode": (f'rate(node_disk_io_time_seconds_total{{device!~"sr0|loop.*",instance="{instances[i]}"}}[1m])'),
            "CPUTotalNode": (f"machine_cpu_cores{{node=\"{instances[i]}\"}}"),
            "MemoryTotalNode": (f"node_memory_MemTotal_bytes{{instance=\"{instances[i]}\"}}"),
            #"DiskTotalNode":Prometheus and node-exporter don't directly expose hardware-level information about the maximum IOPS capabilities of a node's disk,
            "CPUUtilizationTS": (f'rate(container_cpu_usage_seconds_total{{container="torchserve", namespace="{namespaces[i]}"}}[1m])'),
            "MemoryUtilizationTS": (f"MemoryUtilization{{namespace=\"{namespaces[i]}\"}}"),
            "DiskUtilizationTS":(f"DiskUtilization{{namespace=\"{namespaces[i]}\"}}"),
            "PredictionTimeTS": (f"PredictionTime{{namespace=\"{namespaces[i]}\"}}"),
            "TSRequest": (f"ts_inference_requests_total{{namespace=\"{namespaces[i]}\"}}"),
            "TSInferenceLatency": (f"ts_inference_latency_microseconds{{namespace=\"{namespaces[i]}\"}} / 1e3"),
            "TSQueueLatency": (f"ts_queue_latency_microseconds{{namespace=\"{namespaces[i]}\"}} / 1e3"),
            "TSLatency": (f"(ts_inference_latency_microseconds{{namespace=\"{namespaces[i]}\"}} "f"+ ts_queue_latency_microseconds{{namespace=\"{namespaces[i]}\"}}) / 1e3")
            }

        data_by_namespace = {} 
        
        for metric_name, query in queries.items():
            # print(query)
            data = query_prometheus(query)
            #print(data)
            processed_data = process_data(data, metric_name, namespaces[i])
            # print(processed_data)
            
            for item in processed_data:
               namespace = item['namespace']
               if namespace not in data_by_namespace:
                  data_by_namespace[namespace] = {}
               timestamp = item['timestamp']
               if timestamp not in data_by_namespace[namespace]:
                  data_by_namespace[namespace][timestamp] = {'timestamp': timestamp}
               data_by_namespace[namespace][timestamp][metric_name] = item[metric_name]
       
        for namespace, data in data_by_namespace.items():
            sorted_data = sorted(data.values(), key=lambda x: x['timestamp'])
            filename = f"{namespace}_{timestamp}.csv"
            save_to_csv(sorted_data, filename)
            print(f"Data saved to {filename}")


if __name__ == "__main__":
    main()
