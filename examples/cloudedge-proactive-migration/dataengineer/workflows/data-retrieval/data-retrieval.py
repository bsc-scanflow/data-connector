import requests
import pandas as pd
import datetime
import os 
import urllib
import logging 
import click
import sys
import mlflow
from mlflow.tracking import MlflowClient
sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

# Prometheus server URL
prometheus_url = "http://prometheus-k8s.monitoring.svc.cluster.local:9090/api/v1/query_range"

def query_prometheus(query):
    logging.info(f"Running query:{query}")
    current_time = datetime.datetime.now().replace(second=0, microsecond=0)
    now = int(current_time.timestamp())
    params = {
        'query': query,
        'start': now - 600,  # 10 minutes ago
        'end': now,         # current time
        'step': '1m'
    }
    try:
        response = requests.get(prometheus_url, params=params, timeout=10)
        logging.info(f"Full Prometheus query URL: {response.url}")
        
        if response.status_code != 200:
            logging.error(f"Prometheus query error: {response.status_code}")
            logging.error(f"Response content: {response.text}")
        
        response.raise_for_status()
        data = response.json()
        logging.info(f"Prometheus response: {data}")
        return data
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Request to Prometheus failed: {e}")
        raise
    except ValueError as e:
        logging.error(f"Error parsing Prometheus response: {e}")
        raise

def process_data(data, metric_name, instance):
    result = []
    for item in data['data']['result']:
        for value in item['values']:
            timestamp = datetime.datetime.fromtimestamp(value[0])
            result.append({
                'timestamp': timestamp,
                'node': instance,
                metric_name: float(value[1])
            })
    return result

def cleanup_output_directory(output_path: str) -> None:
    """
    Remove all CSV files from the output directory
    """
    logging.info(f"Cleaning up directory: {output_path}")
    try:
        if os.path.exists(output_path):
            for filename in os.listdir(output_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(output_path, filename)
                    os.remove(file_path)
                    logging.info(f"Removed old file: {filename}")
    except Exception as e:
        logging.error(f"Error cleaning up directory: {e}")

def calculate_tsrequest_differential(node_data):
    """
    Calculate the differential of TSRequest values
    Returns a new list with the differential values, excluding the first timestamp
    """
    if not node_data or len(node_data) < 2:
        return []
        
    result = []
    for i in range(1, len(node_data)):
        current_data = node_data[i].copy()  # Create a copy of the current data point
        
        # Calculate differential if TSRequest exists in both current and previous
        if ('TSRequest' in current_data and 
            'TSRequest' in node_data[i-1] and 
            current_data['TSRequest'] is not None and 
            node_data[i-1]['TSRequest'] is not None):
            
            current_data['TSRequest_diff'] = current_data['TSRequest'] - node_data[i-1]['TSRequest']
            # Remove original TSRequest column
            del current_data['TSRequest']
        result.append(current_data)
    
    return result

def find_torchserve_node(all_node_data):
    """
    Find the node that has TorchServe metrics and normalize CPU utilization
    by the total number of CPU cores
    """
    for node_data in all_node_data:
        if not node_data:  # Skip empty data
            continue
        if 'CPUUtilizationTS' in node_data[0] and node_data[0]['CPUUtilizationTS'] is not None and 'CPUTotalNode' in node_data[0]:
            # Normalize CPU utilization for all timestamps
            for data_point in node_data:
                if data_point['CPUUtilizationTS'] is not None and data_point['CPUTotalNode'] is not None:
                    data_point['CPUUtilizationTS'] = (data_point['CPUUtilizationTS'] / data_point['CPUTotalNode'])*100
            return node_data
    return None

def propagate_torchserve_metrics(source_data, target_data):
    """
    Propagate TorchServe metrics from source node to target node with appropriate scaling
    """
    # Convert to DataFrame for easier manipulation
    source_df = pd.DataFrame(source_data)
    target_df = pd.DataFrame(target_data)
    
    # Copy direct metrics
    target_df['TSRequest'] = source_df['TSRequest']
    target_df['PredictionTimeTS'] = source_df['PredictionTimeTS']
    
    # Scale CPU utilization based on the ratio of total resources
    if all(col in source_df.columns for col in ['CPUUtilizationTS', 'CPUTotalNode']) and 'CPUTotalNode' in target_df.columns:
        cpu_ratio = source_df['CPUTotalNode'] / target_df['CPUTotalNode']
        target_df['CPUUtilizationTS'] = source_df['CPUUtilizationTS'] * cpu_ratio
        target_df['CPUUtilizationNode'] = target_df['CPUUtilizationNode'] + target_df['CPUUtilizationTS']
    
    # Scale Memory utilization based on the ratio of total resources
    if all(col in source_df.columns for col in ['MemoryUtilizationTS', 'MemoryTotalNode']) and 'MemoryTotalNode' in target_df.columns:
        memory_ratio = source_df['MemoryTotalNode'] / target_df['MemoryTotalNode']
        target_df['MemoryUtilizationTS'] = source_df['MemoryUtilizationTS'] * memory_ratio
        target_df['MemoryUtilizationNode'] = target_df['MemoryUtilizationNode'] + target_df['MemoryUtilizationTS']
    
    return target_df.to_dict('records')

def format_data(node_data):
    """
    Format the data according to specifications:
    - Add hour column
    - Drop unnecessary columns
    - Reorder columns
    
    Args:
        node_data: List of dictionaries containing the data
    """
    if not node_data:
        return []
        
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(node_data)
    
    # Add hour column
    df['hour'] = df['timestamp'].dt.hour
    
    # Rename timestamp column to date
    df = df.rename(columns={'timestamp': 'date'})
    
    # Drop unnecessary columns
    columns_to_drop = ['CPUTotalNode', 'MemoryTotalNode']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Define desired column order
    desired_columns = [
        'date',
        'node',
        'hour',
        'CPUUtilizationNode',
        'MemoryUtilizationNode',
        'CPUUtilizationTS',
        'MemoryUtilizationTS',
        'TSRequest_diff',
        'PredictionTimeTS'
    ]
    
    # Reorder columns (only include columns that exist in the DataFrame)
    df = df[[col for col in desired_columns if col in df.columns]]
    
    return df.to_dict('records')

@click.command(help="Retrieve real-time data from Prometheus")
@click.option("--experiment_name", default=None, type=str)
@click.option("--team_name", default=None, type=str)
@click.option("--output_path", default="/workflow", type=str)
def main(experiment_name, team_name, output_path="/workflow/data"):
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_path}")

    cleanup_output_directory(output_path)

    namespaces=["experiment-cloudskin-edge-0","experiment-cloudskin-edge-1","experiment-cloudskin-edge-2","experiment-cloudskin-worker-0","experiment-cloudskin-worker-1"]
    instances=["cloudskin-k8s-edge-worker-0.novalocal","cloudskin-k8s-edge-worker-1.novalocal","cloudskin-k8s-edge-worker-2.novalocal","cloudskin-k8s-worker-0.novalocal","cloudskin-k8s-worker-1.novalocal"]
    
    all_node_data = []
    
    for i in range(len(instances)):
        queries = {
            "CPUUtilizationNode": (f'100 - (avg by(instance) (irate(node_cpu_seconds_total{{mode="idle", instance="{instances[i]}"}}[2m])) * 100)'),
            "MemoryUtilizationNode": (f'100 - (avg_over_time(node_memory_MemAvailable_bytes{{instance="{instances[i]}"}}[2m]) 'f'/ avg_over_time(node_memory_MemTotal_bytes{{instance="{instances[i]}"}}[2m]) * 100)'),
            "CPUTotalNode": (f"machine_cpu_cores{{node=\"{instances[i]}\"}}"),
            "MemoryTotalNode": (f"node_memory_MemTotal_bytes{{instance=\"{instances[i]}\"}}"),
            "CPUUtilizationTS": (f'rate(container_cpu_usage_seconds_total{{container="torchserve", namespace="{namespaces[i]}"}}[2m])'),  # Amount of CPU cores used by TorchServe
            "MemoryUtilizationTS": (f"MemoryUtilization{{namespace=\"{namespaces[i]}\"}}"),  # Memory % used by TorchServe (0-100)
            "TSRequest": (f"ts_inference_requests_total{{namespace=\"{namespaces[i]}\"}}"),
            "PredictionTimeTS": (f"PredictionTime{{namespace=\"{namespaces[i]}\"}}"),
        }

        node_data = []
        
        for metric_name, query in queries.items():
            data = query_prometheus(query)
            processed_data = process_data(data, metric_name, instances[i])
            
            # Initialize node_data with timestamp and node information if empty
            if not node_data:
                node_data = [{
                    'timestamp': item['timestamp'],
                    'node': item['node']
                } for item in processed_data]
            
            # Add metric values to existing records
            for idx, item in enumerate(processed_data):
                if idx < len(node_data):
                    node_data[idx][metric_name] = item[metric_name]
        
        if node_data:  # Only append if we have data
            all_node_data.append(node_data)
    
    # Find the node running TorchServe
    source_data = find_torchserve_node(all_node_data)
    if source_data is not None:
        # Process each node's data
        for node_data in all_node_data:
            if not node_data:  # Skip empty data
                continue
                
            if node_data[0]['node'] != source_data[0]['node']:  # If this is not the source node
                node_data = propagate_torchserve_metrics(source_data, node_data)
            
            # Calculate TSRequest differential
            node_data = calculate_tsrequest_differential(node_data)
            
            if node_data:  # Check if we still have data after differential
                # Format the data
                node_data = format_data(node_data)
                
                # Convert to DataFrame for saving
                df = pd.DataFrame(node_data)
                
                # Create the CSV filename and full path
                timestamp = pd.to_datetime(df['date'].iloc[0])
                filename = f"{df['node'].iloc[0]}_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
                file_path = os.path.join(output_path, filename)
                
                # Save to CSV
                df.to_csv(file_path, index=False)
                logging.info(f"Data saved to {file_path}")
    
    store_query_results(
        experiment_name=experiment_name,
        team_name=team_name,
        query_results=file_path
    )

def store_query_results(experiment_name:str, team_name:str , query_results:str) -> None:
    """
    Store the query results as an MLflow artifact
    """
    client = ScanflowTrackerClient(verbose=True)

    logging.info(f"Uploading query results dir {os.path.dirname(query_results)} as artifacts...")
    client.save_app_artifacts(
        app_name=experiment_name,
        team_name=team_name,
        app_dir=os.path.dirname(query_results)
    )

if __name__ == "__main__":
    main()