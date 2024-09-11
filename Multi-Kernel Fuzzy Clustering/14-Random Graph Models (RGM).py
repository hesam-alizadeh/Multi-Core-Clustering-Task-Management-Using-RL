import numpy as np
import networkx as nx
import time
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# Function to generate Erdős-Rényi random graph
def generate_random_graph(n_nodes=100, p_edge=0.1):
    G = nx.erdos_renyi_graph(n_nodes, p_edge)
    return G

# Function to compute metrics for a graph
def compute_graph_metrics(G, n_clusters=2):
    # Convert graph to adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)
    
    # Spectral clustering
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=100)
    labels = spectral.fit_predict(adjacency_matrix)
    
    # Calculate metrics
    silhouette = silhouette_score(adjacency_matrix, labels) if len(set(labels)) > 1 else 'N/A'
    db_index = davies_bouldin_score(adjacency_matrix, labels) if len(set(labels)) > 1 else 'N/A'
    rand_idx = rand_score(G.nodes(), labels)
    ari = adjusted_rand_score(G.nodes(), labels)
    nmi = normalized_mutual_info_score(G.nodes(), labels)
    f1 = f1_score(G.nodes(), labels, average='weighted') if len(set(labels)) > 1 else 'N/A'
    
    return silhouette, db_index, rand_idx, ari, nmi, f1

# Main function to run the experiment
def run_experiment():
    # Parameters
    n_nodes = 100
    p_edge = 0.1
    n_clusters = 3

    # Generate random graph
    G = generate_random_graph(n_nodes, p_edge)

    # Time the metric computation
    start_time = time.time()
    silhouette, db_index, rand_idx, ari, nmi, f1 = compute_graph_metrics(G, n_clusters)
    execution_time = time.time() - start_time
    
    # Print metrics
    print(f"Silhouette Index: {silhouette}")
    print(f"Davies-Bouldin Index: {db_index}")
    print(f"Rand Index: {rand_idx}")
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Normalized Mutual Information (NMI): {nmi}")
    print(f"F1 Score: {f1}")
    print(f"Execution Time: {execution_time:.2f} seconds")

    # Visualize graph
    plt.figure(figsize=(10, 7))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title("Random Graph Visualization")
    plt.show()

# Run the experiment
run_experiment()
