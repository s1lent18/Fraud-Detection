import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt

dataframe = pd.read_csv("dsfd.csv")

dataframe['TransactionDate'] = pd.to_datetime(dataframe['TransactionDate'])

dataframe['PreviousTransactionDate'] = pd.to_datetime(dataframe['PreviousTransactionDate'])

dataframe['TimeGap'] = (dataframe['TransactionDate'] - dataframe['PreviousTransactionDate']).dt.total_seconds()

G = nx.Graph()

for index, row in dataframe.iterrows():
    G.add_edge(row['AccountID'], row['MerchantID'], weight=row['TransactionAmount'])    
    
nx.set_node_attributes(G, dataframe.set_index('AccountID')['AccountBalance'].to_dict(), name='balance')

dataframe['degree_centrality'] = dataframe['AccountID'].map(nx.degree_centrality(G))

dataframe['closeness_centrality'] = dataframe['AccountID'].map(nx.closeness_centrality(G))

dataframe['betweenness_centrality'] = dataframe['AccountID'].map(nx.betweenness_centrality(G))

dataframe['transaction_frequency'] = dataframe['AccountID'].map(dict(G.degree()))

dataframe['average_transaction_amount'] = dataframe.groupby('AccountID')['TransactionAmount'].transform('mean')

features = ['TransactionAmount', 'TimeGap', 'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'transaction_frequency', 'average_transaction_amount']

X = dataframe[features].fillna(0)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = IsolationForest(contamination=0.01, random_state=42)

dataframe['anomaly_score'] = model.fit_predict(X_scaled)

anomalies = dataframe[dataframe['anomaly_score'] == -1]

# Setting Distance for neighbours to be 0.3 and forming cluster with atleast 3 points

db = DBSCAN(eps=0.3, min_samples=3).fit(X_scaled)

dataframe['cluster'] = db.labels_

anomalies = dataframe[dataframe['cluster'] == -1]

lof = LocalOutlierFactor(n_neighbors=25, contamination=0.015)

dataframe['anomaly_score'] = lof.fit_predict(X_scaled)

anomalies = dataframe[dataframe['anomaly_score'] == -1]

communities = list(greedy_modularity_communities(G))

dataframe['community'] = dataframe['AccountID'].apply(lambda x: [i for i, c in enumerate(communities) if x in c][0] if x in G else -1)

small_communities = [c for c in communities if len(c) < 5]

anomalies_in_communities = [node for comm in small_communities for node in comm]

anomalies_by_centrality = dataframe[ (dataframe['degree_centrality'] < dataframe['degree_centrality'].quantile(0.01)) | (dataframe['degree_centrality'] > dataframe['degree_centrality'].quantile(0.99))]

anomalies_nodes = set(anomalies['AccountID']).union(anomalies['AccountID'], anomalies['AccountID'])

node_color = {node: 'red' if node in anomalies_nodes else 'blue' for node in G.nodes()}

pos = nx.spring_layout(G)

labels = {node: node for node in G.nodes() if node in anomalies_nodes}

plt.figure(figsize=(12, 12))

nx.draw(G, pos, with_labels=True, labels = labels, node_size=50, node_color=[node_color[node] for node in G.nodes()],font_size=10, font_weight='bold', edge_color='gray')

plt.title("Graph Visualization - Circular Layout")

plt.show()