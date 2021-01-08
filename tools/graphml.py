import networkx as nx

G = nx.Graph()
edges = nx.read_edgelist('edges.txt')
nodes = nx.read_adjlist("nodes.txt")
G.add_edges_from(edges.edges())
G.add_nodes_from(nodes)
nx.write_graphml(G, "triple-data.graphml")