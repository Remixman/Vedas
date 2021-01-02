import matplotlib.pyplot as plt
from matplotlib import pylab
import networkx as nx

def draw_graph(graph):
  # There are graph layouts like shell, spring, spectral and random.
  # Shell layout usually looks better, so we're choosing it.
  # I will show some examples later of other layouts
  # graph_pos = nx.graphviz_layout(G)
  graph_pos = nx.spring_layout(G,scale=20)
  # graph_pos = nx.spectral_layout(G)

  # draw nodes, edges and labels
  nx.draw_networkx_nodes(G, graph_pos, node_size=50, node_color='blue', alpha=0.3)
  nx.draw_networkx_edges(G, graph_pos)
  # nx.draw_networkx_labels(G, graph_pos, font_size=3, font_family='sans-serif')

  # show graph
  plt.show()


# G = nx.random_geometric_graph(200, 0.125)
G = nx.Graph()
edges = nx.read_edgelist('edges.txt')
nodes = nx.read_adjlist("nodes.txt")
G.add_edges_from(edges.edges())
G.add_nodes_from(nodes)
draw_graph(G)