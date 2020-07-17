import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matchFaces.py as mf

class Node:
    """ Describes a node in a graph, and the edges connected
        to that node."""

    def __init__(self, ID, neighbors, descriptor, truth=None, file_path=None):
        """ 
        Parameters
        ----------
        ID : int
            A unique identifier for this node. Should be a
            value in [0, N-1], if there are N nodes in total.

        neighbors : Sequence[int]
            The node-IDs of the neighbors of this node.

        descriptor : numpy.ndarray
            The (512,) descriptor vector for this node's picture

        truth : Optional[str]
            If you have truth data, for checking your clustering algorithm,
            you can include the label to check your clusters at the end.

            If this node corresponds to a picture of Ryan, this truth
            value can just be "Ryan"

        file_path : Optional[str]
            The file path of the image corresponding to this node, so
            that you can sort the photos after you run your clustering
            algorithm
        """
        self.id = ID  # a unique identified for this node - this should never change

        # The node's label is initialized with the node's ID value at first,
        # this label is then updated during the whispers algorithm
        self.label = ID

        # (n1_ID, n2_ID, ...)
        # The IDs of this nodes neighbors. Empty if no neighbors
        self.neighbors = tuple(neighbors)
        self.descriptor = descriptor

        self.truth = truth
        self.file_path = file_path


def plot_graph(graph, adj):
    """ Use the package networkx to produce a diagrammatic plot of the graph, with
    the nodes in the graph colored according to their current labels.

    Note that only 20 unique colors are available for the current color map,
    so common colors across nodes may be coincidental.

    Parameters
    ----------
    graph : Tuple[Node, ...]
        The graph to plot
    adj : numpy.ndarray, shape=(N, N)
        The adjacency-matrix for the graph. Nonzero entries indicate
        the presence of edges.

    Returns
    -------
    Tuple[matplotlib.fig.Fig, matplotlib.axis.Axes]
        The figure and axes for the plot."""

    g = nx.Graph()
    for n, node in enumerate(graph):
        g.add_node(n)
        
    # construct a network-x graph from the adjacency matrix: a non-zero entry at adj[i, j]
    # indicates that an egde is present between Node-i and Node-j. Because the edges are 
    # undirected, the adjacency matrix must be symmetric, thus we only look ate the triangular
    # upper-half of the entries to avoid adding redundant nodes/edges
    g.add_edges_from(zip(*np.where(np.triu(adj) > 0)))

    # we want to visualize our graph of nodes and edges; to give the graph a spatial representation,
    # we treat each node as a point in 2D space, and edges like compressed springs. We simulate
    # all of these springs decompressing (relaxing) to naturally space out the nodes of the graph
    # this will hopefully give us a sensible (x, y) for each node, so that our graph is given
    # a reasonable visual depiction 
    pos = nx.spring_layout(g)
    
    # make a mapping that maps: node-lab -> color, for each unique label in the graph
    color = list(iter(cm.tab20b(np.linspace(0, 1, len(set(i.label for i in graph))))))
    color_map = dict(zip(sorted(set(i.label for i in graph)), color))
    colors = [color_map[i.label] for i in graph]  # the color for each node in the graph, according to the node's label

    # render the visualization of the graph, with the nodes colored based on their labels!
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(g, pos=pos, ax=ax, nodelist=range(len(graph)), node_color=colors)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges())
    return fig, ax

def find_neighbors(descriptors, threshold):
    """ Uses cosine distances between descriptor vectors to determine edge
    weights between nodes

    Prevents formation of edges between vectors whose distances exceed the
    similarity threshold.

    Parameters
    ----------
    descriptors : numpy.ndarray
                  Matrix of (512,) descriptor vectors for these pictures
    threshold : double
                Cosine distance that is considered "too far" for an edge
                to exist between two nodes

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        The adjacency matrix and matrix of neighbors for each node.
    """
    # initialize adjacency and neighbors matrices
    adj_mat = []
    neighbors = []

    for i, des in enumerate(descriptors):
        dists = []
        for d in des:
            dist = mf.cos_distance(des, descriptors)
            dists.append[dist]
        dists[i] = 1  # set the "dist" between the same descriptors to 1
                      # to avoid a division by zero error
        dists = 1 / (dists ** 2)
        dists[dists<threshold] = 0
        dists[i] = 0 # re-set the "dist" between the same descriptors to 0
        adj_mat.append(dists)
        
        n = np.nonzero(dists)[0]
        neighbors.append(n)
        
    # ensure function returns numpy arrays, not lists
    adj_mat = np.array(adj_mat)
    neighbors = np.array(neighbors)

    return adj_mat, neighbors

def create_graph(descriptors, threshold):
    """ Creates a list of Node objects and the corresponding adjacency
    matrix to the graph (list of Node objects).

    Parameters
    ----------
    descriptors : numpy.ndarray
                  Matrix of (512,) descriptor vectors for these pictures
    threshold : double
                Cosine distance that is considered "too far" for an edge
                to exist between two nodes

    Returns
    -------
    Tuple[list, numpy.ndarray]
        The graph of Nodes and corresponding adjacency matrix
    """
    graph = []
    adj_matrix, neighbors = find_neighbors(descriptors, threshold)
    
    for i in range(len(descriptors)):
        desc_node = Node(i, neighbors[i], descriptors[i])
        graph.append(desc_node)

    return graph, adj_matrix
        
def whisper_iter(graph, adjacent):
    """ Performs one iteration of the whisper algorithm and updates the
    node's label accordingly

    Parameters
    ----------
    graph : list
            List of Node objects for each picture
    adjacent : numpy.ndarray
               Adjacency matrix for Nodes in "graph"

    Returns
    -------
    boolean
        true if a label was changed, false if a label remained the same
    """

    nodes = np.array(graph) # converted to numpy array to enable vectorization
    node = nodes[np.random.randint(0, len(nodes))]
    neighbors = nodes[list(node.neighbors)]
    if len(neighbors) == 0:
#         print("none")
        return True
    
    weights = {}
    for n in neighbors:
        label = n.label
        if label not in weights:
            weights[label] = 0
        weights[label] += adjacent[node.id][n.id]

    max_weight = max(weights.values())
    best_all = [key for key in weights.keys() if weights[key]==max_weight]
    best = best_all[np.random.randint(0, len(best_all))]
    changed = not (node.label==best)
    node.label = best
    return changed

def whisper_algorithm(threshold, graph, adjacent):
    """ Performs the whisper algorithm a maximum of "threshold" times and
    returns a list/dict of the clusters

    Parameters
    ----------
    threshold : int
                Max number of times to perform the whisper algorithm
    graph : list
            List of Node objects for each picture
    adjacent : numpy.ndarray
               Adjacency matrix for Nodes in "graph"

    Returns
    -------
    list
        list of lists (clusters)
    """

    labels_same = 0
    for i in range(threshold):
        changed = whisper_iter(graph, adjacent)
        if not changed:
            labels_same += 1
        else:
            labels_same = 0
            
        if labels_same > 3: ## how to determine convergence
            break
    
    clusters = {}
    for node in graph:
        label = node.label
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node.id)
    # return clusters ## can also return a dict
    return list(clusters.values())
