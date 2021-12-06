from typing import Optional, Callable, Union
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# https://microsoft.github.io/graspologic/tutorials/simulations/erdos_renyi.html
# https://en.wikipedia.org/wiki/Power_iteration

class ErdosRenyiNetwork:
    def __init__(self):
        """
        A network class used to generate an undirected adjacency matrix using the Erdős-Rényi model
        """    
    def __call__(self, n:int, p:float, wFunc:Union[Callable,int,float]=1, seed:Optional[int]=None, **args) -> np.ndarray:
        """
        To generate an adjacency matrix
        
        ER = ErdosRenyiNetwork()
        ER(n<int>, p<float>)
        
        this model specifies the number of node, n, and each pair of node has p probability of an edge existing between the two

        Parameters
        ----------
        n     : int
            The number of nodes in the final graph            
        p     : int
            The number of edges in the final graph            
        wFunc : Callable | int | float
            if wFunc is an callable, weights of edges are assigned with function
            if wFunc is an integer or float, weights of edges are assigned with this constant value (default 1)            
        seed  : int, optional
            Initialize the random number generator with given seed (default None)            
        args  : dict, optional
            Arguments to be used in the callable wFunc (default None)

        Raises
        ------
        ValueError
            If p is not float type            
        AssertionError
            If p probability is not between [0,1]
            
        Returns
        -------
        np.ndarray
            Adjacency array representing an undirected graph without loops
        """
        if isinstance(p,float):                                                     # Checking if the given probability threshold is float type
            assert 0 <= p <= 1, 'Probability must be between [0,1]'                 # Checking if the given probability threshold is valid (i.e., between 0% and 100%)
            np.random.seed(seed)                                                    # Fixing random state for reproducibility
            mask = np.random.rand(n,n) < p                                          # Creating a mask which is True when the generated random number is less than the threshold

        else: raise TypeError('p must be float type')                               # Raising TypeError when the probability threshold is not float type

        if isinstance(wFunc,(float,int)): wArr = np.full((n,n),wFunc)               # Creating a constant weight array full of wFunc if the given wFunc is either integer or float type
        else                            : wArr = wFunc(size=(n,n),**args)           # Creating a randomly generated weight array with the the given callable wFunc

        adj = np.where(mask,wArr,0)                                                 # Creating an adjacency array, where adj_ij is 0 (mask_ij is False) or weight_ij (mask_ij is True)
        np.fill_diagonal(adj, 0)                                                    # To remove self-loops, diagonal values are reassigned as 0
        adj = np.triu(adj) + np.triu(adj).T                                         # Lower-triangle of the array is overwritten by the upper-triangle to make it undirected
        return adj

class BarabasiAlbertNetwork:
    def __init__(self):
        """
        A network class used to generate an undirected adjacency matrix using the Barabási-Albert model
        """    
    def __call__(self, m0:int, m:int, M:int, wFunc:Union[Callable,int,float]=1, seed:Optional[int]=None, **args) -> np.ndarray:
        """
        To generate an adjacency matrix
        
        BA = BarabasiAlbertNetwork()
        BA(m0<int>, m<int>, M<int>)
        
        this model specifies the number of node, M, with m0 starting nodes and m edges created by each new node
        
        Parameters
        ----------
        m0    : int
            Initial number of nodes, where m0 >= m            
        m     : int
            The number of edges added to the graph with each new node, where m > 0            
        M     : int
            The number of edges in the final graph            
        wFunc : Callable | int | float
            if wFunc is an callable, weights of edges are assigned with function
            if wFunc is an integer or float, weights of edges are assigned with this constant value (default 1)            
        seed  : int, optional
            Initialize the random number generator with given seed (default None)            
        args  : dict, optional
            Arguments to be used in the callable wFunc (default None)

        Raises
        ------
        AssertionError
            If the condition m0 >= m > 0 is not met
            
        Returns
        -------
        np.ndarray
            Adjacency array representing an undirected graph without loops
        """
        assert m0 >= m > 0, 'The condition m0 >= m > 0 is not met'                  # Checking if the given m0 and m values are valid (i.e., m0 >= m > 0)
        np.random.seed(seed)                                                        # Fixing random state for reproducibility
        nodes = defaultdict(list, {i:[i-1,i+1] for i in range(m0)})                 # The network begins with an initial connected network of m0 nodes
        nodes[0].remove(-1); nodes[m0-1].remove(m0)                                 # Some connections are manually removed since there isn't neither node(-1) nor node(m0) yet        
        
        for i in range(m0,M):                                                       # New nodes are added to the network in each epoch until the graph has M nodes
            ki  = np.array([sum(vals) for vals in nodes.values()]) + 1E-8           # Degree of each node in the graph, added 1E-8 in order to avoid 0 probability
            kij = ki.sum()                                                          # The sum is made over all pre-existing nodes
            pi  = ki / kij                                                          # The probability of that the new node is connected to node i can be calculated as ki / kij
            j   = np.random.choice(list(nodes.keys()), size=m, replace=False, p=pi) # m nodes are selected according to their corresponding pi probabilities
            
            nodes[i].extend(j.tolist())                                             # edges are created between the newly added node and the selected j nodes
            for _ in j: nodes[_].append(i)                                          # Since it is undirected graph, the connections are created from the other node as well

        mask = np.full((M,M), False, dtype=bool)                                    # A mask array (M,M) created to be used for assigning edge weights
        for ni, njj in nodes.items(): mask[ni,njj] = mask[njj,ni] = True            # The mask array values are assigned as True (nodes are connected) or False (nodes are not connected)
        
        if isinstance(wFunc,(float,int)): wArr = np.full((M,M),wFunc)               # Creating a constant weight array full of wFunc if the given wFunc is either integer or float type
        else                            : wArr = wFunc(size=(M,M),**args)           # Creating a randomly generated weight array with the the given callable wFunc

        adj = np.where(mask,wArr,0)                                                 # Creating an adjacency array, where adj_ij is 0 (mask_ij is False) or weight_ij (mask_ij is True)
        return adj

def Adj2Graph(adj:np.ndarray,only_graph:bool=False) -> Union[nx.Graph,dict]:
    nrow, _ = adj.shape
    nodes   = [f'n{num}' for num in range(nrow)]                                    # The graph is assumed to have nrow nodes
    edges   = []                                                                    # A list for storing edges
    weights = []                                                                    # A list for storing edge weights
    
    ij      = C(range(nrow), 2)                                                     # Since the graph is assumed as undirected and without self-loop
    for i, j in ij:                                                                 # only values in the upper-triangle are used
        if adj[i,j] != 0:                                                           # A non-zero value in adjacency array shows that node_i and node_j are connected
            edges.append([nodes[i], nodes[j], adj[i,j]])
            weights.append(adj[i,j])
    
    G = nx.Graph()
    G.add_nodes_from(nodes)                                                         # Stored nodes are added to the undirected graph
    G.add_weighted_edges_from(edges)                                                # Stored edges are added to the undirected graph with their weights
    
    w33, w66 = np.percentile(weights, [33, 66])                                     # An additional step was to divide weights into three parts and then
    eL,eM,eS = [],[],[]                                                             # named as small, medium, and large edges to emphasize the edge weight
    
    for (u,v,d) in G.edges(data=True):
        if   d['weight'] >= w66: eL.append([u,v])
        elif d['weight'] >= w33: eM.append([u,v])
        else                   : eS.append([u,v])
        
    results = {"graph":G,"edges_large":eL,"edges_medium":eM,"edges_small":eS}       # nx.Graph and divided edges are returned separately for further steps
    
    if only_graph: return G                                                         # An option is provided for those only wanting nx.Graph
    else         : return results
    
def plot_multi(adjs, titles,**kwargs):
    assert len(adjs) == (kwargs["nrows"] * kwargs["ncols"])
    
    results   = [Adj2Graph(adj) for adj in adjs]
    fig, axes = plt.subplots(**kwargs)

    nopt  = {"node_color":"red","node_size":30}
    eoptL = {"edge_color":"black","width": 1.00,"alpha":0.80}
    eoptM = {"edge_color":"gray" ,"width": 0.75,"alpha":0.50}
    eoptS = {"edge_color":"gray" ,"width": 0.25,"alpha":0.20}
    
    for result, title, ax in zip(results, titles, axes.ravel()):
        pos = nx.circular_layout(result["graph"])
        nx.draw_networkx_nodes(result["graph"], pos, ax=ax, **nopt)
        nx.draw_networkx_edges(result["graph"], pos, edgelist=result["edges_large" ], ax=ax0, **eoptL)
        nx.draw_networkx_edges(result["graph"], pos, edgelist=result["edges_medium"], ax=ax0, **eoptM)
        nx.draw_networkx_edges(result["graph"], pos, edgelist=result["edges_small" ], ax=ax0, **eoptS)
        ax.set_title(title)
        ax.set_axis_off()
    
    fig.tight_layout()
    plt.show()
    
def plot_single(adj, title, figsize=(10,10)):
    result = Adj2Graph(adj)
    nopt   = {"node_color":"red","node_size":30}
    eoptL  = {"edge_color":"black","width": 1.00,"alpha":0.80}
    eoptM  = {"edge_color":"gray" ,"width": 0.75,"alpha":0.50}
    eoptS  = {"edge_color":"gray" ,"width": 0.25,"alpha":0.20}
       
    fig  = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(3, 2)
    
    ax0  = fig.add_subplot(grid[0:2,:])
    pos  = nx.circular_layout(result["graph"])  
    nx.draw_networkx_nodes(result["graph"], pos, ax=ax0, **nopt)
    nx.draw_networkx_edges(result["graph"], pos, edgelist=result["edges_large" ], ax=ax0, **eoptL)
    nx.draw_networkx_edges(result["graph"], pos, edgelist=result["edges_medium"], ax=ax0, **eoptM)
    nx.draw_networkx_edges(result["graph"], pos, edgelist=result["edges_small" ], ax=ax0, **eoptS)
    ax0.set_title(title,fontweight="bold")
    ax0.set_axis_off()
    
    degs = [d for (n, d) in result["graph"].degree()]    
    ax1  = fig.add_subplot(grid[2,0])
    ax1.bar(*np.unique(degs, return_counts=True))
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes") 
    
    ax2  = fig.add_subplot(grid[2,1])
    im   = ax2.imshow(adj,cmap="magma")
    plt.colorbar(im, ax=ax2)
    ax2.set_title("Adjacency Matrix")
    ax2.set_axis_off()   
    
    fig.tight_layout()
    plt.show()
