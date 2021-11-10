from typing import Optional, Callable, Union
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# https://microsoft.github.io/graspologic/tutorials/simulations/erdos_renyi.html
class ErdosRenyiNetwork:
    def __init__(self, loop:bool=False, directed:bool=False):
        """
        ...
        """
        self.__loop     = loop
        self.__directed = directed
    
    def __call__(self, n:int, Mp:Union[int,float], wFunc:Union[Callable,int,float]=1, args:dict=None, seed:Optional[int]=None):
        if isinstance(Mp,float):
            assert 0 <= Mp <= 1, 'Probability must be between [0,1]'
            
            np.random.seed(seed)
            p    = Mp
            mask = p < np.random.rand(n,n)
    
        elif isinstance(Mp,int):
            table    = {(True ,True ):('n^2'   ,n*n    ), (True ,False):('n(n+1)/2',n*(n+1)/2),
                        (False,True ):('n(n-1)',n*(n-1)), (False,False):('n(n-1)/2',n*(n-1)/2)}            
            msg, lim = table[(self.__loop, self.__directed)]            
            assert 0 <= Mp < lim, f'Number of edges must be between (0, {lim}), {msg}'
            
            np.random.seed(seed)            
            M = Mp
            
            adj   = np.zeros((n,n))
            table = {(True ,True ): np.where(adj == 0)      , (True ,False): np.triu_indices(n,k=0),
                     (False,True ): np.where(np.eye(n) != 0), (False,False): np.triu_indices(n,k=1)}
            
            rows, cols  = table[(self.__loop, self.__directed)]
            choices     = np.random.choice(len(rows), size=M, replace=False)
            sample      = (rows[choices], cols[choices])                
            adj[sample] = 1                
            mask = adj != 1
        
        else: raise TypeError('Argument Mp must be either integer or float type')
        
        if isinstance(wFunc,(float,int)): wArr = np.full((n,n),wFunc)
        else                            : wArr = wFunc(size=(n,n),**args)
        
        adj   = np.where(mask,0,wArr)
        
        if not self.__loop    : np.fill_diagonal(adj, 0)
        if not self.__directed: adj = np.triu(adj) + np.triu(adj,1).T
        return adj
        
class BarabasiAlbertNetwork:
    def __init__(self, m:int, m0:int=2):
        """
        @param: m0: Initial number of nodes
        @param: m : New edges created by a newly added node
        """
        assert 1 < m <= m0, 'm must be between (1, m0]'
        self.__m0 = m0
        self.__m  = m
    
    def __call__(self, M, seed=None):
        np.random.seed(seed)
        adj = np.zeros((M,M))
        ii  = np.random.choice(self.__m0,size=self.__m,replace=False)
        
        adj[self.__m0,ii] = adj[ii,self.__m0] = 1
        
        
        for i0 in range(self.__m0+1, M):
            p  = adj[:i0,:i0].sum(axis=1) / adj[:i0,:i0].sum()
            ii = np.random.choice(i0,size=self.__m,replace=False, p=p)
            adj[i0,ii] = adj[ii,i0] = 1
        
        return adj
        
def plot_network(adj):
    nrow, ncol = adj.shape
    rows, cols = np.where(adj != 0)

    nodes = [f'n{num}' for num in range(nrow)]
    edges = [(nodes[i], nodes[j]) for (i, j) in zip(rows, cols)]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
        
    degs = [d for (n, d) in sorted(G.degree(), key=lambda arg:-arg[1])]    
    fig  = plt.figure("Degree of a random graph", figsize=(8, 8))
    grid = fig.add_gridspec(3, 3)
    
    ax0  = fig.add_subplot(grid[0:2,:])    
    pos  = nx.spring_layout(G)
    nopt = {"node_color":"red","node_size":30}
    eopt = {"edge_color":"black","width": 0.75,"alpha":0.85}
    nx.draw_networkx_nodes(G, pos, ax=ax0, **nopt)
    nx.draw_networkx_edges(G, pos, ax=ax0, **eopt)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()
    
    ax1  = fig.add_subplot(grid[2:3,0])
    ax1.plot(degs, "b-.", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    
    ax2  = fig.add_subplot(grid[2:3,1])
    ax2.matshow(adj,cmap="bone")
    ax2.set_title("Adjacency Matrix")
    ax2.set_axis_off()   
    
    ax3  = fig.add_subplot(grid[2:3,2])
    ax3.bar(*np.unique(degs, return_counts=True))
    ax3.set_title("Degree histogram")
    ax3.set_xlabel("Degree")
    ax3.set_ylabel("# of Nodes") 

    fig.tight_layout()
    plt.show()