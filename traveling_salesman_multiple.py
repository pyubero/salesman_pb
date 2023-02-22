# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:11:35 2023

@author: Pablo Yubero
"""

import numpy as np
import geopandas as gpd
from pyrosm import OSM, get_data
import networkx as nx
import osmnx as ox
from pyrosm.data import sources
import itertools

from tqdm import tqdm
from matplotlib import pyplot as plt



def random_coordinates(N, bbox ):
    xmin, xmax = bbox[0], bbox[2]
    ymin, ymax = bbox[1], bbox[3]
    
    x_coord = xmin + (xmax-xmin)*np.random.rand(N)
    y_coord = ymin + (ymax-ymin)*np.random.rand(N)
    return x_coord, y_coord

def get_route(start_node, end_node, graph, weight='travel_time'):
    '''
    Computes the shortest path between start_ and end_ nodes in a graph.
    
    Parameters
    ----------
    start_node : str
        Starting node ID according to the graph.
    end_node : str
        End node ID according to the graph.
    graph : DiGraph
        Graph representing all possible routes.
    weight : str, optional
        Shortest route can be determined according to specific attributes of
        the graph edges. The default is 'travel_time'.

    Returns
    -------
    route : list
        Sequence of nodes that conforms the shortest route.
    total_route_length : float
        Total route length in km.
    route_travel_time : float
        Total estimated travel time in h.

    '''
    route = nx.shortest_path(graph, start_node, end_node, weight=weight )

    edge_lengths = ox.utils_graph.get_route_edge_attributes(graph, route, 'length') 
    edge_travel_time = ox.utils_graph.get_route_edge_attributes( graph, route, 'travel_time') 

    total_route_length = round(sum(edge_lengths), 1)
    route_travel_time  = round(sum(edge_travel_time)/60, 2)

    return route, total_route_length, route_travel_time


def routes_length( graph, routes, unit='km'):
    '''
    Computes the length of a route, preferably in km. The route needs to be
    a list of adjacent node ids in the graph. Use in combination with get_route().
    '''
    if unit=='km':
        alpha = 1000
        
    length = [ np.sum( ox.utils_graph.get_route_edge_attributes(graph, route, 'length') ) for route in routes]
    return np.sum(length)/alpha


def sequence_to_routes(all_routes, travel_order):
    '''
    Computes the shortest path between a sequence of nodes (not node ids!) but 
    node indices.
    
    <W> The sequence of graph node ids is not inverted when they should.
    
    travel_order : list
        Order of target nodes indices. For example, travel_order=[0,3,1,2,0]
        would represent a circular route that visits three target locations.
    
    all_routes : dict 
        Contains the routes between any two target locations. The keys are
        start_node_idx-end_node_idx, where start_node_idx < end_node_idx.
        For example, the previous sequence is made of "0-3", "1-3","1-2" and "0-2".
    '''
    routes = []
    
    for jj in np.arange(1, len(travel_order)):
        subidc = travel_order[(jj-1):(jj+1)]   
        name = "{}-{}".format( min(subidc), max(subidc) )
        routes.append( all_routes[name])
    return routes


def compute_distances_and_routes(nodes, graph):   
    '''
    Computes a matrix with all distances between target nodes in a graph, and also
    the shortest routes between any two target nodes (in a dictionnary).
    '''
    dist = np.zeros( (len(nodes), len(nodes) ) )
    routes = {}
    for ii in tqdm(range( len(nodes))):
        for jj in np.arange(ii+1,len(nodes)):
            route, length, time = get_route( nodes[ii], nodes[jj], graph)
            
            dist[ii,jj] = length
            dist[jj,ii] = length
            
            name = "{}-{}".format(ii,jj)
            routes.update( { name : route } )
    return dist, routes

def shortest_path_greedy(dist, starting_idx, n_salesmen=1, circular=False):
    def circularize(order):
        if not circular:
            return order
        
        for salesman in range(n_salesmen):
            idx = np.argwhere(np.isnan(order[salesman,:]))[0,0]
            order[salesman,idx] = starting_idx
        return order
        
    def prune_nans(order):
        floats = [ list(a[a<np.inf]) for a in order]
        return [ list(map(int, a)) for a in floats]
    
    
    order = np.nan*np.zeros( (n_salesmen,len(dist)+1))
    order[:,0] = starting_idx
    
    for jj in range(len(dist)):
        for salesman in range(n_salesmen):
            to_be_visited = [a for a in range(len(dist)) if a not in order ]
            
            if len(to_be_visited)==0: 
                print('Finished!')
                return prune_nans( circularize(order) )
            
            old_loc = int( order[salesman][jj] )
            k2b = np.argmin( dist[old_loc, :][to_be_visited])
            new_loc = to_be_visited[k2b]
            
            order[salesman][jj+1]=new_loc


def SA_next_state(order, new_order, T, graph):
    E0 = compute_energy(order, graph, "quadratic")
    E1 = compute_energy(new_order, graph, "quadratic")
    
    DE = np.exp(-(E1-E0)/T)
    if np.random.rand()<= DE:
        return new_order, E1
    else:
        return order, E0
    
def compute_energy(routes, graph, type_):
    seqs = [ sequence_to_routes(all_routes, rt) for rt in routes]
    lengths= np.array([routes_length(graph, s) for s in seqs])
    # lengths = np.array([routes_length( graph, sequence_to_routes(all_routes, route ) ) for route in routes])
    # print(lengths)

    if type_=="linear":    
        return np.sum(lengths)
    elif type_ =="quadratic":
        return np.sqrt(np.sum(lengths**2))

def k_permutation(order, circular=False ):
    
    idc = range(len(order))
    if circular: 
        idc = idc[1:-1]
    idc = np.random.permutation(idc )
    
    new_order = order.copy()
    new_order[ idc[0] ] = order[idc[1]]
    new_order[ idc[1] ] = order[idc[0]]
    return new_order

def get_temperature(n, nmax):
    # return 1-n/nmax
    a = nmax**2 / 200
    b = n**2
    return 1/(1 + (b/a) )

def randint_weighted(v, w):
    x = [ [v[jj]]*w[jj] for jj in range(len(v))]
    return np.random.permutation([ a for b in x for a in b])[0]

def find_neighbor_state(routes, prob=0.5, circular=False):
    import copy
    # prop : float : The probability of permutations
    starting_idx = routes[0][0]
    order = copy.deepcopy(routes)
    
    # Starting and ending indices (if circular) are immutable
    _=[ rd.remove(starting_idx) for rd in order]
    if circular:
        _=[ rd.remove(starting_idx) for rd in order]
            
    nsalesmen = len(order)
    salesmen = set(list(range(nsalesmen)))
    salesman_from = randint_weighted( range(nsalesmen), [len(rd)-1 for rd in order])
    
    if np.random.rand()<prob:
        # translocations 
        value = order[salesman_from].pop( np.random.randint(len(order[salesman_from])))
        salesman_to = np.random.permutation( list(salesmen-set([salesman_from,]) ))[0]
        idx_to = [np.random.randint(len(order[salesman_to])) if len(order[salesman_to])>0 else 0 ][0]
        order[salesman_to].insert(idx_to,value)
        
    else:
        # permutations
        order = [ k_permutation(rd, False) if jj==salesman_from else rd for jj, rd in enumerate(order)  ]
    
    # Recover the starting and possibly the ending locations too
    _ = [rd.insert(0, starting_idx) for rd in order ]
    if circular:
        _ = [rd.extend([starting_idx,]) for rd in order ]
    
    return order



# Input variables
x_, y_ = random_coordinates(N=40, bbox = [-3.7, 40.40, -3.65, 40.45] )
map_name = "Madrid"
route_type = "driving" #driving, walking, cycling
temp_data_folder = "TEMP"
circular = True
N_SALESMEN = 4
starting_idx=0

# Compute the bbox for the map
# ... and extend it by some margin
xspan = np.max(x_)-np.min(x_)
yspan = np.max(y_)-np.min(y_)
margin = np.max([xspan, yspan])/3
bbox_map= [ np.min(x_)-margin, np.min(y_)-margin, np.max(x_)+margin, np.max(y_)+margin]

# Download map
fp = get_data( map_name , directory = temp_data_folder)
osm = OSM(fp ,bounding_box = bbox_map)

# Create network and graph
nodes,  edges = osm.get_network(nodes=True, network_type=route_type) # ~2mins
graph = osm.to_graph(nodes, edges,  graph_type="networkx") # ~5mins
graph = ox.add_edge_speeds(graph )
graph = ox.add_edge_travel_times(graph )

target_nodes = list(np.unique(ox.nearest_nodes( graph, x_, y_ )))

# Compute distance matrix
dist, all_routes = compute_distances_and_routes(target_nodes, graph)


###################
# Greedy algorithm
shortest_path = shortest_path_greedy(dist, starting_idx, n_salesmen=N_SALESMEN, circular=circular)
routes_greedy = [ sequence_to_routes(all_routes, route) for route in shortest_path ]
total_length = [ routes_length( graph,  route) for route in routes_greedy ]

print("")
print("Distance travelled in the greedy path {:1.1f} km".format(np.sum(total_length)))
print("RSSD: {:1.1f} km".format(np.sqrt(np.sum(np.array(total_length)**2))))

    
# #####################
# # Simulated Annealing
all_E=[]
max_temp = 1
order = shortest_path.copy()
nmax = 4000
for n in tqdm(range(nmax)):
    T = max_temp*get_temperature(n, nmax)+0.001
    new_order = find_neighbor_state(order, prob=0.5, circular=circular)
    order, E = SA_next_state(order, new_order, T, graph)
    all_E.append(E)

plt.plot(all_E)
plt.show()

order_sa  = order.copy()
routes_sa = [ sequence_to_routes(all_routes, rd) for rd in order_sa]
print("")
print("By using the new path you drive {:1.1f} units".format( np.min(all_E) - compute_energy(shortest_path, graph, type_="quadratic")))


# #################
# # Random sequence
# # .. to test efficiency
# mcmax=999
# t_mc = []
# for jj in tqdm( range(mcmax) ):
#     sequence_mc = np.random.permutation( range(len(target_nodes))[1:])
#     sequence_mc = np.append( 0, sequence_mc)
#     if circular:
#         sequence_mc = np.append( sequence_mc, 0)

#     routes_mc = sequence_to_routes(all_routes, sequence_mc)
#     t_mc.append(routes_length(graph, routes_mc) )

# mc_mean = np.mean(t_mc)
# mc_std  = np.std(t_mc)
# print("")
# print("length estimated by random: {:1.1f} +/- {:1.1f} km".format(mc_mean, mc_std) )



# plt.figure(figsize=(5,3), dpi=300)
# plt.plot(all_E,color=[0.2, 0.2, 0.2])
# plt.hlines(total_length, 0, nmax, 'r')
# plt.hlines( mc_mean, 0, nmax,'b')
# plt.xlabel("Sim.Ann. time (steps)")
# plt.ylabel('Traveling distance (km)')
# plt.legend(("Sim.Ann.","Greedy alg.", "Random"))
# plt.tight_layout()
# plt.savefig('graph.png', dpi=300)

# for n in range(N_SALESMEN):
#     fig, ax = ox.plot_graph_routes(graph, routes_sa[n], 
#                                     route_linewidth=2,
#                                     orig_dest_size = 50,
#                                     node_size=0,
#                                     edge_linewidth=0.5,
#                                     save=False,
#                                     dpi=300,
#                                     filepath="./map_greedy_{}.png".format(n)
#                                     )


from matplotlib import cm                                    
def plot_route(route, ax, **kwargs):
    for rt in route:
        assert False
        edges[edges["u"].isin(rt)].plot(ax=ax, **kwargs)


colors= cm.get_cmap('viridis', N_SALESMEN).colors

fig = plt.figure(figsize=(10,5), facecolor='k', frameon=False, dpi=300)
ax = plt.gca()
ax.set_facecolor('w')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

edges.plot(ax=ax, linewidth=0.2, edgecolor=0.5*np.ones((3,)))

for n in range(N_SALESMEN):
    plot_route( routes_sa[n], ax, edgecolor=colors[n], linewidth=1, alpha=1)

nodes[nodes["id"].isin(target_nodes)].plot(ax=ax, color='r', markersize=3, zorder=9999)
nodes[nodes["id"]==target_nodes[starting_idx]].plot(ax=ax, color='k', markersize=8, zorder=9998)

plt.tight_layout()


