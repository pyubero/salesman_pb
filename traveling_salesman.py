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
    get_route() computes the shortest path between two nodes in a graph.
    
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
    if unit=='km':
        alpha = 1000
        
    length = [ np.sum( ox.utils_graph.get_route_edge_attributes(graph, route, 'length') ) for route in routes]
    return np.sum(length)/alpha


def sequence_to_routes(all_routes, travel_order):
    routes = []
    
    for jj in np.arange(1, len(travel_order)):
        subidc = travel_order[(jj-1):(jj+1)]   
        name = "{}-{}".format( min(subidc), max(subidc) )
        routes.append( all_routes[name])
    return routes

def compute_distances_and_routes(nodes, graph):    
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

def shortest_path_greedy(dist, starting_idx, circular = False):
    ddist = dist.copy()    
    ddist[ dist==starting_idx]=np.inf
    
    order = [starting_idx,]     
    for jj in range(len(dist)-1):
        idx = np.argmin( ddist[order[-1],:])
        ddist[:, order[-1]] = np.inf
        ddist[order[-1], :] = np.inf
        
        order.append(idx)
        
    if circular:
        order.append(starting_idx)
        
    return order

def SA_next_state(order, new_order, T):
    E0 = compute_energy(order)
    E1 = compute_energy(new_order)
    
    DE = np.exp(-(E1-E0)/T)
    if np.random.rand()<= DE:
        return new_order, E1
    else:
        return order, E0
    
def compute_energy(order):
    return routes_length( graph,  sequence_to_routes(all_routes, order ) )

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
    return 1-n/nmax



# Input variables
x_, y_ = random_coordinates(N=20, bbox = [-3.75, 40.35, -3.65, 40.45] )
map_name = "Madrid"
route_type = "driving" #driving, walking, cycling
temp_data_folder = "TEMP"
circular = True

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
shortest_path = shortest_path_greedy(dist, 0, circular=circular)
routes_greedy = sequence_to_routes(all_routes, shortest_path )    
total_length = routes_length( graph,  routes_greedy)
print("")
print("Distance traveled using the greedy path {:1.1f} km".format(total_length))

#####################
# Simulated Annealing
all_E=[]
max_temp = 5
order = shortest_path.copy()
nmax = 2_000
for n in tqdm(range(nmax)):
    T = max_temp*get_temperature(n, nmax)+0.1
    new_order = k_permutation(order,circular)
    order, E = SA_next_state(order, new_order, T)
    all_E.append(E)

order_sa  = order.copy()
routes_sa = sequence_to_routes(all_routes, order)
print("")
print("By using the new path you drive {:1.1f} km".format( np.min(all_E) - total_length))

#################
# Random sequence
# .. to test efficiency
mcmax=999
t_mc = []
for jj in tqdm( range(mcmax) ):
    sequence_mc = np.random.permutation( range(len(target_nodes))[1:])
    sequence_mc = np.append( 0, sequence_mc)
    if circular:
        sequence_mc = np.append( sequence_mc, 0)

    routes_mc = sequence_to_routes(all_routes, sequence_mc)
    t_mc.append(routes_length(graph, routes_mc) )

mc_mean = np.mean(t_mc)
mc_std  = np.std(t_mc)
print("")
print("length estimated by random: {:1.1f} +/- {:1.1f} km".format(mc_mean, mc_std) )



# plt.figure(figsize=(5,3), dpi=300)
# plt.plot(all_E,color=[0.2, 0.2, 0.2])
# plt.hlines(total_length, 0, nmax, 'r')
# plt.hlines( mc_mean, 0, nmax,'b')
# plt.xlabel("Sim.Ann. time (steps)")
# plt.ylabel('Traveling distance (km)')
# plt.legend(("Sim.Ann.","Greedy alg.", "Random"))
# plt.tight_layout()
# plt.savefig('graph.png', dpi=300)


# fig, ax = ox.plot_graph_routes(graph, routes_greedy, 
#                                route_linewidth=2,
#                                orig_dest_size = 50,
#                                node_size=0,
#                                edge_linewidth=0.5,
#                                save=True,
#                                dpi=300,
#                                filepath="./map_greedy.png"
#                                )

# ox.plot_graph_routes(graph, routes_sa,
#                      route_color='c',
#                      route_linewidth=2,
#                      orig_dest_size = 50,
#                      node_size=0,
#                      edge_linewidth=0.5,
#                      save=True,
#                      dpi=300,
#                      filepath="./map_simann.png"
#                      )
