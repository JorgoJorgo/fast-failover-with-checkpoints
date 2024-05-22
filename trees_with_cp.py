from platform import node
import sys
import time
from traceback import print_list
from typing import List, Any, Union
import random
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import itertools

from arborescences import *


#################################################### ONETREE WITH CHECKPOINT ######################################################

####################################################################################################################################


#function to generate the 2 trees for each s->d pair (s->cp & cp->d)
#each tree gets generated by expanding the longest edp of each pair
from trees import all_edps, connect_leaf_to_destination, rank_tree, remove_redundant_paths


def one_tree_with_random_checkpoint_pre(graph):
    debug = False
    paths = {}
    
    for source in graph.nodes:

        for destination in graph.nodes:
            
            if source != destination:
                
                if source not in paths:
                    paths[source] = {}
                
                #now compute the chosen checkpoint  
                #first get the longest edp s->d    
                edps = all_edps(source, destination, graph)
                
                edps.sort(key=len)
                
                longest_edp = edps[len(edps)-1]

                #then select the middle node of the longest_edp
                
                print("Longest EDP : " , longest_edp)
                
                cp = longest_edp[ int(len(longest_edp)/2)]
                
                print("Source :", source)
                
                print("Checkpoint :", cp)
                
                print("Destination :", destination)
                
                
                
                #then get the edps + longest_edps_cp_s and the longest_edps_cp_d
                
                edps_cp_to_s = all_edps(cp, source, graph)
                
                edps_cp_to_d = all_edps(cp, destination, graph)
                
                edps_cp_to_s.sort(key=len)
                
                print("EDPs to source :", edps_cp_to_s)
                
                edps_cp_to_d.sort(key=len)
                
                print("EDPs to destination :", edps_cp_to_d)
                
                
                
                #and build trees out of the longest_edps_cp_s and the longest_edps_cp_d
                
                tree_cp_to_s = one_tree_with_random_checkpoint(cp,source,graph,edps_cp_to_s[len(edps_cp_to_s)-1], True)
                
                tree_cp_to_d = one_tree_with_random_checkpoint(cp,destination,graph,edps_cp_to_d[len(edps_cp_to_d)-1], False)
                
                
                ####################################################################################
                if(debug):
                    # print trees for debug

                    pos_s = nx.spring_layout(tree_cp_to_s)  
                    root_position_s = (0, 1)
                    level_height_s = 0.5

                    
                    pos_s[source] = (root_position_s[0] - 0.5, root_position_s[1] - level_height_s)
                    pos_s[destination] = (root_position_s[0] + 0.5, root_position_s[1] - level_height_s)

                    pos_s[cp] = (1, 0)

                    plt.figure(figsize=(7, 7))
                    nx.draw(tree_cp_to_s, pos=pos_s, with_labels=True)
                    plt.title('tree_cp_to_s')
                    plt.show()

                    pos_d = nx.spring_layout(tree_cp_to_d) 
                    root_position_d = (0, 1)
                    level_height_d = 0.5

                    pos_d[source] = (root_position_d[0] - 0.5, root_position_d[1] - level_height_d)
                    pos_d[destination] = (root_position_d[0] + 0.5, root_position_d[1] - level_height_d)

                    pos_d[cp] = (-1, 0)

                    plt.figure(figsize=(7, 7))
                    nx.draw(tree_cp_to_d, pos=pos_d, with_labels=True)
                    plt.title('tree_cp_to_d')
                    plt.show()
                #####################################################################################

                #bc the tree cp->s got build reverse direction the edges need to be reversed again
    
    
                #data structure to give the needed information for the routing (edps, trees, checkpoint)
                
                paths[source][destination] = {
                                                'cp': cp,
                                                'tree_cp_to_s': tree_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'tree_cp_to_d': tree_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                            }
                    
            
    
    return paths


#this algorithm builds a tree for the one_tree_with_checpoint function
#the tree has the source as root of the tree and every leaf is connected with the destination at the end
#the tree is build by expanding the longest edp as much as possible and only keeping the paths that lead to the destination

#special: because the second tree that is required to build by the one_tree_with_random_checkpoint_pre is the tree cp->s
#its directed edges need to flipped (arg: reverse)
def one_tree_with_random_checkpoint(source, destination, graph, longest_edp, reverse):
    
    tree = nx.DiGraph()
    assert source == longest_edp[0] , 'Source is not start of edp'
    tree.add_node(source) # source = longest_edp[0]

    # We need to include the EDP itself here
    for i in range(1,len(longest_edp)-1): # -2 since we don't want to insert the destination
        tree.add_node(longest_edp[i])
        tree.add_edge(longest_edp[i-1],longest_edp[i])

    pathToExtend = longest_edp
    
    for i in range(0,len(pathToExtend)-1): # i max 7
        
        nodes = pathToExtend[:len(pathToExtend) -2]
        it = 0 # to get the neighbors of the neighbors
        while it < len(nodes):

            neighbors = list(nx.neighbors(graph, nodes[it]))
            for j in neighbors:
                if (not tree.has_node(j)) and (j!= destination): #not part of tree already and not the destiantion
                    nodes.append(j)
                    tree.add_node(j) #add neighbors[j] to tree
                    tree.add_edge(nodes[it], j) # add edge to new node
                #end if
            #end for
            it = it+1
        #end while
    #end for
    

    changed = True 
    while changed == True: #keep trying to shorten until no more can be shortened 
        old_tree = tree.copy()
        remove_redundant_paths(source, destination, tree, graph)
        changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.

    #before ranking the tree, if the the is build for cp->s the edges need to be flipped
    if(reverse):
        
        
        longest_edp = list(reversed(longest_edp))
        
         # Flipping all edges if 'reverse' is True
         
        tree_copy = tree.copy()
        
        for u, v in tree.edges():
            
            tree_copy.remove_edge(u,v)
            
            tree_copy.add_edge(v, u)
            
        #end for
        
        tree = tree_copy
        
        #after flipping the edges the source and destination need to change too
        
        old_source = source
        
        source = destination
        
        destination = old_source

        print(list(graph.neighbors(source)))

        ######################################################################################
        plot_tree_with_highlighted_nodes(tree,source,destination,list(graph.neighbors(source)))
        ######################################################################################
        
        rank_tree_for_cp_algorithms(tree , source,longest_edp)
    
        connect_leaf_to_destination(tree, source, destination)
    
        tree.add_edge(longest_edp[len(longest_edp)-2],destination)
    
        #add 'rank' property to the added destinaton, -1 for highest priority in routing
        
        tree.nodes[destination]["rank"] = -1
    
    else: #if the tree build is for cp->d nothing is changed
    
        rank_tree(tree , source,longest_edp)
    
        connect_leaf_to_destination(tree, source, destination)
    
        tree.add_edge(longest_edp[len(longest_edp)-2],destination)
    
        #add 'rank' property to the added destinaton, -1 for highest priority in routing
        tree.nodes[destination]["rank"] = -1
        
        
    #end if
    
    return tree

def rank_tree_for_cp_algorithms(tree, source, edp):
    # Initialize all nodes with a very large rank
    nx.set_node_attributes(tree, float('inf'), name="rank")

    # Extract edges of the EDP
    edp_edges = [(edp[i-1], edp[i]) for i in range(1, len(edp))]

    # Perform topological sorting
    sorted_nodes = list(nx.topological_sort(tree))

    # Assign ranks based on the sorted order
    for node in sorted_nodes:
        # Skip source node
        if node == source:
            continue

        # Determine minimum rank among children
        min_rank = min([tree.nodes[child]["rank"] for child in tree.successors(node)])

        # Assign rank to the node
        tree.nodes[node]["rank"] = min_rank + 1 if min_rank != float('inf') else 0

    # Adjust ranks to prioritize EDP nodes
    for node in sorted_nodes:
        # Skip source node
        if node == source:
            continue

        # Determine successors of the node
        children = list(tree.successors(node))

        for child in children:
            if (node, child) in edp_edges:
                # Sort children by rank
                children.sort(key=lambda x: tree.nodes[x]["rank"])

                # Assign minimum rank to EDP node
                tree.nodes[child]["rank"] = tree.nodes[children[0]]["rank"]

                # Update ranks of other children
                for other_child in children[1:]:
                    tree.nodes[other_child]["rank"] += 1

                break  # Exit loop after processing EDP node



def plot_tree_with_highlighted_nodes(tree, source, destination, highlighted_nodes):
    # Generate positions using spring layout
    pos_s = nx.spring_layout(tree)  
    root_position_s = (0, 0)
    level_height_s = 0.5

    # Set manual positions for source and destination
    pos_s[source] = (root_position_s[0] - 0.5, root_position_s[1] - level_height_s)
    pos_s[destination] = (root_position_s[0] + 0.5, root_position_s[1] - level_height_s)

    # Determine node colors
    node_colors = []
    for node in tree.nodes():
        if node == source:
            node_colors.append('red')
        elif node == destination:
            node_colors.append('green')
        elif node in highlighted_nodes:
            node_colors.append('yellow')
        else:
            node_colors.append('skyblue')

    # Create a figure
    plt.figure(figsize=(10, 10))
    
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Source'),
        Patch(facecolor='green', edgecolor='black', label='Destination'),
        Patch(facecolor='yellow', edgecolor='black', label='Highlighted'),
        Patch(facecolor='skyblue', edgecolor='black', label='Other Nodes')
    ]

    # Draw the graph
    nx.draw(tree, pos=pos_s, with_labels=True, node_color=node_colors)

    # Set the title
    plt.title(f"{source} to {destination}")
    plt.legend(handles=legend_elements, loc='upper left')
    # Display the plot
    plt.show()