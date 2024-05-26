from platform import node
import sys
import time
from traceback import print_list
import traceback
from typing import List, Any, Union
import random
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import itertools
from itertools import combinations, permutations
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
                
                #print("EDPs to source :", edps_cp_to_s)
                
                edps_cp_to_d.sort(key=len)
                
                #print("EDPs to destination :", edps_cp_to_d)
                
                
                
                #and build trees out of the longest_edps_cp_s and the longest_edps_cp_d
                
                faces_cp_to_s = one_tree_with_random_checkpoint(cp,source,graph,edps_cp_to_s[len(edps_cp_to_s)-1], True)
                
                tree_cp_to_d = one_tree_with_random_checkpoint(cp,destination,graph,edps_cp_to_d[len(edps_cp_to_d)-1], False)
                
                

                #bc the tree cp->s got build reverse direction the edges need to be reversed again
    
    
                #data structure to give the needed information for the routing (edps, trees, checkpoint)
                
                paths[source][destination] = {
                                                'cp': cp,
                                                'faces_cp_to_s': faces_cp_to_s, 
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
        
        #the current tree has:
        # source = cp (from the global graph)
        # destination = source (from the global graph)
        
        
        connect_leaf_to_destination(tree,source,destination)
        
        #in order to find and traverse faces the tree need to be an undirected graph
        undirected_tree = tree.to_undirected()
        
        tree = undirected_tree
        
        is_planar, P = nx.check_planarity(tree)
        
        # Generate planar layout from the planar embedding
        planar_pos = nx.planar_layout(P)
        
        faces = find_faces(P,planar_pos )
        
        
        print("Faces in OneTree Aglorithm : ", faces)
        
        return faces


    else: #if the tree build is for cp->d nothing is changed
    
        rank_tree(tree , source,longest_edp)
    
        connect_leaf_to_destination(tree, source, destination)
    
        tree.add_edge(longest_edp[len(longest_edp)-2],destination)
    
        #add 'rank' property to the added destinaton, -1 for highest priority in routing
        tree.nodes[destination]["rank"] = -1
        
        return tree    
        
    #end if
    
    

# Find all the faces of a planar graph
def find_faces(G, pos):
    
    #print(G)
    
    #input(" ")
    
    face_nodes = ()
    
    half_edges_in_faces = set()

    faces = []


    for node in G.nodes:

        for dest in nx.neighbors(G, node):

            # check every half edge of node if it is in a face
            if (node, dest) not in half_edges_in_faces:

                # This half edge has no face assigned
                found_half_edges = set()

                try:
                    face_nodes = G.traverse_face(node, dest, found_half_edges)

                except Exception as e:

                    nx.draw(G, pos, with_labels=True, node_size=700, node_color="red", font_size=8)

                    plt.show()
                    
                    traceback.print_exc()
                    
                    print(f"An unexpected error occurred: {e}")
                    
                half_edges_in_faces.update(found_half_edges)

                # Create a subgraph for the face
                face_graph = G.subgraph(face_nodes).copy()

                # Add positions to nodes in the face graph
                for face_node in face_graph.nodes:
                    face_graph.nodes[face_node]['pos'] = pos[face_node]

                faces.append(face_graph)


    #ganz am ende muss der ganze Graph noch rein um die imaginäre Kante in jedem Durchlauf zu bilden
    #und dann immer die Schnittpunkte zu bestimmen
    
    graph_last = G.copy()
    
    for node in graph_last:
        
        graph_last.nodes[node]['pos'] = pos[node]
        
    faces.append(graph_last)
    
    
    
    #print("Faces : ")
    #for face in faces[:-1]:
    #    print(list(face))
        
    #nx.draw(G, pos, with_labels=True, node_size=1200, node_color="green", font_size=9)
    
    #plt.show()
    print("Faces in END of find_faces : ", faces)
    
    return faces

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