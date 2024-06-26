import sys
from typing import List, Any, Union

import networkx as nx
import numpy as np
import itertools
import random
import time
import glob
from arborescences import reset_arb_attribute
from objective_function_experiments import *
from trees import one_tree_pre
from trees_with_cp import one_tree_with_random_checkpoint_pre
from routing import RouteOneTree, RouteOneTreeCLUSTERED, RouteWithOneCheckpointOneTree, RouteWithOneCheckpointOneTreeCLUSTERED, SimulateGraph, SimulateGraphClustered, Statistic
DEBUG = True

# Data structure containing the algorithms under
# scrutiny. Each entry contains a name and a pair
# of algorithms.
#
# The first algorithm is used for any precomputation
# to produce data structures later needed for routing
# on the graph passed along in args. If the precomputation
# fails, the algorithm must return -1.
# Examples for precomputation algorithms can be found in
# arborescences.py
#
# The second algorithm decides how to forward a
# packet from source s to destination d, despite the
# link failures fails using data structures from precomputation
# Examples for precomputation algorithms can be found in
# routing.py
#




graph = None

algos = {
         'One Tree Clustered': [one_tree_pre, RouteOneTreeCLUSTERED],
         'One Tree Checkpoint Clustered':[one_tree_with_random_checkpoint_pre,RouteWithOneCheckpointOneTreeCLUSTERED],
         #'One Tree': [one_tree_pre, RouteOneTreeCLUSTERED],
         #'One Tree Checkpoint':[one_tree_with_random_checkpoint_pre,RouteWithOneCheckpointOneTreeCLUSTERED],
         #'SquareOne':[PrepareSQ1,RouteSQ1]
         }

# run one experiment with graph g
# out denotes file handle to write results to
# seed is used for pseudorandom number generation in this run
# returns a score for the performance:
#       if precomputation fails : 10^9
#       if success_ratio == 0: 10^6
#       otherwise (2 - success_ratio) * (stretch + load)
def one_experiment(g, seed, out, algo):
    [precomputation_algo, routing_algo] = algo[:2]
    if DEBUG: print('experiment for ', algo[0])

    # precomputation
    reset_arb_attribute(g)
    random.seed(seed)
    t = time.time()
    precomputation = precomputation_algo(g)
    print('Done with precomputation algo')
    pt = time.time() - t
    if precomputation == -1:  # error...
        out.write(', %f, %f, %f, %f, %f, %f\n' %
                  (float('inf'), float('inf'), float('inf'), 0, 0, pt))
        score = 1000*1000*1000
        return score

    # routing simulation (hier gebe ich den routing algorithmus mit)#################################################################################################################################
    print("Start routing")
    stat = Statistic(routing_algo, str(routing_algo))
    stat.reset(g.nodes())
    random.seed(seed)
    t = time.time()
    print("Before simulate graph")
    #hier sage ich dass ich den routing algorithmus simulieren soll (in stat steht welchen routing algorithmus ich ausführen will))#################################################################################################################################
    SimulateGraphClustered(g, True, [stat], f_num, samplesize, precomputation=precomputation, targeted=True)
    print("After simulate")
    rt = (time.time() - t)/samplesize
    success_ratio = stat.succ/ samplesize
    # write results
    if stat.succ > 0:
        if DEBUG: print('success', stat.succ, algo[0])
        # stretch, load, hops, success, routing time, precomputation time
        out.write(', %i, %i, %i, %f, %f, %f\n' %
                  (np.max(stat.stretch), stat.load, np.max(stat.hops),
                   success_ratio, rt, pt))
        score = (2 - success_ratio) * (np.max(stat.stretch) + stat.load)

    else:
        if DEBUG: print('no success_ratio', algo[0])
        out.write(', %f, %f, %f, %f, %f, %f\n' %
                  (float('inf'), float('inf'), float('inf'), 0, rt, pt))
        score = 1000*1000
    return score

# run experiments with d-regular graphs
# out denotes file handle to write results to
# seed is used for pseudorandom number generation in this run
# rep denotes the number of repetitions in the secondary for loop
def run_regular(out=None, seed=0, rep=5):
    ss = min(int(n / 2), samplesize)
    fn = min(int(n * k / 4), f_num)
    set_parameters([n, rep, k, ss, fn, seed, name + "regular-"])
    write_graphs()
    for i in range(rep):
        random.seed(seed + i)
        g = read_graph(i)
        random.seed(seed + i)
        for (algoname, algo) in algos.items():
            # graph, size, connectivity, algorithm, index,
            out.write('%s, %i, %i, %s, %i' % ("regular", n, k, algoname, i))
            algos[algoname] += [one_experiment(g, seed + i, out, algo)]


# run experiments with zoo graphs
# out denotes file handle to write results to
# seed is used for pseudorandom number generation in this run
# rep denotes the number of repetitions in the shuffle for loop
def run_zoo(out=None, seed=0, rep=2):
    global f_num
    fr = 15 #die zahl muss geändert werden damit man die fr ändert
    min_connectivity = 2
    original_params = [n, rep, k, samplesize, f_num, seed, name]
    if DEBUG:
        print('n_before, n_after, m_after, connectivity, degree')
    zoo_list = list(glob.glob("./benchmark_graphs/*.graphml"))
    for i in range(len(zoo_list)):
        random.seed(seed)
        g = read_zoo(i, min_connectivity)
        if g is None:
            continue

        print("Len(g) = " , len(g.nodes))
        kk = nx.edge_connectivity(g)
        nn = len(g.nodes())
        if nn < 200:
            print("Passender Graph ")
            mm = len(g.edges())
            ss = min(int(nn / 2), samplesize)
            f_num = kk * fr
            fn = min(int(mm / 4), f_num)
            if fn == int(mm / 4):
                print("SKIP ITERATION")
                continue
            print("Fehleranzahl : ", fn)
            set_parameters([nn, rep, kk, ss, fn, seed, name + "zoo-"])
            print("Node Number : " , nn)
            print("Connectivity : " , kk)
            print("Failure Number : ", fn)
            #print('parameters', nn, rep, kk, ss, fn, seed)
            shuffle_and_run(g, out, seed, rep, str(i))
            set_parameters(original_params)
            for (algoname, algo) in algos.items():
                index_1 = len(algo) - rep
                index_2 = len(algo)
                print('intermediate result: %s \t %.5E' % (algoname, np.mean(algo[index_1:index_2])))

# shuffle root nodes and run algorithm
def shuffle_and_run(g, out, seed, rep, x):
    random.seed(seed)
    nodes = list(g.nodes())
    random.shuffle(nodes)
    for count in range(rep):
        g.graph['root'] = nodes[count % len(nodes)]
        graph = g
        for (algoname, algo) in algos.items():
            # graph, size, connectivity, algorithm, index,
            out.write('%s, %i, %i, %s, %i' % (x, len(nodes), g.graph['k'], algoname, count))
            algos[algoname] += [one_experiment(g, seed + count, out, algo)]


# start file to capture results
def start_file(filename):
    out = open(filename + ".txt", 'w')
    out.write(
        "#graph, size, connectivity, algorithm, index, " +
        "stretch, load, hops, success, " +
        "routing computation time, pre-computation time in seconds\n")
    out.write(
        "#" + str(time.asctime(time.localtime(time.time()))) + "\n")
    return out




# run experiments
# seed is used for pseudorandom number generation in this run
# switch determines which experiments are run
def experiments(switch="all", seed=0, rep=3):
    

    if switch in ["zoo", "all"]:
        out = start_file("results/benchmark-ZOO-all-onetreeCP-CLUSTERED-FR" + str(i) + "-" + str(k))
        run_zoo(out=out, seed=seed, rep=rep)
        out.close()

    if switch in ["regular", "all"]:
       out = start_file("results/benchmark-regular-onetree_CP_CLUSTERED-" + str(n) + "-" + str(k))
       run_regular(out=out, seed=seed, rep=rep)
       out.close()

    print()
    for (algoname, algo) in algos.items():
        print('%s \t %.5E' % (algoname, np.mean(algo[2:])))
    print("\nlower is better")


if __name__ == "__main__":
    f_num = 5 #number of failed links
    for i in range(1,13):
        n = 50 # number of nodes
        k = 5 #base connectivity
        samplesize = 5 #number of sources to route a packet to destination
        rep = 5 #number of experiments
        switch = 'all' #which experiments to run with same parameters
        seed = 0  #random seed
        name = "benchmark-" #result files start with this name
        short = None #if true only small zoo graphs < 25 nodes are run
        start = time.time()
        print(time.asctime(time.localtime(start)))
        if len(sys.argv) > 1:
            switch = sys.argv[1]
        if len(sys.argv) > 2:
            seed = int(sys.argv[2])
        if len(sys.argv) > 3:
            rep = int(sys.argv[3])
        if len(sys.argv) > 4:
            n = int(sys.argv[4])
        if len(sys.argv) > 4:
            samplesize = int(sys.argv[5])
        random.seed(seed)
        set_parameters([n, rep, k, samplesize, f_num, seed, "benchmark-"])
        experiments(switch=switch, seed=seed, rep=rep)
        end = time.time()
        print("time elapsed", end - start)
        print("start time", time.asctime(time.localtime(start)))
        print("end time", time.asctime(time.localtime(end)))
        f_num = f_num * i







