Running Clustered Failures on Real World Graphs

To start the experiments with graphs from the Topology Zoo using clustered failures, execute the following command:

bash

python3 new_clustered_experiments.py zoo 45 5 100 5 40 CLUSTER False

Explanation of the inputs (from left to right):

    zoo: Specifies which experiments to run with the same parameters.
    45: Random seed for choosing the source and destination.
    5: Number of experiments to run.
    100: Number of nodes in the graph.
    5: Number of sources to route a packet to the destination.
    40: Number of failed links.
    CLUSTER: Method for choosing edge failures.
    False: If true, only small zoo graphs with fewer than 25 nodes are run.
