# Standard Algorithm Implementation
# Sampling-based Algorithms PRM
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import spatial

# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.points = []                      # list of all points
        self.obs = []                         # list of all points that lie on obstacles
        self.tree = None

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.path = []                        # list of nodes of the found path

        # list of points
        for i in range(0, self.size_row):
            for j in range(0, self.size_col):
                if self.map_array[i][j] == 1:
                    self.points.append([i, j])
                elif self.map_array[i][j] == 0:
                    self.obs.append([i, j])
                    self.points.append([i, j])


    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''
        rows_bet = np.linspace(p1[0], p2[0], dtype=int)
        col_bet = np.linspace(p1[1], p2[1], dtype=int)

        for p in zip(rows_bet, col_bet):
            if self.map_array[p[0]][p[1]] == 0:
                return True
        return False


    def dis(self, point1, point2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        e_dist = math.sqrt(sum([(a-b)**2 for a, b in zip(point1, point2)]))
        return e_dist


    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valid points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        samples = []
        numr = int(np.sqrt(n_pts * self.size_row/self.size_col))
        numc = int(n_pts / numr)

        uniform_r = np.linspace(0, self.size_row - 1, numr, dtype=int)
        uniform_c = np.linspace(0, self.size_col - 1, numc, dtype=int)

        # create matrix of uniform samples
        row, col = np.meshgrid(uniform_r, uniform_c)

        row = row.flatten()
        col = col.flatten()

        # collision check
        for r, c in zip(row, col):
            if self.map_array[r][c] == 1:
                self.samples.append([r, c])
    
    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        samples = []

        row = np.random.randint(0, self.size_row - 1, n_pts, dtype=int)
        col = np.random.randint(0, self.size_col - 1, n_pts, dtype=int)

        samples.append([row, col])

        # collision check
        for r, c in zip(row, col):
            if self.map_array[r][c] == 1:
                self.samples.append([r, c])
        # print("s =", self.samples)

    def gaussian_sample(self, n_pts):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        # for i in range(0, n_pts):
        row1 = np.random.randint(0, self.size_row - 1, n_pts, dtype=int)
        col1 = np.random.randint(0, self.size_col - 1, n_pts, dtype=int)

        scale = 10
        row2 = row1 + np.random.normal(0.0, scale, n_pts).astype(int)
        col2 = col1 + np.random.normal(0.0, scale, n_pts).astype(int)

        for r1, c1, r2, c2 in zip(row1, col1, row2, col2):
            if not (0 <= r2 < self.size_row) or not (0 <= c2 < self.size_col):
                continue
            if self.map_array[r1][c1] == 1 and self.map_array[r2][c2] == 0:
                self.samples.append([r1, c1])
            elif self.map_array[r1][c1] == 0 and self.map_array[r2][c2] == 1:
                self.samples.append([r2, c2])

    def bridge_sample(self, n_pts):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        row1 = np.random.randint(0, self.size_row - 1, n_pts, dtype=int)
        col1 = np.random.randint(0, self.size_col - 1, n_pts, dtype=int)
        scale = 15
        row2 = row1 + np.random.normal(0.0, scale, n_pts).astype(int)
        col2 = col1 + np.random.normal(0.0, scale, n_pts).astype(int)

        for r1, c1, r2, c2 in zip(row1, col1, row2, col2):
            if (not(0 <= r2 < self.size_row) or not(0 <= c2 < self.size_col) or self.map_array[r2][c2] == 0) and self.map_array[r1][c1] == 0:
                # finding mid-point
                mp_row, mp_col = int((r1+r2)/2), int((c1+c2)/2)
                # checking where it lies
                if 0 <= mp_row < self.size_row and 0 <= mp_col < self.size_col and self.map_array[mp_row][mp_col] == 1:
                    self.samples.append([mp_row, mp_col])

    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict( zip( range( len(self.samples) ), node_pos) )
        pos['start'] = (self.samples[-2][1], self.samples[-2][0])
        pos['goal'] = (self.samples[-1][1], self.samples[-1][0])
        
        # draw constructed graph
        nx.draw(self.graph, pos, node_size=3, node_color='y', edge_color='y' ,ax=ax)

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.path, node_size=8, node_color='b')
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=final_path_edge, width=2, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['start'], node_size=12,  node_color='g')
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['goal'], node_size=12,  node_color='r')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()


    def sample(self, n_pts=1000, sampling_method="uniform"):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []

        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
        elif sampling_method == "random":
            self.random_sample(n_pts)
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts)
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts)

        # Find the pairs of points that need to be connected. # Finding NN
        r = 20
        samples = list(self.samples)

        self.tree = spatial.KDTree(samples)
        pairs = self.tree.query_pairs(r)
        # print("pp = ", pairs)

        self.graph.add_nodes_from(range(0, len(self.samples)))

        for p in pairs:
            if p[0] == "start":
                point1 = self.samples[-2]
            elif p[0] == "goal":
                point1 = self.samples[-1]
            else:
                point1 = self.samples[p[0]]
            point2 = self.samples[p[1]]

            # Collision checking and computing their distance/weight.
            if self.check_collision(point1, point2) == False:
                weight = self.dis(point1, point2)
                # Storing them as pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02) ...]
                connect = [(p[0], p[1], weight)]
                self.graph.add_weighted_edges_from(connect)

        # Use sampled points and pairs of points to build a graph.

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" %(n_nodes, n_edges))


    def search(self, start, goal):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(['start', 'goal'])

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.

        s_tree = spatial.KDTree([start, goal])
        r = 100
        nn = s_tree.query_ball_tree(self.tree, r)

        start_pairs = ([['start', n] for n in nn[0]])
        goal_pairs = ([['goal', n] for n in nn[1]])

        for p in start_pairs:
            if p[0] == "start":
                point1 = self.samples[-2]
            elif p[0] == "goal":
                point1 = self.samples[-1]
            else:
                point1 = self.samples[p[0]]
            point2 = self.samples[p[1]]

            # Collision checking and computing their distance/weight.
            if self.check_collision(point1, point2) == False:
                weight = self.dis(point1, point2)
                # Storing them as pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02) ...]
                connect = [(p[0], p[1], weight)]
                # Add the edge to graph
                self.graph.add_weighted_edges_from(connect)

        for p in goal_pairs:
            if p[0] == "start":
                point1 = self.samples[-2]
            elif p[0] == "goal":
                point1 = self.samples[-1]
            else:
                point1 = self.samples[p[0]]
            point2 = self.samples[p[1]]

            # Collision checking and computing their distance/weight.
            if self.check_collision(point1, point2) == False:
                weight = self.dis(point1, point2)
                # Storing them as pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02) ...]
                connect = [(p[0], p[1], weight)]
                # Add the edge to graph
                self.graph.add_weighted_edges_from(connect)
        
        # Search using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.graph, 'start', 'goal')
            print("The path length is %.2f" %path_length)
        except nx.exception.NetworkXNoPath:
            print("No path found")
        
        # Draw result
        self.draw_map()

        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(['start', 'goal'])
        self.graph.remove_edges_from(start_pairs)
        self.graph.remove_edges_from(goal_pairs)
        