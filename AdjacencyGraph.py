import numpy as np
# import scipy.sparse as sp

class AdjacencySolver:
    def __init__(self,onehot=True,use_connect=False,use_flow = False,invert = False):
        """
        onehot:      if true, the adjacency matrix don't consider distance, set edge to 1 if is neighbor
        use_connect: if true, let node pairs that are not neighbors can be connected by a path
        use_flow:    only if use_connect is true when it matters,
                     let every element a in A_init: lambda a: 1/a
        invert:      apply a: 1-a, after normalize
        """
        A_gen = A_generator(onehot=onehot,use_connect=use_connect)
        self.A_init = np.array(A_gen.A_init,dtype=np.float64)

        # TODO:limited use
        if use_flow and use_connect:
            self.add_I(0.5)
            self.A_init = 1/self.A_init
            # self.add_I(1) # additional I for self connected node
        else:
            self.add_I()

        self.normalize_adj()
        if invert:
            self.A_init = 1 - self.A_init
        
    def add_I(self,rate = 1):
        """calculate A_ = (A + rate*I)"""
        self.A_init += rate * np.eye(15,dtype=np.float64)
        
    def normalize_adj(self):
        """calculate L=D^-0.5 * A_ * D^-0.5"""
        degree = np.array(self.A_init.sum(1))
        d_hat = np.diag(np.power(degree, -0.5).flatten())

        self.A_init =  d_hat.dot(self.A_init).dot(d_hat)


class A_generator:
    """
    generator a basic A_init for later use
    """
    def __init__(self,onehot = False,use_connect = True):
        self.A_init = [
        #   [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
            [ 0,  0,  3,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], #0
            [ 0,  0,  3,  0,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], #1
            [ 3,  3,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0], #2
            [ 5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], #3
            [ 0,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], #4
            [ 0,  0,  3,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0], #5
            [ 0,  0,  0,  0,  0,  1,  0,  3,  3,  3,  0,  0,  0,  0,  0], #6
            [ 0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  2,  0,  0,  0,  0], #7
            [ 0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  3,  0,  0,  0], #8
            [ 0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  3,  0,  0], #9
            [ 0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0], #10
            [ 0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  5,  0], #11
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  5], #12
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0], #13
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  0,  0], #14
        ]
        
        if onehot:
            def func(num):
                if num > 0:
                    return 1
                else:
                    return 0
            self.A_init = [list(map(func,row)) for row in self.A_init]

        if use_connect:
            self.connected_graph()

    def connected_graph(self):
        for row in range(len(self.A_init)):
            # find any dest-non-zero adjacent node
            n_zero_id_value = [[id,value] for id,value in enumerate(self.A_init[row]) if value!=0]
            for id,value in n_zero_id_value:
                # the select node has dest info which is calculated
                id_dest = [[id_s,value_s] for id_s,value_s in enumerate(self.A_init[id]) if value_s!=0]
                for id_s,value_s in id_dest:
                    if id_s == row:
                        continue
                    if self.A_init[row][id_s] == 0:
                        self.A_init[row][id_s] = value+value_s
                        self.A_init[id_s][row] = value+value_s


