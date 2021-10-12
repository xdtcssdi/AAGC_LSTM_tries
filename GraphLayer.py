import tensorflow as tf
import numpy as np
from AdjacencyGraph import AdjacencySolver

class AAGCLayer(tf.keras.layers.Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, input_dim = 15, output_dim = 15, Hidden_len = 50,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform', # 均匀分布
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None):
        super(AAGCLayer, self).__init__()

        self.Hidden_len = Hidden_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activation = activation

        # adjacency graph inside
        ad = AdjacencySolver(onehot=True,use_connect=False,use_flow = False,invert = False)
        self.A_init = ad.A_init.tolist()# tf.constant
        self.Hidden = tf.Variable(np.zeros((15,self.Hidden_len)),name = 'Hidden',trainable=False,dtype=tf.float32)
        # reg_l2 = tf.keras.regularizers.L2

    def build(self, nodes_shape):
        self.A_change = self.add_weight(shape = (15,15),
                                      initializer = 'zeros',
                                      name = 'A_change',
                                      regularizer = tf.keras.regularizers.L2(0.1),
                                    #   trainable=True,
                                      dtype=tf.float32)
        self.sigma = self.add_weight(shape = (self.Hidden_len, self.Hidden_len),
                                      initializer = self.kernel_initializer,
                                      name = 'sigma',
                                      regularizer = self.kernel_regularizer,
                                      dtype=tf.float32)
        self.kernel = self.add_weight(shape = (self.Hidden_len, 15),
                                      initializer = self.kernel_initializer,
                                      name = 'kernel',
                                      regularizer = self.kernel_regularizer,
                                      dtype=tf.float32)
        if self.use_bias:
            self.bias = self.add_weight(shape=(15, self.Hidden_len),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer = self.bias_regularizer,
                                        dtype=tf.float32)
        else:
            self.bias = None
            
        self.built = True

    def call(self, new_features):

        A_cur = tf.add(self.A_init,self.A_change) # 15, 15

        support = tf.matmul(A_cur, self.Hidden) # 15, self.Hidden_len
        support = tf.math.sigmoid(support)

        Hidden_new = tf.matmul(support, self.sigma) # 15, self.Hidden_len
        if self.use_bias:
            Hidden_new += self.bias # 15, self.Hidden_len
        Hidden_new = tf.math.sigmoid(Hidden_new)

        self.Hidden.assign(Hidden_new)  # 15, self.Hidden_len
        
        mapfuc = tf.matmul(Hidden_new,self.kernel) # 15, 15
        mapfuc = tf.math.sigmoid(mapfuc)

        new_features = tf.reshape(new_features,[-1,15,9])
        feature_by_A = tf.matmul(A_cur,new_features)
        
        output = tf.matmul(mapfuc,feature_by_A )
        output = tf.reshape(output,[-1,135])
        output = tf.math.sigmoid(output)
            
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'Hidden_len': self.Hidden_len,
            'A_init': self.A_init,
            # https://blog.csdn.net/qq_39269347/article/details/111464049
            # NO need to write tf.Variabl-like here, just a name would be enough
            # 'A_change': self.A_change,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer':self.bias_initializer,
            'kernel_regularizer':self.kernel_regularizer,
            'bias_regularizer':self.bias_regularizer,
            'activation':self.activation,
        })
        return config