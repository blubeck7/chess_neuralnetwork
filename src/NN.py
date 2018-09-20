import os
import numpy as np
import tensorflow as tf

# dataset import functions

x = tf.placeholder(tf.float64, shape = [None, numInputNodes])
y = tf.placeholder(tf.float64, shape = [None, numOutputNodes])

def loadData

def my_input_fn():

    # Preprocess your data here...

    # ...then return 1) a mapping of feature columns to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return feature_cols, labels

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

if __name__ == "__main__":
    main()
