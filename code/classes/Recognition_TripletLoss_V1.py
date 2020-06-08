# Project: Prototyping-Encoder-Decoder-with-Triplet-Loss
# version: 1.0
# Author: CSULB Data Science Lab Team
# Date: June 2020
#
#Include a reference to this site if you will use this code.

'''
Code from tensorflow addon library
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

@tf.function
def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    # yapf: disable
    pairwise_distances_squared = tf.math.add(
        tf.math.reduce_sum(
            tf.math.square(feature),
            axis=[1],
            keepdims=True),
        tf.math.reduce_sum(
            tf.math.square(tf.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * tf.matmul(feature, tf.transpose(feature))
    # yapf: enable

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared,
                                                 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared +
            tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = tf.math.multiply(
        pairwise_distances,
        tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32))

    num_data = tf.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data]))
    pairwise_distances = tf.math.multiply(pairwise_distances,
                                          mask_offdiagonals)
    return pairwise_distances

'''
Code from tensorflow addon library
'''
def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.
    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = tf.math.reduce_max(
        tf.math.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums


def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.
    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = tf.math.reduce_min(
        tf.math.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums


#This is the loss function interface used in the definition of the network arquitecture
def custom_loss(margin, beta):
  '''
  Use this for triplet semi hard loss
  '''
  def triplet_semihard_loss(y_true, y_pred):
    """Computes the triplet loss with semi-hard negative mining.
    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        multiclass integer labels. of shape m * 1
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized. of shape m * embedding_size
      margin: Float, margin term in the loss definition.
    """
    
    labels, embeddings = y_true, y_pred
    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)

    # Compute the mask.
    pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
    mask = tf.math.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.math.greater(pdist_matrix_tile,
                        tf.reshape(tf.transpose(pdist_matrix), [-1, 1])))
    mask_final = tf.reshape(
        tf.math.greater(
            tf.math.reduce_sum(
                tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = tf.transpose(mask_final)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    mask = tf.cast(mask, dtype=tf.dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = tf.reshape(
        _masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = tf.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = tf.tile(
        _masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = tf.where(mask_final, negatives_outside,
                                   negatives_inside)

    loss_mat = tf.math.add(margin, pdist_matrix - beta * semi_hard_negatives)

    mask_positives = tf.cast(
        adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
            tf.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = tf.math.reduce_sum(mask_positives)

    triplet_loss = tf.math.truediv(
        tf.math.reduce_sum(
            tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)),
        num_positives)

    return triplet_loss
  return triplet_semihard_loss



#Computes the triplet loss with semi-hard negative mining and the prototype loss
def combined_loss(y_true, y_pred, decoded, c_mean, alpha, beta, lamda):
            
    #***********************************************************************
    # Triplet Loss
    #***********************************************************************
    labels, embeddings = y_true, y_pred
    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)

    # Compute the mask.
    pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
    mask = tf.math.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.math.greater(pdist_matrix_tile,
                        tf.reshape(tf.transpose(pdist_matrix), [-1, 1])))
    mask_final = tf.reshape(
        tf.math.greater(
            tf.math.reduce_sum(
                tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = tf.transpose(mask_final)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    mask = tf.cast(mask, dtype=tf.dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = tf.reshape(
        _masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = tf.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = tf.tile(
        _masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = tf.where(mask_final, negatives_outside,
                                   negatives_inside)

    loss_mat = tf.math.add(alpha, pdist_matrix - beta * semi_hard_negatives) 
    
    mask_positives = tf.cast(
        adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
            tf.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = tf.math.reduce_sum(mask_positives)

    triplet_loss = tf.math.truediv(
        tf.math.reduce_sum(
            tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)),
        num_positives)
        
    #***********************************************************************
    # Prototype Loss
    #***********************************************************************
    n = tf.norm(decoded, axis=1)
    n = tf.reshape(tf.tile(n, [1, decoded.shape[1]]), (batch_size, decoded.shape[1], decoded.shape[2]))
    decoded = tf.math.divide_no_nan(decoded, n)
    difference = tf.math.subtract(decoded, c_mean)
    norm_squared = tf.math.square(tf.norm(difference, axis=(1,2)))
    mask_positives = tf.math.reduce_sum(mask_positives, axis=0)
    mask_positives =  tf.cast(tf.math.greater(mask_positives, 0.0), dtype=tf.dtypes.float32) 
    norm_positives = tf.math.multiply(norm_squared, mask_positives)
    sum_norm_positives = tf.math.reduce_sum(norm_positives)
    num_positives = tf.math.reduce_sum(mask_positives)
    prototype_loss = tf.math.truediv(sum_norm_positives, num_positives)
    
    #***********************************************************************
    # Combined Loss
    #***********************************************************************    
    loss = tf.math.add(triplet_loss, lamda * prototype_loss)
    return loss  
