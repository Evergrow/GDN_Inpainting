import tensorflow as tf
import os
import numpy as np
import inspect


def seg_loss(pred, mask):
    y_pred = tf.nn.softmax(pred, axis=-1)
    pred1, pred2 = tf.split(y_pred, [1, 1], axis=-1)
    label = tf.to_float(mask)
    log1 = tf.log(tf.clip_by_value(pred1, 1e-8, 1.0 - 1e-8))
    log2 = tf.log(tf.clip_by_value(pred2, 1e-8, 1.0 - 1e-8))
    loss = - tf.transpose(tf.multiply(1 - label, log1), perm=[1, 2, 3, 0]) - \
           tf.transpose(tf.multiply(label, log2), perm=[1, 2, 3, 0])
    loss = tf.transpose(loss, perm=[3, 0, 1, 2])
    loss = tf.reduce_mean(loss)
    return loss

def focal_loss(pred, mask, ratio, gamma=2):
    y_pred = tf.nn.softmax(pred, axis=-1)
    pred1, pred2 = tf.split(y_pred, [1, 1], axis=-1)
    label = tf.to_float(mask)
    log1 = tf.multiply(pred2 ** gamma, tf.log(tf.clip_by_value(pred1, 1e-8, 1.0 - 1e-8)))
    log2 = tf.multiply(pred1 ** gamma, tf.log(tf.clip_by_value(pred2, 1e-8, 1.0 - 1e-8)))
    loss = - ratio * tf.transpose(tf.multiply(1 - label, log1), perm=[1, 2, 3, 0]) - \
           (1 - ratio) * tf.transpose(tf.multiply(label, log2), perm=[1, 2, 3, 0])
    loss = tf.transpose(loss, perm=[3, 0, 1, 2])
    loss = tf.reduce_mean(loss)
    return loss

def l1_loss(inputs, targets):
    inputs = tf.reshape(inputs, [-1])
    targets = tf.reshape(targets, [-1])
    loss = tf.reduce_mean(tf.abs(inputs - targets))
    return loss
