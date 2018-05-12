import numpy as np
import tensorflow as tf

Class Network:
    
    def __init__(self,checkpoint_file):
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.checkpoint_file = checkpoint_file
    
    @staticmethod
    def conv_layer(intensor,conWidth,conStride,inUnits,outUnits,name="conv"):
        with tf.name_scope(name):
            W = tf.Variable(tf.truncated_normal([conWidth, conWidth, inUnits, outUnits], stddev=0.01),name="W")
            b = tf.Variable(tf.zeros([outUnits]),name="b")
            conv = tf.nn.conv2d(intensor, W, strides=[1, conStride, conStride, 1], padding="VALID")
            relu = tf.nn.relu(conv+b)
        return relu
    
    @staticmethod
    def maxp_layer(intensor,width,stride,name="maxpooling"):
        with tf.name_scope(name):
            pool = tf.nn.max_pool(intensor, ksize=[1,width,width,1], strides=[1,stride,stride,1], padding="VALID")
        return pool
    
    @staticmethod
    def fc_layer(intensor,inUnits,outUnits,keep_prob,name="fc"):
        with tf.name_scope(name):
            W = tf.Variable(tf.truncated_normal([inUnits, outUnits], stddev=0.01),name="W")
            b = tf.Variable(tf.zeros([outUnits]),name="b")
            relu = tf.nn.relu(tf.matmul(intensor,W)+b)
            dropout = tf.nn.dropout(relu, keep_prob)
        return dropout
    
    @staticmethod
    def out_layer(intensor,inUnits,outUnits,name="out"):
        with tf.name_scope(name):
            W = tf.Variable(tf.truncated_normal([inUnits, outUnits], stddev=0.01),name="W")
            b = tf.Variable(tf.zeros([outUnits]),name="b")
            out = tf.matmul(intensor,W)+b
        return out
    
    def save_checkpoint(self):
        print("Save checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def restore_checkpoint(self):
        print("Restore checkpoint...")
        try:
            self.saver.restore(self.sess, self.checkpoint_file)
        except Exception as err:
            print("Fail to restore...")
            print(err)