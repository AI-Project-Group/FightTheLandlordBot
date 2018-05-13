import numpy as np
import tensorflow as tf
from simulator import Hand

LearningRate = 1e-4

class Network:
    
    def __init__(self,graph,checkpoint_file):
        self.graph = graph
        with self.graph.as_default():
            self.sess = tf.InteractiveSession(graph=self.graph)
            tf.global_variables_initializer().run()      
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
        with self.graph.as_default():      
            self.saver.save(self.sess, self.checkpoint_file)

    def restore_checkpoint(self):
        print("Restore checkpoint...")
        try:
            with self.graph.as_default():
                self.saver.restore(self.sess, self.checkpoint_file)
        except Exception as err:
            print("Fail to restore...")
            print(err)

class PlayModel(Network):

    def __init__(self,checkpoint_file):
        inUnits = 7*15
        fcUnits = [inUnits,512,1024]
        outUnits = 252
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, inUnits])
            self.keep_prob = tf.placeholder(tf.float32)
            
            fc_in = self.x
            for i in range(len(fcUnits)-1):
                fc_in = self.fc_layer(fc_in, fcUnits[i], fcUnits[i+1], self.keep_prob, name="fc"+str(i))
            
            self.out = [self.out_layer(fc_in, fcUnits[-1], outUnits, name="out"+str(i)) for i in range(3)]
            self.y = [tf.nn.softmax(self.out)]
            self.y_ = tf.placeholder(tf.float32, [None, outUnits])
            self.rewards = tf.placeholder(tf.float32, [None])
            
            self.cross_entropy = [tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.out[i]) for i in range(3)]
            self.tmploss = [tf.reduce_sum(tf.multiply(self.cross_entropy[i], self.rewards)) for i in range(3)]
            self.identity = tf.placeholder(tf.float32,[3])
            self.loss = tf.reduce_sum(tf.multiply(self.tmploss, self.identity))
            self.train_step = tf.train.AdamOptimizer(LearningRate).minimize(self.loss)
            #print(tf.all_variables())
        
        super(PlayModel, self).__init__(self.graph,checkpoint_file)
        #print(tf.all_variables())
    
    def cards2NumArray(self,cards):
        point = Hand.getCardPoint(cards)
        res = np.zeros(15)
        for p in point:
            res[p] += 1
        return res
    
    def ch2input(self,playerID,myCards,publicCards,history,lastPlay,lastLastPlay):
        net_input = []
        net_input.append(self.cards2NumArray(myCards))
        net_input.append(self.cards2NumArray(publicCards))
        player = playerID
        for _ in range(3):
            tmphis = []
            for his in history[player]:
                tmphis.extend(his)
            net_input.append(self.cards2NumArray(tmphis))
            player = (player - 1) % 3
        net_input.append(self.cards2NumArray(lastPlay))
        net_input.append(self.cards2NumArray(lastLastPlay))
        return np.array(net_input).flatten()
    
    #def hand2idx(self,hand):
        #if 
            
#PlayModel("test.ckpt")