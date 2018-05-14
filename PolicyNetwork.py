import random
import numpy as np
import tensorflow as tf
from simulator import Hand

LearningRate = 1e-4
BatchSize = 20
Gamma = 0.95
MaxEpoch = 5
TrainKeepProb = 0.75

class Network:
    
    def __init__(self,modelname,checkpoint_file):
        with tf.name_scope(modelname):
            self.sess = tf.InteractiveSession()
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
    
    def save_model(self):
        print("Save checkpoint...")
        with self.graph.as_default():      
            self.saver.save(self.sess, self.checkpoint_file)

    def load_model(self):
        print("Restore checkpoint...")
        try:
            with self.graph.as_default():
                self.saver.restore(self.sess, self.checkpoint_file)
        except Exception as err:
            print("Fail to restore...")
            print(err)

class PlayModel(Network):

    def __init__(self,modelname,checkpoint_file):
        inUnits = 7*15
        fcUnits = [inUnits,512,1024]
        outUnits = 364
        self.outUnits = outUnits
        self.name = modelname
        
        #self.graph = tf.Graph()
        with tf.name_scope(modelname):
            self.x = tf.placeholder(tf.float32, [None, inUnits])
            self.keep_prob = tf.placeholder(tf.float32)
            
            fc_in = self.x
            for i in range(len(fcUnits)-1):
                fc_in = self.fc_layer(fc_in, fcUnits[i], fcUnits[i+1], self.keep_prob, name="fc"+str(i))
            
            self.out = [self.out_layer(fc_in, fcUnits[-1], outUnits, name="out"+str(i)) for i in range(3)]
            self.y = [tf.nn.softmax(self.out[i]) for i in range(3)]
            self.y_ = tf.placeholder(tf.float32, [None, outUnits])
            self.rewards = tf.placeholder(tf.float32, [None])
            
            self.cross_entropy = [tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.out[i]) for i in range(3)]
            self.loss = [tf.reduce_sum(tf.multiply(self.cross_entropy[i], self.rewards)) for i in range(3)]
            #self.identity = tf.placeholder(tf.float32,[None, 3])
            #self.loss = tf.reduce_sum(tf.multiply(self.tmploss, self.identity))
            self.train_step = [tf.train.AdamOptimizer(LearningRate).minimize(self.loss[i]) for i in range(3)]
            #self.train_step = tf.train.AdamOptimizer(LearningRate).minimize(self.loss)
            #print(tf.all_variables())
        
        super(PlayModel, self).__init__(modelname,checkpoint_file)
        #print(tf.all_variables())
        self.trainBatch = [[],[],[]]
        self.episodeTemp = [[],[],[]]
        
        #for var in tf.all_variables():
            #print(var)
    
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
    
    def getDisFromChain(self,chain,baseChain,maxNum):
        chaindis = chain - baseChain
        num = maxNum
        res = 0
        for _ in range(chaindis):
            res += num
            num -= 1
        return res
    
    # change cardPoints to the index of network output
    def cardPs2idx(self,cardPoints):
        if cardPoints and isinstance(cardPoints[0],list):
            tmphand = cardPoints[1:]
            lenh = len(tmphand)
            kickers = []
            if lenh % 3 == 0:
                kickers = cardPoints[0][:lenh//3]
            elif lenh % 4 == 0:
                kickers = cardPoints[0][:lenh//2]
            for k in kickers:
                tmphand.extend(k)
            cardPoints = tmphand
        hand = Hand([],cardPoints)
        lencp = len(cardPoints)
        if hand.type == "None" or hand.type == "Pass":
            return 0
        elif hand.type == "Solo":
            if hand.chain == 1:
                return 1 + hand.primal
            else:
                idx = 108
                idx += self.getDisFromChain(hand.chain,5,8)
                idx += hand.primal
                return idx
        elif hand.type == "Pair":
            if hand.chain == 1:
                return 16 + hand.primal
            else:
                idx = 144
                idx += self.getDisFromChain(hand.chain,3,10)
                idx += hand.primal
                return idx
        elif hand.type == "Trio":
            if hand.chain == 1:
                return 29 + hand.primal + hand.kickerNum * 13
            else:
                idx = 196
                idx += self.getDisFromChain(hand.chain,2,11)
                return idx + hand.primal + hand.kickerNum * 45
        elif hand.type == "Four":
            if hand.chain == 1:
                return 68 + hand.primal + (hand.kickerNum-1) * 13
            else:
                return 331 + hand.primal + hand.kickerNum * 11
        elif hand.type == "Bomb":
            return 94 + hand.primal
        elif hand.type == "Rocket":
            return 107
        else:
            return 0
    
    # change all possible hands to one hot tensor
    def hand2one_hot(self,allhands):
        res = np.zeros(self.outUnits)
        for hand in allhands:
            idx = self.cardPs2idx(hand)
            res[idx] = 1
        return res
        
    def getChainFromDis(self,dis,baseChain,maxNum):
        chain = baseChain
        num = maxNum
        while dis >= num:
            dis -= num
            num -= 1
            chain += 1
        return chain, dis
    
    # change index of one hot output to cardPoints
    # if possible hand has kickers, the first element will be dict
    def idx2CardPs(self,idx):
        res = []
        if idx == 0: # Pass
            res = []
        elif idx <= 15: # Solo
            res = [idx - 1] 
        elif idx <= 28: # Pair
            res = [idx - 16]*2
        elif idx <= 41: # Trio without kickers
            res = [idx - 29]*3
        elif idx <= 54: # Trio with one kicker
            res.append({"kickerNum":1,"chain":1,"type":"Trio"})
            res.extend([idx-42]*3)
        elif idx <= 67: # Trio with two kickers
            res.append({"kickerNum":2,"chain":1,"type":"Trio"})
            res.extend([idx-55]*3)
        elif idx <= 80: # Four with one kicker
            res.append({"kickerNum":1,"chain":1,"type":"Four"})
            res.extend([idx-68]*4)
        elif idx <= 93: # Four with two kickers
            res.append({"kickerNum":2,"chain":1,"type":"Four"})
            res.extend([idx-81]*4)
        elif idx <= 106: # Bomb
            res = [idx - 94]*4
        elif idx == 107: # Rocket
            res = [13,14]
        elif idx <= 143: # Solo Chain
            chain,primal = self.getChainFromDis(idx-108,5,8)
            res = list(range(primal,primal+chain))
        elif idx <= 195: # Pair Chain
            chain,primal = self.getChainFromDis(idx-144,3,10)
            res = list(range(primal,primal+chain)) * 2
        elif idx <= 240: # Airplane without wings
            chain,primal = self.getChainFromDis(idx-196,2,11)
            res = list(range(primal,primal+chain)) * 3
        elif idx <= 285: # Airplane with small wings
            chain,primal = self.getChainFromDis(idx-241,2,11)
            res.append({"kickerNum":1,"chain":chain,"type":"Trio"})
            res.extend(list(range(primal,primal+chain)) * 3)
        elif idx <= 330: # Airplane with big wings
            chain,primal = self.getChainFromDis(idx-286,2,11)
            res.append({"kickerNum":2,"chain":chain,"type":"Trio"})
            res.extend(list(range(primal,primal+chain)) * 3)
        elif idx <= 341: # Shuttle without wings
            chain,primal = self.getChainFromDis(idx-331,2,11)
            res = list(range(primal,primal+chain)) * 4
        elif idx <= 352: # Shuttle with small wings
            chain,primal = self.getChainFromDis(idx-342,2,11)
            res.append({"kickerNum":1,"chain":chain,"type":"Four"})
            res.extend(list(range(primal,primal+chain)) * 4)
        elif idx <= 363: # Shuttle with big wings
            chain,primal = self.getChainFromDis(idx-353,2,11)
            res.append({"kickerNum":2,"chain":chain,"type":"Four"})
            res.extend(list(range(primal,primal+chain)) * 4)            
        return res
    
    # get possible actions
    def getActions(self,netinput,playerID,allonehot):
        epsilon = 1e-6
        output = self.y[playerID].eval(feed_dict={self.x:[netinput], self.keep_prob:1.0})
        output = output.flatten()
        #print(output)
        legalOut = np.multiply(output, allonehot)
        #print(legalOut)
        total = np.sum(legalOut)
        #print(total)
        randf = random.uniform(0,total)
        sum = 0
        #print(randf)
        for i in range(len(legalOut)):
            if abs(allonehot[i] - 1) < epsilon:
                sum += legalOut[i]
                #print(sum)
                if randf < sum:
                    return self.idx2CardPs(i)
        return self.idx2CardPs(np.argmax(allonehot))
    
    # store the train samples
    def storeSamples(self, netinput, playerID, action):
        actidx = self.cardPs2idx(Hand.getCardPoint(action))
        self.episodeTemp[playerID].append([netinput,actidx,0])
        #print(self.episodeTemp)
    
    # compute rewards and train
    def finishEpisode(self, scores):
        
        for p in range(3):
            for tmp in self.episodeTemp[p][::-1]:
                tmp[2] = scores[p]
                scores[p] *= Gamma
        
        #print(self.episodeTemp)
        for p in range(3):
            for data in self.episodeTemp[p]:
                self.trainBatch[p].append(data)
                if len(self.trainBatch[p]) == BatchSize:
                    random.shuffle(self.trainBatch[p])
                    self.trainModel(p)
                    self.trainBatch[p] = []
        
        self.episodeTemp = [[],[],[]]
        
    # train the model
    def trainModel(self,player):
        batch = self.trainBatch[player]
        netinput = [d[0] for d in batch]
        actidxs = [d[1] for d in batch]
        rewards = [d[2] for d in batch]
        acts = np.zeros((BatchSize,self.outUnits))
        for i in range(BatchSize):
            acts[i][actidxs[i]] = 1
        #print(actidxs)
        #print(player)
        #print(acts.tolist())
        
        '''tmpvar = []
        for var in tf.global_variables():
            if var.name == self.name+"/out1/W:0" or var.name == self.name+"/out0/W:0" or var.name == self.name+"/out2/W:0":
                tmpvar.append(var)
        for var in tmpvar:
            print(self.sess.run(var))
        print(self.loss[player].eval(feed_dict={self.x:netinput, self.keep_prob:1.0,  self.y_:acts, self.rewards:rewards}))'''
        #print(self.loss.eval(feed_dict={self.x:netinput, self.keep_prob:1.0, self.identity:players, self.y_:acts, self.rewards:rewards}))
        for _ in range(MaxEpoch):
            self.sess.run(self.train_step[player], feed_dict={self.x:netinput, self.keep_prob:TrainKeepProb, self.y_:acts, self.rewards:rewards})
        '''for var in tmpvar:
            print(self.sess.run(var))
        print(self.loss[player].eval(feed_dict={self.x:netinput, self.keep_prob:1.0, self.y_:acts, self.rewards:rewards}))'''
        #exit(0)
        
#print(a.cardPs2idx([0,0,0,0,1,2,2,1]))
