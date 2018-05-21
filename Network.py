import random
import numpy as np
import tensorflow as tf
from simulator import Hand
from collections import deque

LearningRate = 1e-4
BatchSize = 256
BufferSize = 6400
KickersBatch = 16
Gamma = 0.98
MaxEpoch = 1
TrainKeepProb = 0.8
BaseScore = 200

class Network:
    
    def __init__(self,modelname,sess,checkpoint_file):
        self.sess = sess    
        self.saver = tf.train.Saver()
        self.checkpoint_file = checkpoint_file
        self.name = modelname
    
    @staticmethod
    def conv_layer(intensor,conWidth,conStride,inUnits,outUnits,name="conv"):
        with tf.name_scope(name):
            W = tf.Variable(tf.truncated_normal([conWidth, conWidth, inUnits, outUnits], stddev=0.1),name="W")
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
    def fc_layer(intensor,inUnits,outUnits,keep_prob,name="fc",mean=0,initstd=0.1):
        with tf.name_scope(name):
            W = tf.Variable(tf.truncated_normal([inUnits, outUnits], mean=mean, stddev=initstd),name="W")
            b = tf.Variable(tf.zeros([outUnits]),name="b")
            relu = tf.nn.relu(tf.matmul(intensor,W)+b)
            dropout = tf.nn.dropout(relu, keep_prob)
        return dropout
    
    @staticmethod
    def out_layer(intensor,inUnits,outUnits,name="out",mean=0,initstd=0.1):
        with tf.name_scope(name):
            W = tf.Variable(tf.truncated_normal([inUnits, outUnits], mean=mean, stddev=initstd),name="W")
            b = tf.Variable(tf.zeros([outUnits]),name="b")
            out = tf.matmul(intensor,W)+b
        return out
    
    def save_model(self):
        print("Save checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def load_model(self):
        print("Restore checkpoint...")
        try:
            self.saver.restore(self.sess, self.checkpoint_file)
        except Exception as err:
            print("Fail to restore...")
            print(err)

class PlayModel(Network):

    def __init__(self,modelname,sess,player,checkpoint_file):
        inUnits = 7*15
        fcUnits = [inUnits,256,512,512]
        outUnits = 364
        self.outUnits = outUnits
        self.name = modelname
        self.LearningRate = 1e-4
        
        #self.graph = tf.Graph()
        with tf.name_scope(modelname):
            self.x = tf.placeholder(tf.float32, [None, inUnits])
            self.keep_prob = tf.placeholder(tf.float32)
            
            fc_in = self.x
            for i in range(len(fcUnits)-1):
                fc_in = self.fc_layer(fc_in, fcUnits[i], fcUnits[i+1], self.keep_prob, name="fc"+str(i))
            
            self.out = self.out_layer(fc_in, fcUnits[-1], outUnits, name="out")
            self.y = tf.nn.softmax(self.out)
            self.y_ = tf.placeholder(tf.float32, [None, outUnits])
            self.rewards = tf.placeholder(tf.float32, [None])
            
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.out)
            self.loss = tf.reduce_sum(tf.multiply(self.cross_entropy, self.rewards))
            self.train_step = tf.train.AdamOptimizer(self.LearningRate).minimize(self.loss)
        
        super(PlayModel, self).__init__(modelname,sess,checkpoint_file)
        #print(tf.all_variables())
        self.trainBatch = []
        self.episodeTemp = []
        self.player = player
        
        #for var in tf.all_variables():
            #print(var)
    
    @staticmethod
    def cards2NumArray(cards):
        point = Hand.getCardPoint(cards)
        res = np.zeros(15)
        for p in point:
            res[p] += 1
        return res
    
    @staticmethod
    def ch2input(playerID,myCards,publicCards,history,lastPlay,lastLastPlay):
        net_input = []
        net_input.append(PlayModel.cards2NumArray(myCards))
        net_input.append(PlayModel.cards2NumArray(publicCards))
        player = playerID
        for _ in range(3):
            tmphis = []
            for his in history[player]:
                tmphis.extend(his)
            net_input.append(PlayModel.cards2NumArray(tmphis))
            player = (player - 1) % 3
        net_input.append(PlayModel.cards2NumArray(lastPlay))
        net_input.append(PlayModel.cards2NumArray(lastLastPlay))
        return np.array(net_input).flatten() + 1
    
    @staticmethod
    def getDisFromChain(chain,baseChain,maxNum):
        chaindis = chain - baseChain
        num = maxNum
        res = 0
        for _ in range(chaindis):
            res += num
            num -= 1
        return res
    
    # change cardPoints to the index of network output
    @staticmethod
    def cardPs2idx(cardPoints):
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
                idx += PlayModel.getDisFromChain(hand.chain,5,8)
                idx += hand.primal
                return idx
        elif hand.type == "Pair":
            if hand.chain == 1:
                return 16 + hand.primal
            else:
                idx = 144
                idx += PlayModel.getDisFromChain(hand.chain,3,10)
                idx += hand.primal
                return idx
        elif hand.type == "Trio":
            if hand.chain == 1:
                return 29 + hand.primal + hand.kickerNum * 13
            else:
                idx = 196
                idx += PlayModel.getDisFromChain(hand.chain,2,11)
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
            idx = PlayModel.cardPs2idx(hand)
            res[idx] = 1
        return res
    
    @staticmethod
    def getChainFromDis(dis,baseChain,maxNum):
        chain = baseChain
        num = maxNum
        while dis >= num:
            dis -= num
            num -= 1
            chain += 1
        return chain, dis
    
    # change index of one hot output to cardPoints
    # if possible hand has kickers, the first element will be dict
    @staticmethod
    def idx2CardPs(idx):
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
            chain,primal = PlayModel.getChainFromDis(idx-108,5,8)
            res = list(range(primal,primal+chain))
        elif idx <= 195: # Pair Chain
            chain,primal = PlayModel.getChainFromDis(idx-144,3,10)
            res = list(range(primal,primal+chain)) * 2
        elif idx <= 240: # Airplane without wings
            chain,primal = PlayModel.getChainFromDis(idx-196,2,11)
            res = list(range(primal,primal+chain)) * 3
        elif idx <= 285: # Airplane with small wings
            chain,primal = PlayModel.getChainFromDis(idx-241,2,11)
            res.append({"kickerNum":1,"chain":chain,"type":"Trio"})
            res.extend(list(range(primal,primal+chain)) * 3)
        elif idx <= 330: # Airplane with big wings
            chain,primal = PlayModel.getChainFromDis(idx-286,2,11)
            res.append({"kickerNum":2,"chain":chain,"type":"Trio"})
            res.extend(list(range(primal,primal+chain)) * 3)
        elif idx <= 341: # Shuttle without wings
            chain,primal = PlayModel.getChainFromDis(idx-331,2,11)
            res = list(range(primal,primal+chain)) * 4
        elif idx <= 352: # Shuttle with small wings
            chain,primal = PlayModel.getChainFromDis(idx-342,2,11)
            res.append({"kickerNum":1,"chain":chain,"type":"Four"})
            res.extend(list(range(primal,primal+chain)) * 4)
        elif idx <= 363: # Shuttle with big wings
            chain,primal = PlayModel.getChainFromDis(idx-353,2,11)
            res.append({"kickerNum":2,"chain":chain,"type":"Four"})
            res.extend(list(range(primal,primal+chain)) * 4)            
        return res
    
    # get possible actions
    def getAction(self,netinput,allonehot):
        output = self.y.eval(feed_dict={self.x:[netinput], self.keep_prob:1.0})
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
            if abs(allonehot[i] - 1) < 1e-6:
                sum += legalOut[i]
                #print(sum)
                if randf < sum:
                    return self.idx2CardPs(i)
        return self.idx2CardPs(np.argmax(allonehot))
    
    # store the train samples
    def storeSamples(self, netinput, action, isPass):
        actidx = PlayModel.cardPs2idx(Hand.getCardPoint(action))
        #hand = Hand(action)
        self.episodeTemp.append([netinput,actidx,0, isPass])
        #print(self.episodeTemp)
    
    # compute rewards and train
    def finishEpisode(self, turnscores, istrain=False):
        
        #print([d[2] for d in self.episodeTemp])
        for i in range(len(self.episodeTemp)):
            self.episodeTemp[i][2] = turnscores[i] / 100.0
        
        #print([d[2] for d in self.episodeTemp])
        #nowWinner = 0 if scores[0] > 0 else 1
        #if nowWinner != self.lastWinner:
        #print("Add to Train Batch...")
        for data in self.episodeTemp:
            if data[2] <= 0 or data[3] or not istrain:
                continue
            self.trainBatch.append(data)
            if len(self.trainBatch) == BatchSize:
                random.shuffle(self.trainBatch)
                self.trainModel()
                self.trainBatch = []
                        
        #self.lastWinner = nowWinner
        #print(self.trainBatch)
        self.episodeTemp = []
        #for i in range(3):
        
    # train the model
    def trainModel(self):
        print("Train for PlayModel and player=%d..." %(self.player))
        batch = self.trainBatch
        netinput = [d[0] for d in batch]
        actidxs = [d[1] for d in batch]
        rewards = [d[2] for d in batch]
        acts = np.zeros((BatchSize,self.outUnits))
        for i in range(BatchSize):
            acts[i][actidxs[i]] = 1
        #for i in range(BatchSize):
            #if rewards[i] >= 0:
             #   acts[i][actidxs[i]] = 1
            #elif rewards[i] < 0:
                #rewards[i] = -rewards[i]
                #randi = random.randint(0,self.outUnits-1)
                #acts[i][randi] = 1
        #print(actidxs)
        #print(rewards)        
        
        '''tmpvar = []
        for var in tf.global_variables():
            if var.name == self.name+"/out1/W:0" or var.name == self.name+"/out0/W:0" or var.name == self.name+"/out2/W:0":
                tmpvar.append(var)
        for var in tmpvar:
            print(self.sess.run(var))'''
        #vals = self.y.eval(feed_dict={self.x:netinput,self.keep_prob:1.0})
        #print(vals)
        #print(np.sum(vals))
        for _ in range(MaxEpoch):
            self.sess.run(self.train_step, feed_dict={self.x:netinput, self.keep_prob:TrainKeepProb, self.y_:acts, self.rewards:rewards})
        print(self.y.eval(feed_dict={self.x:netinput,self.keep_prob:1.0}))
        '''for var in tmpvar:
            print(self.sess.run(var))
        print(self.loss[player].eval(feed_dict={self.x:netinput, self.keep_prob:1.0, self.y_:acts, self.rewards:rewards}))'''

class ValueModel(Network):
    
    def __init__(self,modelname,sess,player,checkpoint_file):
        inUnits = 7*15
        fcUnits = [inUnits,512,512,512]
        outUnits = 364
        self.inUnits = inUnits
        self.outUnits = outUnits
        self.name = modelname
        self.learningRate = 1e-4
        
        #self.graph = tf.Graph()
        with tf.name_scope(modelname):
            self.x = tf.placeholder(tf.float32, [None, inUnits])
            #self.keep_prob = tf.placeholder(tf.float32)
            
            fc_in = self.x
            for i in range(len(fcUnits)-1):
                fc_in = self.fc_layer(fc_in, fcUnits[i], fcUnits[i+1], 1.0, name="fc"+str(i), initstd=0.1)
            
            self.out = self.out_layer(fc_in, fcUnits[-1], outUnits, name="out",  initstd=0.1)
            self.act = tf.placeholder(tf.float32, [None, outUnits])
            self.y = tf.reduce_sum(tf.multiply(self.out, self.act), reduction_indices=1)
            self.y_ = tf.placeholder(tf.float32, [None])
            
            self.loss = tf.reduce_mean(tf.square(self.y - self.y_))
            self.train_step = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
        
        super(ValueModel, self).__init__(modelname,sess,checkpoint_file)
        #print(tf.all_variables())
        self.trainBatch = deque()
        self.episodeTemp = []
        self.player = player
    
    # change all possible hands to one hot tensor
    def hand2one_hot(self,allhands):
        res = np.zeros(self.outUnits)
        for hand in allhands:
            idx = PlayModel.cardPs2idx(hand)
            res[idx] = 1
        return res    
    
    def getAction(self,netinput,allonehot,epsilon=None):
        output = self.out.eval(feed_dict={self.x:[netinput]})
        output = output.flatten() + BaseScore
        #print(output)
        legalOut = np.multiply(output, allonehot)
        #print(legalOut)
        minval = np.min(legalOut)
        if minval < 0:
            legalOut -= (minval-1)
            legalOut = np.multiply(legalOut,allonehot)
            #print(legalOut)
        allidx = [i for i,v in enumerate(allonehot) if v > 1e-6]
        #print(allidx)
        randf = random.random()
        #print(randf)
        outidx = -1
        if epsilon is None or randf > epsilon:
            outidx = np.argmax(legalOut)
        else:
            if len(allidx) == 0:
                print("Error!!!")
                print(allonehot)
                outidx = random.argmax(allonehot)
            else:
                outidx = random.choice(allidx)
        #print(outidx)
        return PlayModel.idx2CardPs(outidx)

    def probAction(self,netinput,allonehot):
        output = self.out.eval(feed_dict={self.x:[netinput]})
        output = output.flatten() + BaseScore
        #print(output)
        legalOut = np.multiply(output, allonehot)
        #print(legalOut)
        minval = np.min(legalOut)
        if minval < 0:
            print("Minus Value!!!")
            print(legalOut)
            legalOut -= (minval-1)
            legalOut = np.multiply(legalOut,allonehot)
        vals = [v for v in legalOut if v > 1e-6]
        actidxs = [i for i,v in enumerate(legalOut) if v > 1e-6]
        #print(actidxs)
        if len(vals) == 0:
            return PlayModel.idx2CardPs(np.argmax(legalOut))
        minval = np.min(vals)
        vals -= minval
        #print(vals)
        probs = np.exp(vals)/np.sum(np.exp(vals),axis=0)
        #print(probs)
        
        randf = random.random()
        #print(randf)
        sum = 0
        outidx = np.argmax(legalOut)
        for i in range(len(probs)):
            sum += probs[i]
            if randf < sum:
                outidx = actidxs[i]
                break
        #print(outidx)
        return PlayModel.idx2CardPs(outidx)
        
    def storeSamples(self,netinput,action,allonehot):
        actidx = PlayModel.cardPs2idx(Hand.getCardPoint(action))
        hand = Hand(action)
        self.episodeTemp.append([netinput,actidx,hand.getHandScore(),allonehot])

    def finishEpisode(self,score):      
        #print("Player %d add to the train batch"%(self.player))
        nlen = len(self.episodeTemp)
        for i in range(nlen,0,-1):
            data = self.episodeTemp[i-1]
            if i == nlen:
                self.trainBatch.append((data[0],data[1],np.zeros(self.inUnits),np.zeros(self.outUnits),score))
            else:
                ndata = self.episodeTemp[i]
                self.trainBatch.append((data[0],data[1],ndata[0],ndata[3],0))
        
        #print(self.trainBatch)
        while len(self.trainBatch) > BufferSize:
            self.trainBatch.popleft()
        if len(self.trainBatch) == BufferSize:
            self.trainModel()
        
        netinput = [d[0] for d in self.episodeTemp]
        actidxs = [d[1] for d in self.episodeTemp]
        acts = np.zeros((nlen,self.outUnits))
        for i in range(nlen):
            acts[i][actidxs[i]] = 1
        res = self.y.eval(feed_dict={self.x:netinput, self.act:acts})
        turnscores = res.flatten().tolist()
        #print(turnscores)
        turnscores = turnscores[1:]
        turnscores.append(score)
        '''for i in range(nlen):
            turnscores.append(score)
            score *= 1
        turnscores = turnscores[::-1]'''
        #print(turnscores)

        self.episodeTemp = []
        return turnscores                

    def trainModel(self):
        print("Train for ValueModel and player=%d..." %(self.player))
        batch = random.sample(self.trainBatch, BatchSize)
        netinput = [d[0] for d in batch]
        actidxs = [d[1] for d in batch]
        nextinput = [d[2] for d in batch]
        acts = np.zeros((BatchSize,self.outUnits))
        for i in range(BatchSize):
            acts[i][actidxs[i]] = 1
            
        nextscores = self.out.eval(feed_dict={self.x:nextinput})
        #print(nextscores)
        tvals = []
        for i in range(BatchSize):
            if abs(batch[i][4] - 0) < 1e-6:
                nscores = np.multiply(nextscores[i],batch[i][3])
                nscores = [s for s in nscores if abs(s) > 1e-6]
                if len(nscores) == 0:
                    tscore = 0
                else:
                    tscore = np.max(nscores)
                tvals.append(tscore * Gamma)
            else:
                tvals.append(batch[i][4])
        #print(tvals)
        #print(actidxs)       
        
        #print(self.y.eval(feed_dict={self.x:netinput, self.act:acts}))
        for _ in range(MaxEpoch):
            self.sess.run(self.train_step, feed_dict={self.x:netinput, self.act:acts, self.y_:tvals})
        #print(self.y.eval(feed_dict={self.x:netinput, self.act:acts}))
        
class KickersModel(Network):
    
    def __init__(self,modelname,sess,checkpoint_file):
        inUnits = 8*15
        fcUnits = [inUnits,512,512]
        outUnits = 28
        self.outUnits = outUnits
        self.name = modelname
        
        with tf.name_scope(modelname):
            self.x = tf.placeholder(tf.float32, [None, inUnits])
            self.keep_prob = tf.placeholder(tf.float32)
            
            fc_in = self.x
            for i in range(len(fcUnits)-1):
                fc_in = self.fc_layer(fc_in, fcUnits[i], fcUnits[i+1], self.keep_prob, name="fc"+str(i),initstd=0.01)
            
            self.out = self.out_layer(fc_in, fcUnits[-1], outUnits, name="out",initstd=0.01)
            self.y = tf.nn.softmax(self.out)
            self.y_ = tf.placeholder(tf.float32, [None, outUnits])
            self.rewards = tf.placeholder(tf.float32, [None])
            
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.out)
            self.loss = tf.reduce_sum(tf.multiply(self.cross_entropy, self.rewards))
            self.train_step = tf.train.AdamOptimizer(LearningRate).minimize(self.loss)
        
        super(KickersModel, self).__init__(modelname,sess,checkpoint_file)
        #print(tf.all_variables())
        self.trainBatch = []
        self.episodeTemp = []
    
    @staticmethod
    def ch2input(playmodel_input, primPoints):
        res = playmodel_input.tolist()
        prims = np.zeros(15)
        for p in primPoints:
            prims[p] += 1
        res.extend(prims)
        return np.array(res)
        
    def cardPs2idx(self,cardPoints):
        if len(cardPoints) == 1:
            return cardPoints[0]
        elif len(cardPoints) == 2 and cardPoints[0] < 13:
            return cardPoints[0] + 15
        else:
            return -1
    
    def allkickers2onehot(self,allkicers):
        res = np.zeros(self.outUnits)
        for k in allkicers:
            idx = self.cardPs2idx(k)
            res[idx] = 1
        return res        
    
    def idx2CardPs(self,idx):
        if idx < 15:
            return [idx]
        else:
            return [idx-15]*2
    
    def getKickers(self,netinput,allonehot):
        epsilon = 1e-6
        output = self.y.eval(feed_dict={self.x:[netinput], self.keep_prob:1.0})
        output = output.flatten()
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

    def storeSamples(self, netinput, playerID, kickers, turn):
        actidx = self.cardPs2idx(kickers)
        self.episodeTemp.append([netinput, playerID, actidx, turn])
     
    def finishEpisode(self,turnscores):
        for tmp in self.episodeTemp:
            t = tmp[3]
            p = tmp[1]
            tmp[3] = (turnscores[p][t]+BaseScore) / (2*BaseScore)
        
        #print([d[3] for d in self.episodeTemp])
        #print(len(self.trainBatch))
        for data in self.episodeTemp:
            if data[3] <= 0:
                continue
            self.trainBatch.append([data[0],data[2],data[3]])
            if len(self.trainBatch) >= KickersBatch:
                random.shuffle(self.trainBatch)
                self.trainModel()
                self.trainBatch = []
        
        self.episodeTemp = []
        
    def trainModel(self):
        print("Train for KickersModel...")
        batch = self.trainBatch
        netinput = [d[0] for d in batch]
        actidxs = [d[1] for d in batch]
        rewards = [d[2] for d in batch]
        acts = np.zeros((KickersBatch,self.outUnits))
        for i in range(KickersBatch):
            acts[i][actidxs[i]] = 1
        #print(actidxs)

        '''tmpvar = []
        for var in tf.global_variables():
            if var.name == self.name+"/out/W:0" :
                tmpvar.append(var)
        for var in tmpvar:
            print(self.sess.run(var))  '''      
 
        #vals = self.y.eval(feed_dict={self.x:netinput, self.keep_prob:1.0})
        #print(vals)
        for _ in range(MaxEpoch):
            self.sess.run(self.train_step, feed_dict={self.x:netinput, self.keep_prob:TrainKeepProb, self.y_:acts, self.rewards:rewards})
        vals = self.y.eval(feed_dict={self.x:netinput, self.keep_prob:1.0})
        print(vals)
        '''for var in tmpvar:
            print(self.sess.run(var))'''  
