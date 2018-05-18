import random
import numpy as np
import tensorflow as tf
import json

LearningRate = 1e-4
BatchSize = 256
KickersBatch = 16
Gamma = 0.96
MaxEpoch = 1
TrainKeepProb = 0.8

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
        #print("Restore checkpoint...")
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
            self.train_step = tf.train.AdamOptimizer(LearningRate).minimize(self.loss)
        
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
        return np.array(net_input).flatten()
    
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
    def getAction(self,netinput,playerID,allonehot):
        epsilon = 1e-6
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
            if abs(allonehot[i] - 1) < epsilon:
                sum += legalOut[i]
                #print(sum)
                if randf < sum:
                    return self.idx2CardPs(i)
        return self.idx2CardPs(np.argmax(allonehot))
    
    # store the train samples
    def storeSamples(self, netinput, action, isPass):
        actidx = PlayModel.cardPs2idx(Hand.getCardPoint(action))
        hand = Hand(action)
        self.episodeTemp.append([netinput,actidx, hand.getHandScore()/100.0, isPass])
        #print(self.episodeTemp)
    
    # compute rewards and train
    def finishEpisode(self, score):
        
        #print([d[2] for d in self.episodeTemp])
        turnscores = []
        sum = score
        for tmp in self.episodeTemp[::-1]:
            #sum += tmp[2]
            tmp[2] = sum
            turnscores.append(sum)
            sum *= Gamma
        
        #print([d[2] for d in self.episodeTemp])
        #nowWinner = 0 if scores[0] > 0 else 1
        #if nowWinner != self.lastWinner:
        print("Add to Train Batch...")
        for data in self.episodeTemp:
            if data[2] == 0 or (data[1] == 0 and not data[3]):
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
        turnscores = turnscores[::-1]
        return turnscores
        
    # train the model
    def trainModel(self):
        print("Train for PlayModel and player=%d..." %(self.player))
        batch = self.trainBatch
        netinput = [d[0] for d in batch]
        actidxs = [d[1] for d in batch]
        rewards = [d[2] for d in batch]
        acts = np.zeros((BatchSize,self.outUnits))
        #for i in range(BatchSize):
            #acts[i][actidxs[i]] = 1
        for i in range(BatchSize):
            if rewards[i] >= 0:
                acts[i][actidxs[i]] = 1
            elif rewards[i] < 0:
                rewards[i] = -rewards[i]
                randi = random.randint(0,self.outUnits-1)
                acts[i][randi] = 1
        print(actidxs)
        print(rewards)        
        
        '''tmpvar = []
        for var in tf.global_variables():
            if var.name == self.name+"/out1/W:0" or var.name == self.name+"/out0/W:0" or var.name == self.name+"/out2/W:0":
                tmpvar.append(var)
        for var in tmpvar:
            print(self.sess.run(var))'''
        vals = self.y.eval(feed_dict={self.x:netinput,self.keep_prob:1.0})
        print(vals)
        print(np.sum(vals))
        for _ in range(MaxEpoch):
            self.sess.run(self.train_step, feed_dict={self.x:netinput, self.keep_prob:TrainKeepProb, self.y_:acts, self.rewards:rewards})
        print(self.y.eval(feed_dict={self.x:netinput,self.keep_prob:1.0}))
        '''for var in tmpvar:
            print(self.sess.run(var))
        print(self.loss[player].eval(feed_dict={self.x:netinput, self.keep_prob:1.0, self.y_:acts, self.rewards:rewards}))'''

class ValueModel(Network):
    
    def __init__(self,modelname,sess,player,checkpoint_file):
        inUnits = 7*15
        fcUnits = [inUnits,256,512,512]
        outUnits = 364
        self.outUnits = outUnits
        self.name = modelname
        self.learningRate = 1e-4
        
        #self.graph = tf.Graph()
        with tf.name_scope(modelname):
            self.x = tf.placeholder(tf.float32, [None, inUnits])
            #self.keep_prob = tf.placeholder(tf.float32)
            
            fc_in = self.x
            for i in range(len(fcUnits)-1):
                fc_in = self.fc_layer(fc_in, fcUnits[i], fcUnits[i+1], 1.0, name="fc"+str(i), mean=0.01, initstd=0.01)
            
            self.out = self.out_layer(fc_in, fcUnits[-1], outUnits, name="out", mean=0.01, initstd=0.01)
            self.act = tf.placeholder(tf.float32, [None, outUnits])
            self.y = tf.reduce_sum(tf.multiply(self.out, self.act), reduction_indices=1)
            self.y_ = tf.placeholder(tf.float32, [None])
            
            self.loss = tf.reduce_mean(tf.square(self.y - self.y_))
            self.train_step = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
        
        super(ValueModel, self).__init__(modelname,sess,checkpoint_file)
        #print(tf.all_variables())
        self.trainBatch = []
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
        output = output.flatten()
        #print(output)
        legalOut = np.multiply(output, allonehot)
        print(legalOut)
        allidx = [i for i,v in enumerate(allonehot) if v > 0]
        #print(allidx)
        randf = random.random()
        #print(randf)
        outidx = -1
        if epsilon is None or randf > epsilon:
            outidx = np.argmax(legalOut)
        else:
            outidx = random.choice(allidx)
        #print(outidx)
        return PlayModel.idx2CardPs(outidx),legalOut[outidx]     
        
    def storeSamples(self,netinput,action):
        actidx = PlayModel.cardPs2idx(Hand.getCardPoint(action))
        hand = Hand(action)
        self.episodeTemp.append([netinput,actidx,hand.getHandScore()])

    def finishEpisode(self,score):
        turnscores = []
        sum = score
        for tmp in self.episodeTemp[::-1]:
            sum += tmp[2]
            tmp[2] = sum
            turnscores.append(sum)
            sum *= Gamma
        
        #print([d[2] for d in self.episodeTemp])
        turns = len(self.episodeTemp)
        print("Add to Train Batch...")
        for data in self.episodeTemp:
            self.trainBatch.append(data)
            if len(self.trainBatch) == BatchSize:
                random.shuffle(self.trainBatch)
                self.trainModel()
                self.trainBatch = []            

        self.episodeTemp = []
        turnscores = turnscores[::-1]
        return turnscores                

    def trainModel(self):
        print("Train for ValueModel and player=%d..." %(self.player))
        batch = self.trainBatch
        netinput = [d[0] for d in batch]
        actidxs = [d[1] for d in batch]
        scores = [d[2] for d in batch]
        acts = np.zeros((BatchSize,self.outUnits))
        for i in range(BatchSize):
            acts[i][actidxs[i]] = 1
        print(actidxs)
        print(scores)        
        
        #print(self.y.eval(feed_dict={self.x:netinput, self.act:acts}))
        for _ in range(MaxEpoch):
            self.sess.run(self.train_step, feed_dict={self.x:netinput, self.act:acts, self.y_:scores})
        print(self.y.eval(feed_dict={self.x:netinput, self.act:acts}))
        
class KickersModel(Network):
    
    def __init__(self,modelname,sess,checkpoint_file):
        inUnits = 8*15
        fcUnits = [inUnits,512,1024]
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
            tmp[3] = turnscores[p][t] / 100.0
        
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
        print(actidxs)

        '''tmpvar = []
        for var in tf.global_variables():
            if var.name == self.name+"/out/W:0" :
                tmpvar.append(var)
        for var in tmpvar:
            print(self.sess.run(var))  '''      
 
        vals = self.y.eval(feed_dict={self.x:netinput, self.keep_prob:1.0})
        print(vals)
        for _ in range(MaxEpoch):
            self.sess.run(self.train_step, feed_dict={self.x:netinput, self.keep_prob:TrainKeepProb, self.y_:acts, self.rewards:rewards})
        vals = self.y.eval(feed_dict={self.x:netinput, self.keep_prob:1.0})
        print(vals)
        '''for var in tmpvar:
            print(self.sess.run(var))'''      
class FTLSimulator:

    # Simulation Functions
    # Initialization
    def __init__(self, nowPlayer, nowTurn, publicCard):
        # Format Descriptions
        self.nowPlayer = 0 # player 0: Landlord, 1: Farmer A(Jia), 2: Farmer B(Yi)
        self.cardCnt = [0, 0, 0] # the number of card every player now holds
        self.myCards = [] # What cards do I have NOW?
        self.publicCard = [] # 3 public cards owned by the landlord when dealing
        self.nowTurn = -1 # the present gaming turn (3 plays (including pass) each turn), -1 for not played yet
        self.history = [[], [], []] # history[p][k]: the card played by player p in the k-th turn

        # Operation Accessories
        self.isCardPlayed = [0] * 54 # isCardPlayed[i]: is card serial i played by any player
        self.cardsToFollow = [] # which hand shall I follow now ? empty for my turn to choose hand type

        self.nowPlayer = nowPlayer
        self.cardCnt = [20, 17, 17]
        self.nowTurn = nowTurn
        self.publicCard = publicCard
        self.lastPlay = []
        self.lastLastPlay = []
        # If I'm the landlord, public card is already included in my dealing

    # Serve cards
    def deal(self, cards):
        self.myCards.extend(cards)

    # Play cards, record history by sequential order
    def play(self, player, cards):
        self.cardCnt[player] -= len(cards)
        for card in cards:
            self.isCardPlayed[card] = 0
        # It's me playing
        if player == self.nowPlayer:
            for card in cards: # play my cards
                self.myCards.remove(card)
        # Record this play
        self.history[player].append(cards)
        self.lastLastPlay = self.lastPlay
        self.lastPlay = cards

    def setCardsToFollow(self, cards):
        self.cardsToFollow = cards

    # set history directly & retrieve current status
    # no need to set cards to follow any more
    def setHistory(self, historyData):
        lenh = [len(h) for h in historyData]
        for t in range(lenh[0]):
            for player in range(3):
                if t >= lenh[player]:
                    continue
                self.play(player, historyData[player][t])
        self.cardsToFollow = self.lastPlay or self.lastLastPlay
        # after this, self.history will be identical to historyData

class Hand:
    # point:       0 1 2 3 4 5 6 7  8 9 10 11 12 13 14
    # actual card: 3 4 5 6 7 8 9 10 J Q K  A  2  Jb Jr
    @staticmethod
    def getCardPoint(card):
        if isinstance(card, list): # a card list
            return [Hand.getCardPoint(c) for c in card]
        # a single card
        if card == 52:
            return 13
        elif card == 53:
            return 14
        else:
            return card//4

    # distinguish cards as hand patterns
    def __init__(self, cards, cardPoints=None):
        if isinstance(cards, Hand): # Copy instance
            self.type = cards.type
            self.primal = cards.primal
            self.kickerNum = cards.kickerNum
            self.chain = cards.chain
            return

        # Assume that cards has been sorted:
        cards.sort()
        # And, is a LEGAL type
        #self.cards = cards
        self.type = "None"
        self.primal = -1 # the basic card type, used for judging greater hand
        self.kickerNum = 0 # cards added to the basic type, 0: None, 1: Solo, 2: Pair
        # kicker element format: (point, type), type = 1 or 2
        self.chain = 1 # consecutive count. e.g. trio + solo && chain == 3 : Airplane with wings
        
        if cardPoints is not None:
            point = cardPoints
        else:
            point = Hand.getCardPoint(cards) # the point representing card points
        point.sort()

        if len(point) == 0: # passed
            self.type = "Pass"
        elif len(point) == 1: # Solo
            self.type = "Solo"
            self.primal = point[0]
        elif len(point) == 2: # Pair or Rocket
            if point[0] == 13: # Rocket
                self.type = "Rocket"
            elif point[0] == point[1]: # Pair
                self.type = "Pair"
                self.primal = point[0]
        else: # Above, types can be regarded as kickers
            pointU = list(set(point)) # delete duplicate
            pointU.sort()
            pointUCnt = [point.count(p) for p in pointU] # count the number of each point
            pattern = list(set(pointUCnt)) # get the pattern of point
            pattern.sort()
            # distinguish the pattern
            if pattern == [1]: # Solo chain
                self.type = "Solo"
                self.primal = pointU[0]
                self.chain = len(pointU)
            elif pattern == [2]: # Pair chain
                self.type = "Pair"
                self.primal = pointU[0]
                self.chain = len(pointU)
            elif pattern == [3]: # Trio, including airplane
                self.type = "Trio"
                self.primal = min(pointU)
                self.chain = len(pointU)
            elif len(pointU) % 2 == 0 and (pattern == [1, 3] or pattern == [2, 3]): # Trio + Solo/Pair, including airplane
                self.type = "Trio"
                self.primal = min([c for i, c in enumerate(pointU) if pointUCnt[i] == 3])
                self.chain = len(pointU) // 2
                self.kickerNum = pattern[0]
            elif pattern == [4]: # Bomb or Four chain (Shuttle)
                if len(pointU) == 1: # Only 1 point: Bomb
                    self.type = "Bomb"
                    self.primal = pointU[0]
                else: # Four chain
                    self.type = "Four"
                    self.primal = min(pointU)
                    self.chain = len(pointU)
            elif (pattern == [1, 4] or pattern == [2, 4]) and len(pointU) % 3 == 0: # Four + Dual Solo/Pair, including shuttle
                # originally, error when cards = [0,1,2,3,4,5]
                self.type = "Four"
                self.primal = min([c for i, c in enumerate(pointU) if pointUCnt[i] == 4])
                self.chain = len(pointU) // 3
                self.kickerNum = pattern[0]

    # get report string
    def report(self):
        return "%s From %d Len = %d KickerNum = %d" % (self.type, self.primal, self.chain, self.kickerNum)

    # compare two hands, is this pattern able to follow the other one : T / F
    def isAbleToFollow(self, other):
        if self.type == "Pass" and other.type == "Pass": # Must follow sth.
            return False
        if self.type == "Pass" or other.type == "Pass": # always can follow
            return True
        if other.type == "Rocket": # Nothing can follow
            return False
        if self.type == "Rocket": # surpasses every hand
            return True
        if self.type == "Bomb": # dispose bomb separately
            if other.type == "Bomb":
                return self.primal > other.primal
            return True # surpasses other hands
        if self.type != other.type: # mixed patterns
            return False
        return self.chain == other.chain and self.primal > other.primal
        
    def getHandScore(self):
        score = 0
        if self.type == "Pass":
            score = 0
        elif self.type == "Solo" and self.chain == 1:
            score = 1
        elif self.type == "Pair" and self.chain == 1:
            score = 2
        elif self.type == "Trio" and self.chain == 1:
            score = 4
        elif self.type == "Solo" and self.chain >= 5:
            score = 6
        elif self.type == "Pair" and self.chain >= 3:
            score = 6
        elif self.type == "Trio" and self.chain >= 2:
            score = 8
        elif self.type == "Four" and self.chain == 1:
            score = 8
        elif self.type == "Bomb":
            score = 10
        elif self.type == "Four" and self.chain == 2:
            score = 10
        elif self.type == "Rocket":
            score = 16
        elif self.type == "Four" and self.chain > 2:
            score = 20
        return score #/ 100.0


class CardInterpreter:
    # get the written name of the card
    @staticmethod
    def getCardName(cid):
        if isinstance(cid, list): # a card list
            return str([CardInterpreter.getCardName(c) for c in cid])
        if cid == 52:
            return "Black Joker"
        elif cid == 53:
            return "Red Joker"
        else:
            return ("Heart","Diamond","Spade","Club")[cid%4] + " " + \
                ("10" if cid//4 == 7 else "3456789_JQKA2"[cid//4])

    # parse the written name into id, for debug only
    @staticmethod
    def getCardID(cardName):
        cardName.lower()
        cardName.replace(" ","")
        if cardName == "jb" or cardName == "blackjoker":
            return 52
        if cardName == "jr" or cardName == "redjoker":
            return 53
        lastNum = {"3":0,"4":1,"5":2,"6":3,"7":4,"8":5,"9":6,"0":7,"j":8,"q":9,"k":10,"a":11,"2":12}[cardName[-1]]
        colorNum = {"h":0,"d":1,"s":2,"c":3}[cardName[0]]
        return lastNum * 4 + colorNum

    # get all possible hands in cards, meanwhile get all solos and pairs
    # cards: [card_s], cardsToFollow: the card set to follow
    # assume that cards is already sorted
    # return the point representation, may NOT sorted
    # when the hand has kickers, like "Trio" or "Four", the first element will be kickers
    @staticmethod
    def splitCard(cards, cardsToFollow = []):
        allHands = [[]] # all legal cards in points, originally 'pass' is a legal actions
        # *** colors, kickers, difference between bomb and four are not considered
        lastHand = Hand(cardsToFollow)
        # solos records all solo kickers
        # pairs records all pair kickers
        if lastHand.type == "Rocket": # can't follow
            return allHands

        # deal with Rocket
        if 52 in cards and 53 in cards: # Rocket
            allHands.append([13,14])

        # No Joker from now on, deal with bombs
        point = Hand.getCardPoint(cards) # get the point expressions
        pointU = list(set(point)) # delete duplicate
        pointUCnt = [point.count(p) for p in pointU] # count the number of each point

        # including card 2 (point == 12)
        soloRec = pointU[:] # including joker cards
        pairRec = [p for i, p in enumerate(pointU) if pointUCnt[i] >= 2] # p >= 13: no pair/trio/four
        trioRec = [p for i, p in enumerate(pointU) if pointUCnt[i] >= 3]
        fourRec = [p for i, p in enumerate(pointU) if pointUCnt[i] >= 4] # Bombs or Four + Dual Solo/Pair

        if lastHand.type == "Bomb": # append specific bombs
            allHands.extend([[pr]*4 for pr in fourRec if pr > lastHand.primal])
            return allHands

        allHands.extend([[pr]*4 for pr in fourRec]) # append bombs

        if lastHand.type == "Four": # kickers appended
            # if lastHand is a shuttle, then here all bombs are listed
            for i,c in enumerate(fourRec):
                if lastHand.chain == 1 and c > lastHand.primal:
                    kickers = CardInterpreter.getKickers(cards, lastHand.kickerNum, [c])
                    if len(kickers) >= 2:
                        allHands.append([kickers,c,c,c,c])
            return allHands

        # relative to xxxRec lists
        soloChainCnt = [1]
        pairChainCnt = [1] if pairRec else []
        trioChainCnt = [1] if trioRec else [] # airplane with no wings
        # do not consider four chains (shuttle)

        # card 2 (point 12), joker cards only in chain 1
        for i, p in enumerate(soloRec):
            if i: # start from the second place
                soloChainCnt.append(soloChainCnt[i-1] + 1 if p == soloRec[i-1] + 1 and p < 12 else 1)
        for i, p in enumerate(pairRec):
            if i: # start from the second place
                pairChainCnt.append(pairChainCnt[i-1] + 1 if p == pairRec[i-1] + 1 and p < 12 else 1)
        for i, p in enumerate(trioRec):
            if i: # start from the second place
                trioChainCnt.append(trioChainCnt[i-1] + 1 if p == trioRec[i-1] + 1 and p < 12 else 1)

        # Record Chain
        if lastHand.type == "Solo":
            for i, c in enumerate(soloRec):
                if soloChainCnt[i] >= lastHand.chain and c-soloChainCnt[i]+1 > lastHand.primal: # able to follow
                    allHands.append(list(range(c-lastHand.chain+1, c+1)))
            return allHands
        if lastHand.type == "Pair":
            for i, c in enumerate(pairRec):
                if pairChainCnt[i] >= lastHand.chain and c-pairChainCnt[i]+1 > lastHand.primal: # able to follow
                    allHands.append(list(range(c-lastHand.chain+1, c+1))*2)
            return allHands
        if lastHand.type == "Trio":
            for i, c in enumerate(trioRec):
                if trioChainCnt[i] >= lastHand.chain and c-trioChainCnt[i]+1 > lastHand.primal: # able to follow
                    kickers = CardInterpreter.getKickers(cards, lastHand.kickerNum, list(range(c-lastHand.chain+1, c+1)))
                    if len(kickers) >= lastHand.chain:
                        allHands.append([kickers])
                        allHands[-1].extend(list(range(c-lastHand.chain+1, c+1))*3)
            return allHands

        # Here, lastHand.type = "Pass". you can play any type you want
        allHands.remove([]) # remove "Pass" actions, you must take out some cards
        for i, c in enumerate(soloRec): # record solo and solo chains (>=5)
            allHands.append([c])
            if soloChainCnt[i] >= 5: # able to play
                for length in range(5, soloChainCnt[i]+1):
                    allHands.append(list(range(c-length+1, c+1)))
        for i, c in enumerate(pairRec): # record pair and pair chains (>=3)
            allHands.append([c]*2)
            if pairChainCnt[i] >= 3: # able to play
                for length in range(3, pairChainCnt[i]+1):
                    allHands.append(list(range(c-length+1, c+1))*2)
        for i, c in enumerate(trioRec): # record trio and trio chains i.e. airplane (>=2)
            allHands.append([c]*3)
            for knum in range(1,3):
                kickers = CardInterpreter.getKickers(cards, knum, [c])
                if len(kickers) > 0:
                    allHands.append([kickers])
                    allHands[-1].extend([c]*3)
            if trioChainCnt[i] >= 2: # able to play
                for length in range(2, trioChainCnt[i]+1):
                    for knum in range(1,3):
                        kickers = CardInterpreter.getKickers(cards, knum, list(range(c-length+1, c+1)))
                        if len(kickers) >= length:
                            allHands.append([kickers])
                            allHands[-1].extend(list(range(c-length+1, c+1))*3)
        for i, c in enumerate(fourRec):
            for knum in range(1,3):
                kickers = CardInterpreter.getKickers(cards, knum, [c])
                if len(kickers) >= 2:
                    allHands.append([kickers])
                    allHands[-1].extend([c]*4)

        return allHands
    
    @staticmethod
    def getKickers(cards, kickerNum, primCards):
        point = Hand.getCardPoint(cards) # get the point expressions
        pointU = list(set(point)) # delete duplicate
        res = []

        # including card 2 (point == 12)
        if kickerNum == 1:
            res = [[p] for p in pointU] # including joker cards
            for p in primCards:
                if [p] in res:
                    res.remove([p])

        if kickerNum == 2:
            pointUCnt = [point.count(p) for p in pointU] # count the number of each point
            res = [[p]*2 for i, p in enumerate(pointU) if pointUCnt[i] >= 2] # p >= 13: no pair/trio/four
            for p in primCards:
                if [p,p] in res:
                    res.remove([p,p])
        return res

    # Given a hand of point set, select out a set of card of this pattern
    @staticmethod
    def selectCardByHand(cards, pointSet):
        cardsNotUsed = cards[:]
        pointNotUsed = Hand.getCardPoint(cards)
        cardsSelected = []
        for p in pointSet:
            index = pointNotUsed.index(p)
            cardsSelected.append(cardsNotUsed[index])
            cardsNotUsed.pop(index)
            pointNotUsed.pop(index)

        return cardsSelected

# Fight The Landlord executor

# Initialization using the JSON input
class FTLBot:
    def __init__(self, playmodel, kickersmodel, data, dataType = "Judge"):
        self.dataType = dataType
        self.playmodel = playmodel
        self.kickersmodel = kickersmodel
        if dataType == "JSON": # JSON req
            rawInput = json.loads(data)
            rawRequest = rawInput["requests"]
            initInfo = rawRequest[0]
            initHistory = initInfo["history"]
            myBotID = 2 if len(initHistory[0]) else 1 if len(initHistory[1]) else 0
            myCards = initInfo["own"]
            myCards.sort()
            publicCard = initInfo["publiccard"]
            nowTurn = len(rawInput["responses"])

            # Construct game simulator
            sim = FTLSimulator(myBotID, nowTurn, publicCard)
            sim.deal(myCards)

            # Retrace the history plays
            # first request
            if myBotID == 1:
                sim.play(0, initHistory[1])
            if myBotID == 2:
                sim.play(0, initHistory[0])
                sim.play(1, initHistory[1])
            # following rounds
            roundN = len(rawInput["responses"])
            for rnd in range(roundN):
                otherPlayerHistory = rawRequest[rnd+1]["history"]
                plays = [rawInput["responses"][rnd], otherPlayerHistory[0], otherPlayerHistory[1]]
                for turn in range(3):
                    #print(str((myBotID+turn)%3)+" Plays "+simulator.Hand(plays[turn]).report())
                    sim.play((myBotID+turn)%3, plays[turn])

            # check if this is my turn
            lastHistory = rawRequest[roundN]["history"]
            sim.setCardsToFollow(lastHistory[1] or lastHistory[0])
            self.simulator = sim

        elif dataType == "Judge": # data from judgement
            sim = FTLSimulator(data["ID"], data["nowTurn"], data["publicCard"])
            sim.deal(data["deal"])
            sim.setHistory(data["history"])
            self.simulator = sim

    def makeData(self, data, val="0"):
        if self.dataType == "JSON":
            return json.dumps({"response": data,"debug":{"val":val}})
        elif self.dataType == "Judge":
            return data

    # Return the decision based on type
    def makeDecision(self):
        lastHand = Hand(self.simulator.cardsToFollow)   
        possiblePlays = CardInterpreter.splitCard(self.simulator.myCards, lastHand)
        #print(possiblePlays)

        # @TODO You need to modify the following part !!
        # A little messed up ...
        if not len(possiblePlays):
            return self.makeData([])
        
        sim = self.simulator
        net_input = PlayModel.ch2input(sim.nowPlayer,sim.myCards,sim.publicCard,sim.history,sim.lastPlay,sim.lastLastPlay)
        one_hot_t = self.playmodel.hand2one_hot(possiblePlays)
        choice,val = self.playmodel.getAction(net_input, one_hot_t)
        #print(choice)

        # Add kickers, if first element is dict, the choice must has some kickers
        # @TODO get kickers from kickers model
        if choice and isinstance(choice[0],dict):
            tmphand = choice[1:]
            kickers_input = self.kickersmodel.ch2input(net_input,tmphand)
            allkickers = CardInterpreter.getKickers(sim.myCards, choice[0]["kickerNum"], list(set(tmphand)))
            kickers_onehot = self.kickersmodel.allkickers2onehot(allkickers)
            num = choice[0]['chain']
            if choice[0]["type"] == "Four":
                num *= 2
            kickers = []
            for _ in range(num):
                kickers.append(self.kickersmodel.getKickers(kickers_input,kickers_onehot))
                kidx = self.kickersmodel.cardPs2idx(kickers[-1])
                kickers_onehot[kidx] = 0
            for k in kickers:
                tmphand.extend(k)
                self.kickersmodel.storeSamples(kickers_input,sim.nowPlayer,k,sim.nowTurn)
            #print(self.kickersmodel.episodeTemp)
            choice = tmphand
        cardChoice = CardInterpreter.selectCardByHand(self.simulator.myCards, choice)
        
        self.playmodel.storeSamples(net_input,cardChoice)

        # You need to modify the previous part !!
        return self.makeData(cardChoice,val)

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    playmodel = [ValueModel("val"+str(i),sess,i,"data/FTL/test.ckpt") for i in range(3)]
    kickersmodel = KickersModel("kick",sess,"data/FTL/test.ckpt")
    tf.global_variables_initializer().run()
    kickersmodel.load_model()
    player = FTLBot(playmodel[0], kickersmodel, input(), "JSON")
    id = player.simulator.nowPlayer
    player.playmodel = playmodel[id]
    res = player.makeDecision()
    print(res)