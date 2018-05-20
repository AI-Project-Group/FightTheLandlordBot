import random
import numpy as np
import tensorflow as tf
from simulator import Hand

LearningRate = 1e-4
BatchSize = 20
Gamma = 0.95
MaxEpoch = 5
TrainKeepProb = 0.75
class exnetwork:
    def __init__(self,modelname,checkpoint_file):
        inUnits=8*15
        outUnits=28
        self.outUnits = outUnits
        self.name = modelname
        self.learning_rate=LearningRate   
        self.x = tf.placeholder(tf.float32, [None, inUnits])   
        self.y = tf.placeholder(tf.float32, [None, outUnits]) 
        #self.ob=tf.placeholder(tf.float32,[None, 6400])   
        layer1=tf.layers.dense(self.x, units=512,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.layer2=tf.layers.dense(layer1,units=1024,activation=tf.sigmoid,kernel_initializer=tf.contrib.layers.xavier_initializer())
        #self.sample_actions=tf.placeholder(tf.float32,[None, 1])
        self.rewards=tf.placeholder(tf.float32,[None],name='rewards')
        self.loss=tf.losses.log_loss(labels=self.y,predictions=self.layer2,weights=self.rewards)  
        op=tf.train.AdamOptimizer(self.learning_rate)
        self.train_op=op.minimize(self.loss)
        self.sess=tf.InteractiveSession()
        tf.global_variables_initializer().run()
        init=tf.global_variables_initializer()
        self.saver=tf.train.Saver()     
        with tf.Session() as sess:
            sess.run(init)
            save_path=self.saver.save(sess,"./ckptfiles/Pong.ckpt")
            print("Save to path: ",save_path)
    #def load_checkpoint(self):
        #print("...Loading checkpoint...")
        #self.saver.restore(self.sess, self.checkpoint_file)

    #def save_checkpoint(self):
        #print("...Saving checkpoint...")
        #self.saver.save(self.sess, self.checkpoint_file)
    
            #self.y = [tf.nn.softmax(self.out[i]) for i in range(3)]
            #self.y_ = tf.placeholder(tf.float32, [None, outUnits])
            #self.rewards = tf.placeholder(tf.float32, [None])
            
            #self.cross_entropy = [tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.out[i]) for i in range(3)]
            #self.loss = [tf.reduce_sum(tf.multiply(self.cross_entropy[i], self.rewards)) for i in range(3)]
            #self.identity = tf.placeholder(tf.float32,[None, 3])
            ##self.loss = tf.reduce_sum(tf.multiply(self.tmploss, self.identity))
            #self.train_step = [tf.train.AdamOptimizer(LearningRate).minimize(self.loss[i]) for i in range(3)]
            #self.train_step = tf.train.AdamOptimizer(LearningRate).minimize(self.loss)
            #print(tf.all_variables())
        
        #super(PlayModel, self).__init__(modelname,checkpoint_file)
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
    def extra_ch2input(self,playerID,myCards,publicCards,history,lastPlay,lastLastPlay,cardstoplay):
        extra_net_input=[]
        extra_net_input.append(self.cards2NumArray(myCards))
        extra_net_input.append(self.cards2NumArray(publicCards))
        player = playerID
        for _ in range(3):
            tmphis = []
            for his in history[player]:
                tmphis.extend(his)
            extra_net_input.append(self.cards2NumArray(tmphis))
            player = (player - 1) % 3
        extra_net_input.append(self.cards2NumArray(lastPlay))
        extra_net_input.append(self.cards2NumArray(lastLastPlay))
        ######加一维
        extra_net_input.append(cardstoplay)
        return np.array(extra_net_input).flatten()
    
    def getDisFromChain(self,chain,baseChain,maxNum):
        chaindis = chain - baseChain
        num = maxNum
        res = 0
        for _ in range(chaindis):
            res += num
            num -= 1
        return res
    
    # change cardPoints to the index of network output
    def extra_cardPs2idx(self,cardPoints):
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
        if hand.type == "Solo":
            return 0 + hand.primal
        elif hand.type == "Pair":
            return 15 + hand.primal
    
    # change all possible hands to one hot tensor
    def hand2one_hot(self,allhands):
        res = np.zeros(self.outUnits)
        for hand in allhands:
            idx = self.extra_cardPs2idx(hand)
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
        if idx <= 14: # Pass
            res = []
        elif idx <= 27: # Solo
            res = [idx - 14] *2          
        return res
    
    # get possible actions
    def getActions(self,netinput,playerID,allonehot):
        epsilon = 1e-6
        output = self.y[playerID].eval(feed_dict={self.x:[netinput]})
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
        actidx = self.extra_cardPs2idx(Hand.getCardPoint(action))
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
            self.sess.run(self.train_op[player], feed_dict={self.x:netinput, self.y:acts, self.rewards:rewards})
        '''for var in tmpvar:
            print(self.sess.run(var))
        print(self.loss[player].eval(feed_dict={self.x:netinput, self.keep_prob:1.0, self.y_:acts, self.rewards:rewards}))'''
        #exit(0)
