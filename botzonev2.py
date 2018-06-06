import random,copy
import numpy as np
import tensorflow as tf
import json

class DuelingDQN:
    def __init__(
            self,
            n_features,
            n_actions,
            modelname,
            sess,
            learning_rate=1e-4,
            reward_decay=0.97,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=3200,
            batch_size=128,
            e_greedy_increment=None,
            dueling=True,
            output_graph=False,
            BaseScore=0,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.modelname = modelname
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.BaseScore = BaseScore
        
        self.n_l1 = 512
        self.n_l2 = 512
        
        self.dueling = dueling
        self.lr = learning_rate
       
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = []#Memory(capacity=memory_size)

        # consist of [target_net, evaluate_net]
        with tf.variable_scope(self.modelname):
            self._build_net()
        
        self.sess = sess

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []
    
    #修改
    def _build_net(self):
        def build_layers(s, c_names, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, self.n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, self.n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
                
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [self.n_l1, self.n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            
            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w3 = tf.get_variable('w3', [self.n_l2, 1], initializer=w_initializer, collections=c_names)
                    b3 = tf.get_variable('b3', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l2, w3) + b3
                   
                with tf.variable_scope('Advantage'):
                    w3 = tf.get_variable('w3', [self.n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                    b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l2, w3) + b3
                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True))     # Q = V(s) + A(s,a)
              
            else:
                with tf.variable_scope('l3'):
                    w3 = tf.get_variable('w3', [self.n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                    b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l2, w3) + b3
                    
            return out

        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        
        self.action_possible = tf.placeholder(tf.float32, [None, self.n_actions], name='action_possible')  # input Action
        self.ISWeights = tf.placeholder(tf.float32, [None,], name='IS_weights')

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)  # config of layers
            self.q_eval = build_layers(self.s, c_names, w_initializer, b_initializer)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            q_next = build_layers(self.s_, c_names, w_initializer, b_initializer)
            self.q_next = q_next + self.BaseScore
            #print(self.q_next.shape)

        with tf.variable_scope('q_target'):
            q_next_max = tf.reduce_max(self.q_next*self.action_possible, axis=1, name='Qmax_s_')
            #print(q_next_max.shape)
            q_target = self.r + self.gamma * (q_next_max - self.BaseScore)    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
            #print(self.q_target.shape)

        with tf.variable_scope('q_eval'):
            a_one_hot = tf.one_hot(self.a, depth=self.n_actions, dtype=tf.float32)
            self.q_eval_wrt_a = tf.reduce_sum(self.q_eval * a_one_hot, axis=1)     # shape=(None, )
            #print(self.q_eval_wrt_a.shape)

        with tf.variable_scope('loss'):
            self.abs_errors = tf.abs(self.q_target - self.q_eval_wrt_a)    # for updating Sumtree
            #print(self.abs_errors.shape)
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_, actions_one_hot,player=None,initr=None):
        #print((s,[a, r], s_, actions_one_hot))
        if player is None and initr is None:
            transition = np.hstack((s, [a, r], s_, actions_one_hot))
        else:
            transition = np.hstack((s, [a, r], [player,initr], s_, actions_one_hot))
        self.memory.store(transition)
    
    def get_action(self, netinput, actions_one_hot, norand=False, addNonZero=0):
        output = self.sess.run(self.q_eval,feed_dict={self.s:[netinput]})
        output = output.flatten() + self.BaseScore + addNonZero
        output[0] -= addNonZero
        legalOut = np.multiply(output, actions_one_hot)
        #print(legalOut)
        minval = np.min(legalOut)
        if minval < 0:
            legalOut -= (minval-1)
            legalOut = np.multiply(legalOut,actions_one_hot)
        #print(legalOut)
        allidx = [i for i,v in enumerate(actions_one_hot) if v > 1e-6]
        #print(allidx)
        randf = random.random()
        #print(randf)
        #print(self.epsilon)
        outidx = -1
        if norand or randf < self.epsilon:
            outidx = np.argmax(legalOut)
        else:
            outidx = random.choice(allidx)
        #print(outidx)
        return outidx, legalOut[outidx]

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
    
    def learn(self,iskickers=False):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget_params_replaced\n')

        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        
        ins = batch_memory[:, :self.n_features]
        ina = batch_memory[:, self.n_features]
        inr = batch_memory[:, self.n_features+1]
        if not iskickers:
            ins_ = batch_memory[:, self.n_features+2:self.n_features+2+self.n_features]
            inaction_possible = batch_memory[:, self.n_features+2+self.n_features:]
        else:
            #print(inr)
            inr = inr + self.BaseScore*self.gamma
            ins_ = np.zeros((self.batch_size,self.n_features))
            inaction_possible = np.zeros((self.batch_size,self.n_actions))
        
        _,loss = self.sess.run([self._train_op,self.loss],
                          feed_dict={self.s: ins, self.a: ina, self.r: inr, self.s_: ins_, 
                                     self.action_possible: inaction_possible, self.ISWeights: ISWeights})
        abs_errors = self.sess.run(self.abs_errors, 
                                   feed_dict={self.s: ins, self.a: ina, self.r: inr, self.s_: ins_,
                                              self.action_possible: inaction_possible})
        
        #print(abs_errors)
        for i in range(len(tree_idx)):  # update priority
            idx = tree_idx[i]
            self.memory.update(idx, abs_errors[i]/50.0)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        
        return loss
        
    #新增
    def save_model(self,path,ckptname):
        saver = tf.train.Saver() 
        saver.save(self.sess, path+ckptname+".ckpt") 

    #新增
    def load_model(self,path,ckptname):
        saver = tf.train.Saver() 
        try:
            saver.restore(self.sess, path+ckptname+".ckpt")
        except Exception as err:
            print(err)
            print("Fail to restore!")
        
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

class PlayModel(DuelingDQN):

    def __init__(self,modelname,sess,player):
        self.player = player
        super(PlayModel, self).__init__(105+364,364,modelname,sess,memory_size=3200,batch_size=128,e_greedy_increment=0.9/2e5)
        self.episodeTemp = []

    @staticmethod
    def cards2NumArray(cards):
        point = Hand.getCardPoint(cards)
        res = np.zeros(15)
        for p in point:
            res[p] += 1
        return res
    
    @staticmethod
    def ch2input(playerID,myCards,publicCards,history,lastPlay,lastLastPlay,actions_one_hot):
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
        net_input = np.array(net_input).flatten()
        net_input = np.hstack([net_input,actions_one_hot])
        return net_input + 1
    
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
        res = np.zeros(self.n_actions)
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
        
class KickersModel(DuelingDQN):
    
    def __init__(self,modelname,sess):
        super(KickersModel, self).__init__(105+364+15,28,modelname,sess,memory_size=320,batch_size=16,e_greedy_increment=0.9/1e5)
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
        res = np.zeros(self.n_actions)
        for k in allkicers:
            idx = self.cardPs2idx(k)
            res[idx] = 1
        return res        
    
    def idx2CardPs(self,idx):
        if idx < 15:
            return [idx]
        else:
            return [idx-15]*2
        
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
            #print(pattern)
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
            score = 2 #+ 1
        elif self.type == "Trio" and self.chain == 1:
            score = 4 #+ 2
        elif self.type == "Solo" and self.chain >= 5:
            score = 6 #+ self.chain
        elif self.type == "Pair" and self.chain >= 3:
            score = 6 #+ self.chain * 2
        elif self.type == "Trio" and self.chain >= 2:
            score = 8 #+ self.chain * 3
        elif self.type == "Four" and self.chain == 1:
            score = 8 #+ self.chain * 4
        elif self.type == "Bomb":
            score = 10 #+ 10
        elif self.type == "Four" and self.chain == 2:
            score = 10 #+ self.chain * 4
        elif self.type == "Rocket":
            score = 16 #+ 16
        elif self.type == "Four" and self.chain > 2:
            score = 20 #+ self.chain * 4
        return score * 2 #/ 100.0


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
                    allHands.append(list(range(c-length+1, c+1))*3)
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

SoloPairScore = [[60,56,52,48,44,40,36,32,28,24,20,12,4,1,0],
                 [40,38,36,34,30,28,26,24,20,16,12,6,2,0,0]]

# Initialization using the JSON input
class FTLBot:
    def __init__(self, playmodel,kickersmodel, data, dataType = "Judge", norand=False, addHuman=False):
        self.dataType = dataType
        self.playmodel = playmodel
        #self.valuemodel = valuemodel
        self.kickersmodel = kickersmodel
        self.norand = norand
        self.addHuman = addHuman
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
                    #print(str((myBotID+turn)%3)+" Plays "+Hand(plays[turn]).report())
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

    def makeData(self, data,val=0):
        if self.dataType == "JSON":
            return json.dumps({"response": data,"debug":{"val":val}})
        elif self.dataType == "Judge":
            return data
            
    @staticmethod
    def maxValueKickers(solos,pairs,sknum,pknum):
        initsolos = copy.deepcopy(solos)
        initpairs = copy.deepcopy(pairs)
        initsolos.sort()
        initpairs.sort()
        tmpsknum = copy.deepcopy(sknum)
        tmpsknum.sort(key=lambda x:x[0],reverse=True)
        tmppknum = pknum[:]
        bval = 0
        success = True
        while True:
            tmpsolos = copy.deepcopy(initsolos)
            tmppairs = copy.deepcopy(initpairs)
            #print(tmpsolos)
            #print(tmppairs)
            try:
                for pnum in tmppknum:
                    for _ in range(pnum):
                        tmpc = tmppairs[0]
                        tmppairs.remove(tmpc)   
            except Exception as err:
                success = False
                break                        
            try:
                for snum,cset in tmpsknum:
                    lastc = -1
                    for _ in range(snum):
                        i = 0
                        tmpc = tmpsolos[0][i]                        
                        while tmpc == lastc or tmpc in cset:
                            i += 1
                            tmpc = tmpsolos[0][i]
                        lastc = tmpc
                        tmpsolos.remove([tmpc])
            except Exception as err:
                if len(initpairs) == 0:
                    success = False
                    break
                tmpp = initpairs[0]
                initsolos.extend([[tmpp[0]]]*2)
                initsolos.sort()
                initpairs.remove(tmpp)
                continue
            val = 1000
            for s in tmpsolos:
                val -= SoloPairScore[0][s[0]]
            for p in tmppairs:
                val -= SoloPairScore[1][p[0]]
            if val > bval:
                bval = val
            else:
                break
        if success:
            return success,bval,initsolos,initpairs
        else:
            return success,10,initsolos,initpairs

    @staticmethod
    def searchHuman(cards,sknum,pknum,bonus=0,selectHand=None):
        possiblePlays = CardInterpreter.splitCard(cards, [])
        solos = []
        pairs = []
        bombs = []
        twos = []
        for p in possiblePlays:
            lenp = len(p)
            if lenp == 1:
                solos.append(p)
            elif lenp == 2:
                if p[0] == 13:
                    bombs.append(p)
                else:
                    pairs.append(p)
            #elif lenp == 4 and not isinstance(p[0],list):
                #bombs.append(p)
            elif selectHand is None and isinstance(p[0],list) and p[1] == 12:
                twos.append(p)
            elif selectHand is None and lenp == 3 and p[0] == 12:
                twos.append(p)
        for p in solos:
            possiblePlays.remove(p)
        for p in pairs:
            possiblePlays.remove(p)
        for p in bombs:
           possiblePlays.remove(p)
        for p in twos:
            possiblePlays.remove(p)
        #print(possiblePlays)
        #print(solos)
        #print(pairs)
        #print(bombs)
        #print(twos)
        #print(bombs)
        if len(possiblePlays) == 0:
            for p in pairs:
                if p[0] == 12 and len(twos) != 0:continue
                solos.remove([p[0]])
            for p in bombs:
                if p[0] == 13: solos.remove([p[1]])
                solos.remove([p[0]])
            success,val,solos,pairs = FTLBot.maxValueKickers(solos,pairs,sknum,pknum)
            '''print(solos)
            print(pairs)
            print(bonus)'''
            val += bonus
            #print(val)
            #print(val)
            return success,val,[],solos,pairs,bombs
        maxsuccess = False
        maxval = 0
        maxlist = []
        maxsolos = []
        maxpairs = []
        maxchoice = []
        maxbombs = []
        for p in possiblePlays:
            if selectHand is not None and p != selectHand:
                continue
            nsknum = copy.deepcopy(sknum)
            npknum = pknum[:]
            nbonus = bonus
            if isinstance(p[0],list):
                tmpc = p[1:]             
                lenc = len(tmpc)
                if lenc % 3 == 0:
                    chain = lenc // 3
                else:
                    chain = 2*(lenc // 4)
                if len(p[0][0]) == 1:
                    nsknum.append([chain,list(set(tmpc))])
                else:
                    npknum.append(chain)
            else:
                tmpc = p
            nextcards = cards[:]           
            cardsc = CardInterpreter.selectCardByHand(nextcards, tmpc)
            for c in cardsc:
                nextcards.remove(c)
            hand = Hand(cardsc)
            if hand.type == "Bomb" and not isinstance(p[0],list):
                nbonus += 90
            elif hand.type == "Trio" and hand.chain > 1:
                nbonus += hand.chain*hand.chain*10
            elif hand.type == "Four" and hand.chain > 1:
                nbonus += hand.chain*hand.chain*10
            '''print(p)
            print(tmpc)
            print(nextcards)
            print(nsknum)
            print(npknum)
            print(nbonus)'''
            success,tval,tlist,tsolos,tparis,tbombs = FTLBot.searchHuman(nextcards,nsknum,npknum,nbonus)
            if tval > maxval:
                maxsuccess = success
                maxval = tval
                maxchoice = p
                maxlist = tlist
                maxsolos = tsolos
                maxpairs = tparis
                maxbombs = tbombs
        nchoice = len(maxchoice)
        if nchoice == 4 and not isinstance(maxchoice[0],list):
            maxbombs.append(maxchoice)
        elif nchoice == 5 and isinstance(maxchoice[0],list):
            maxbombs.append(maxchoice[1:])
        else:
            maxlist.append(maxchoice)
        return maxsuccess,maxval,maxlist,maxsolos,maxpairs,maxbombs
    
    def isAddHuman(self):
        if self.addHuman:
            return True
            
    # Return the decision based on type
    def makeDecision(self):
        sim = self.simulator
        lastHand = Hand(sim.cardsToFollow)
        possiblePlays = []
        usedHuman = False
        if self.addHuman and (lastHand.type == "Pass" or len(sim.cardsToFollow) == 1 or len(sim.cardsToFollow) == 2):
            # Human Policy
            success,maxval,pPlays,psolos,ppairs,pbombs = self.searchHuman(self.simulator.myCards,[],[])
            if lastHand.type == "Pass" and success:
                possiblePlays = pPlays
                if possiblePlays:
                    possiblePlays.extend(psolos)
                    possiblePlays.extend(ppairs)
            elif success:
                if lastHand.type == "Solo":possiblePlays=psolos
                else:possiblePlays=ppairs
                ablelist = []
                for p in possiblePlays:
                    nowHand = Hand([],p)
                    if nowHand.isAbleToFollow(lastHand):
                        ablelist.append(p)
                possiblePlays = ablelist
                if possiblePlays:
                    possiblePlays.append([])
                    possiblePlays.extend(pbombs)
            #print("Search Human!!!")
            #print(possiblePlays)
        if possiblePlays == []:
            possiblePlays = CardInterpreter.splitCard(self.simulator.myCards, lastHand)
        else:
            usedHuman = True
        #print(possiblePlays)
        
        addNonZero = 0
        if self.addHuman:
            if sim.nowPlayer == 0 and (sim.cardCnt[1] <= 2 or sim.cardCnt[2] <= 2):
                addNonZero += 50
            elif sim.nowPlayer == 1 and sim.cardCnt[0] <= 2 and sim.lastPlay:
                addNonZero += 50
            elif sim.nowPlayer == 2 and sim.cardCnt[0] <= 2 and sim.lastPlay == []:
                addNonZero += 50
        #print(addNonZero)
                      
        if not len(possiblePlays):
            return self.makeData([])
        
        one_hot_t = self.playmodel.hand2one_hot(possiblePlays)
        net_input = self.playmodel.ch2input(sim.nowPlayer,sim.myCards,sim.publicCard,sim.history,sim.lastPlay,sim.lastLastPlay,one_hot_t)
        actidx,val = self.playmodel.get_action(net_input, one_hot_t, self.norand, addNonZero)
        choice = self.playmodel.idx2CardPs(actidx)

        # Add kickers, if first element is dict, the choice must has some kickers
        # get kickers from kickers model
        if choice and isinstance(choice[0],dict):
            tmphand = choice[1:]
            allkickers = CardInterpreter.getKickers(sim.myCards, choice[0]["kickerNum"], list(set(tmphand)))
            if self.addHuman:
                tmpchoice = choice[:]
                tmpchoice[0] = allkickers
                success,maxval,pPlays,psolos,ppairs,_ = self.searchHuman(sim.myCards,[],[],0,tmpchoice)
                if success:
                    if choice[0]["kickerNum"] == 1:
                        allkickers = psolos
                    else:
                        allkickers = ppairs
                    pointU = list(set(tmphand))
                    for p in pointU:
                        if [p] in allkickers:
                            allkickers.remove([p])
                        if [p]*2 in allkickers:
                            allkickers.remove([p]*2)
            kickers_input = self.kickersmodel.ch2input(net_input,tmphand)
            kickers_onehot = self.kickersmodel.allkickers2onehot(allkickers)
            num = choice[0]['chain']
            if choice[0]["type"] == "Four":
                num *= 2
            kickers = []
            for _ in range(num):
                actidx,val = self.kickersmodel.get_action(kickers_input,kickers_onehot,self.norand)
                kickers.append(self.kickersmodel.idx2CardPs(actidx))
                kidx = self.kickersmodel.cardPs2idx(kickers[-1])
                kickers_onehot[kidx] = 0
            for k in kickers:
                tmphand.extend(k)
            #print(self.kickersmodel.episodeTemp)
            choice = tmphand
        cardChoice = CardInterpreter.selectCardByHand(self.simulator.myCards, choice)

        return self.makeData(cardChoice,val)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.InteractiveSession(config=config)
    playmodel = [PlayModel("play"+str(i),sess,i) for i in range(3)]
    kickersmodel = KickersModel("kick",sess)
    tf.global_variables_initializer().run()
    kickersmodel.load_model("data/FTL/","DQN")
    player = FTLBot(playmodel[0], kickersmodel, input(), "JSON", True, True)
    id = player.simulator.nowPlayer
    #print(id)
    player.playmodel = playmodel[id]
    res = player.makeDecision()
    print(res)