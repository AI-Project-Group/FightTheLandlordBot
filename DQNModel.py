import random
import numpy as np
import tensorflow as tf
from simulator import Hand
from collections import deque

'''
original_vesion https://github.com/thuxugang/doudizhu & https://morvanzhou.github.io/tutorials/
modified by Firmlyzhu
'''

class DuelingDQN:
    def __init__(
            self,
            n_features,
            n_actions,
            modelname,
            sess,
            learning_rate=1e-4,
            reward_decay=0.95,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=3200,
            batch_size=128,
            e_greedy_increment=None,
            dueling=True,
            output_graph=False,
            BaseScore=200,
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
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2 + n_actions))

        # consist of [target_net, evaluate_net]
        with tf.variable_scope(self.modelname):
            self._build_net()
        
        self.sess = sess

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
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
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
              
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

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            self.q_eval = build_layers(self.s, c_names, w_initializer, b_initializer)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            q_next = build_layers(self.s_, c_names, w_initializer, b_initializer)
            self.q_next = q_next + self.BaseScore

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * (tf.reduce_max(self.q_next*self.action_possible, axis=1, name='Qmax_s_') - self.BaseScore)    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_one_hot = tf.one_hot(self.a, depth=self.n_actions, dtype=tf.float32)
            self.q_eval_wrt_a = tf.reduce_sum(self.q_eval * a_one_hot, axis=1)     # shape=(None, )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, actions_one_hot, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, actions_one_hot, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def get_action(self, netinput, actions_one_hot, norand=False):
        output = self.q_eval.eval(feed_dict={self.x:[netinput]})
        output = output.flatten() + self.BaseScore
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
        outidx = -1
        if norand or randf < self.epsilon:
            outidx = np.argmax(legalOut)
        else:
            outidx = random.choice(allidx)
        return outidx, legalOut[outidx]

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
    
    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.action_possible: batch_memory[:, self.n_features:self.n_features+self.n_actions],
                self.a: batch_memory[:, self.n_features+self.n_actions],
                self.r: batch_memory[:, self.n_features +self.n_actions+1],
                self.s_: batch_memory[:, self.n_features +self.n_actions+2:],
            })
    
        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        
        return cost
    
    #新增
    def save_model(self,path,ckptname):
        saver = tf.train.Saver() 
        saver.save(self.sess, path+ckptname+".ckpt") 

    #新增
    def load_model(self,path,ckptname):
        saver = tf.train.Saver() 
        saver.restore(self.sess, path+ckptname+".ckpt") 
        
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

class PlayModel(DuelingDQN):

    def __init__(self,modelname,sess,player):
        self.player = player
        super(PlayModel, self).__init__(105+364,364,modelname,sess)

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
    
    
        

if __name__ == '__main__':
    #accelerate MLP
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    DQN = DuelingDQN(3,4,"test",sess)
    vals = DQN.q_eval.eval(session=sess, feed_dict={DQN.s:[[0,0,0]]})
    print(vals)
    vals = DQN.q_next.eval(session=sess,feed_dict={DQN.s_:[[1,0,0]],DQN.r:[0]})
    print(vals)
    print(DQN.q_target.eval(session=sess,feed_dict={DQN.s_:[[1,0,0]],DQN.r:[10],DQN.action_possible:[[0,1,0,0]]}))