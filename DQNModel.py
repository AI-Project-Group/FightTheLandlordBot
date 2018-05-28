import random
import numpy as np
import tensorflow as tf
from simulator import Hand

'''
original_vesion https://github.com/thuxugang/doudizhu & https://morvanzhou.github.io/tutorials/
modified by Firmlyzhu
'''

class SumTree(object):
    """
    This SumTree code is from:
    https://github.com/thuxugang/doudizhu/blob/rl_pdqn/rl/prioritized_dqn_max_ori.py
    &
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    
    Story the data with it priority in tree and data frameworks.
    """    

    def __init__(self, capacity):
        self.capacity = capacity    # for all priority values
        self.tree = np.zeros(2*capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)    # for all transitions
        self.data_pointer = 0
        # [--------------data frame-------------]
        #             size: capacity

    def add_new_priority(self, p, data):
        leaf_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data # update data_frame
        self.update(leaf_idx, p)    # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):    # end search when no more child
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound-self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]     # the root


class Memory(object):   # stored as ( s, a, r, s_ ) in SumTree

    epsilon = 1e-4  # small amount to avoid zero priority
    alpha = 0.6     # [0~1] convert the importance of TD error to priority
    beta = 0.4      # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 1e-4
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add_new_priority(max_p, transition)   # set the max p for new p

    def sample(self, n):
        batch_idx, batch_memory, ISWeights = [], [], []
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        #print(self.tree.root_priority)
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.root_priority
            ISWeights.append(self.tree.capacity * prob)
            batch_idx.append(idx)
            batch_memory.append(data)
        
        #print(ISWeights)
        ISWeights = np.hstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return batch_idx, np.vstack(batch_memory), ISWeights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        error += self.epsilon  # avoid 0
        clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)

class DuelingDQN:
    def __init__(
            self,
            n_features,
            n_actions,
            modelname,
            sess,
            learning_rate=1e-4,
            reward_decay=0.98,
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
        self.memory = Memory(capacity=memory_size)

        # consist of [target_net, evaluate_net]
        with tf.variable_scope(self.modelname):
            self._build_net()
        
        self.sess = sess

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []
        
        self.t_params = tf.get_collection(self.modelname+'/target_net_params')
        self.e_params = tf.get_collection(self.modelname+'/eval_net_params')
        self.params_in = []
        self.assign_ops = []
        for param in self.t_params:
            tmpname = param.name.replace("/","_").replace(":","_")
            self.params_in.append(tf.placeholder(tf.float32, param.shape, name=tmpname+"_in"))
            self.assign_ops.append(tf.assign(param, self.params_in[-1]))
        self.saver = tf.train.Saver()
    
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
        self.ISWeights = tf.placeholder(tf.float32, [None,], name='IS_weights')

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, w_initializer, b_initializer = \
                [self.modelname+'/eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)  # config of layers
            self.q_eval = build_layers(self.s, c_names, w_initializer, b_initializer)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = [self.modelname+'/target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
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
    
    def get_action(self, netinput, actions_one_hot, norand=False):
        output = self.q_eval.eval(feed_dict={self.s:[netinput]})
        output = output.flatten() + self.BaseScore
        #print(output)
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
        e_params_vals = self.sess.run(self.e_params)
        #print(e_params_vals)
        for i in range(len(e_params_vals)):
            self.sess.run(self.assign_ops[i],feed_dict={self.params_in[i]:e_params_vals[i]})
    
    def learn(self,iskickers=False):
        #print(self.sess.run(self.t_params[0]))
        #print(self.sess.run(self.e_params[0]))
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
        '''print(ins)
        print(ina)
        print(inr)
        print(ins_)
        print(inaction_possible)'''
        '''qt,qe,abs_errors = self.sess.run([self.q_target,self.q_eval_wrt_a,self.abs_errors],
                                          feed_dict={self.s: ins,
                                                     self.a: ina,
                                                     self.r: inr,
                                                     self.s_: ins_,
                                                     self.action_possible: inaction_possible})
        #print(qt)
        #print(qe)
        print(abs_errors)'''
        
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
        self.saver.save(self.sess, path+ckptname+".ckpt", write_meta_graph=False)

    #新增
    def load_model(self,path,ckptname):
        try:
            self.saver.restore(self.sess, path+ckptname+".ckpt")
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
        super(PlayModel, self).__init__(105+364,364,modelname,sess,memory_size=640,batch_size=32,e_greedy_increment=0.9/2e5)
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

    def storeSamples(self,netinput,action,allonehot):
        actidx = PlayModel.cardPs2idx(Hand.getCardPoint(action))
        hand = Hand(action)
        self.episodeTemp.append([netinput,actidx,0,allonehot])

    def finishEpisode(self,score,istrain=True):      
        #print("Player %d add to the train batch"%(self.player))
        nlen = len(self.episodeTemp)
        #print(nlen)
        for i in range(nlen,0,-1):
            data = self.episodeTemp[i-1]
            if i == nlen:
                self.store_transition(data[0],data[1],data[2]+score+self.BaseScore*self.gamma,np.zeros(self.n_features),np.zeros(self.n_actions))
            else:
                ndata = self.episodeTemp[i]
                self.store_transition(data[0],data[1],data[2],ndata[0],ndata[3])
        
        #print(self.memory.tree.tree)
        minp = np.min(self.memory.tree.tree)
        #print(minp)
        if istrain and minp > 0:
            print("Train for PlayModel Player: %d" % self.player)
            loss = self.learn()
            print("learn_step_counter:"+str(self.learn_step_counter)+" loss:"+str(loss)+" epsilon:"+str(self.epsilon)+" root_p:"+str(self.memory.tree.root_priority))
        
        self.episodeTemp = []
        
class KickersModel(DuelingDQN):
    
    def __init__(self,modelname,sess):
        super(KickersModel, self).__init__(105+364+15,28,modelname,sess,memory_size=320,batch_size=16,e_greedy_increment=0.9/1e5)
        self.episodeTemp = []
        tf.Graph().finalize()
    
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
            
    def storeSamples(self, netinput, playerID, kickers, turn):
        actidx = self.cardPs2idx(kickers)
        self.episodeTemp.append([netinput, playerID, actidx, turn])
        
    def finishEpisode(self,playmodels,scores):
        for tmp in self.episodeTemp:
            p = tmp[1]
            t = tmp[3]
            datatemp = playmodels[p].episodeTemp
            if t >= len(datatemp)-1:
                self.store_transition(tmp[0], tmp[2], 0, np.zeros(playmodels[p].n_features), np.zeros(playmodels[p].n_actions), p, scores[p]+self.BaseScore*self.gamma)
            else:
                self.store_transition(tmp[0], tmp[2], 0, datatemp[t+1][0], datatemp[t+1][3], p, 0)
        
        #print(self.episodeTemp)
        #print(self.memory.tree.tree)
        minp = np.min(self.memory.tree.tree)
        #print(minp)
        if len(self.episodeTemp) != 0 and minp > 0:
            print("Train for KickersModel...")
            
            # update reward
            data = self.memory.tree.data
            targetd = [[],[],[]]
            targetidx = [[],[],[]]
            #print(data)
            for i,d in enumerate(data):
                player = int(d[self.n_features+2])
                targetidx[player].append(int(i))
                targetd[player].append(d[self.n_features+3:])
            
            for p in range(3):
                if len(targetidx[p]) == 0:
                    continue
                model = playmodels[p]
                tdata = np.vstack(targetd[p])
                #print(tdata[:,0])
                tvals = self.sess.run(model.q_target, feed_dict={model.s_: tdata[:, 1:model.n_features+1], 
                                                                 model.r: tdata[:,0],
                                                                 model.action_possible: tdata[:,model.n_features+1:]})
                #print(tvals)
                for i,idx in enumerate(targetidx[p]):
                    data[idx][self.n_features+1] = tvals[i]
                    #print(data[idx].tolist())
            
            # learn
            loss = self.learn(True)
            print("learn_step_counter:"+str(self.learn_step_counter)+" loss:"+str(loss)+" epsilon:"+str(self.epsilon)+" root_p:"+str(self.memory.tree.root_priority))
        
        self.episodeTemp = []

    

if __name__ == '__main__':
    #accelerate MLP
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    play = PlayModel("play",sess,0)
    kick = KickersModel("kickers",sess)
    sess.run(tf.global_variables_initializer())
    kick.save_model("data/test/","test")
    '''vals = DQN.q_eval.eval(session=sess, feed_dict={DQN.s:[[0,0,0]]})
    print(vals)
    vals = DQN.q_next.eval(session=sess,feed_dict={DQN.s_:[[1,0,0]],DQN.r:[0]})
    print(vals)
    print(DQN.q_target.eval(session=sess,feed_dict={DQN.s_:[[1,0,0]],DQN.r:[10],DQN.action_possible:[[0,1,0,0]]}))'''