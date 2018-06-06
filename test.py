import tensorflow as tf
import numpy as np
import random
from ftl_judgement import FTLJudgement
from DQNModel import PlayModel, KickersModel

MaxEpisode = 1000

if __name__ == "__main__":
    lordg = tf.Graph()
    farmg = tf.Graph()
    playmodel = [[],[]]
    with lordg.as_default():
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess1 = tf.Session(config=config,graph=lordg)
        lordmodel = PlayModel("play0",sess1,0)
        playmodel[0].append(lordmodel)
        playmodel[1].append(lordmodel)
        playmodel[1].append(PlayModel("play1",sess1,1))
        playmodel[1].append(PlayModel("play2",sess1,2))
        kickersmodel = KickersModel("kick",sess1)
        
    with farmg.as_default():
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess2 = tf.Session(config=config,graph=farmg)
        playmodel[1][0] = PlayModel("play0",sess2,0)
        playmodel[0].append(PlayModel("play1",sess2,1))
        playmodel[0].append(PlayModel("play2",sess2,2))
    kickersmodel.load_model("data/best/","DQN")
    playmodel[0][2].load_model("data/FTL/","DQN")
    addHuman= [[False,True,True],[True,False,False]]
    
    twins = 0
    sum_scores = [[],[]]
    for ep in range(MaxEpisode):
        print("Test Episode: %d"%(ep))
        cards = list(range(0, 54))
        random.shuffle(cards)
        for t in range(2):
            ftlJudge = FTLJudgement(cards, False)
            winner,scores,_ = ftlJudge.work(playmodel[t],kickersmodel,ep,"Test",addHuman[t])
            if t == 0:
                if winner == 0:
                    twins += 1
                sum_scores[0].append(scores[0])
                sum_scores[1].append(scores[1])
            else:
                if winner != 0:
                    twins += 1
                sum_scores[0].append(scores[1])
                sum_scores[1].append(scores[0])
        ep += 1
    print("Target Player wins:"+str(twins))
    aves = []
    for i in range(2):
        aves.append(np.average(sum_scores[i]))
    print("Average Scores:"+str(aves))