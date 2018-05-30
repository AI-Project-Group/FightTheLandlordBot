import tensorflow as tf
import numpy as np
from ftl_judgement import FTLJudgement
from DQNModel import PlayModel, KickersModel

MaxEpisode = 2000

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.InteractiveSession(config=config)
    playmodel = [PlayModel("play"+str(i),sess,i) for i in range(3)]
    #playmodel = [PlayModel("play"+str(i),sess,i,"data/FTL/test.ckpt") for i in range(3)]
    kickersmodel = KickersModel("kick",sess)
    tf.global_variables_initializer().run()
    kickersmodel.load_model("data/FTL/","DQN")
    wins = [0,0,0]
    sum_scores = [[],[],[]]
    for ep in range(MaxEpisode):
        print("Test Episode: %d"%(ep))
        ftlJudge = FTLJudgement([], False)
        winner,scores,_ = ftlJudge.work(playmodel,kickersmodel,ep,"Test")
        wins[winner] += 1
        for i in range(3):
            sum_scores[i].append(scores[i])
        ep += 1
    print("Each Player wins:"+str(wins))
    aves = []
    for i in range(3):
        aves.append(np.average(sum_scores[i]))
    print("Average Scores:"+str(aves))