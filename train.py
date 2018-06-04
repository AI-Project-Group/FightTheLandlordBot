import tensorflow as tf
from ftl_judgement import FTLJudgement
from DQNModel import PlayModel, KickersModel

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.InteractiveSession(config=config)
    playmodel = [PlayModel("play"+str(i),sess,i) for i in range(3)]
    #playmodel = [PlayModel("play"+str(i),sess,i,"data/FTL/test.ckpt") for i in range(3)]
    kickersmodel = KickersModel("kick",sess)
    tf.global_variables_initializer().run()
    kickersmodel.load_model("data/FTL/","DQN")
    episode = 1
    while True:
        learn_step = playmodel[0].learn_step_counter
        if learn_step != 0 and learn_step % (2*playmodel[0].replace_target_iter) == 0:
            print("\nSave Model!\n")
            kickersmodel.save_model("data/FTL/","DQN")
        print("Train Episode: %d"%(episode))
        ftlJudge = FTLJudgement([], False)
        ftlJudge.work(playmodel,kickersmodel,episode)
        episode += 1