# FightTheLandlordBot
A Robot of Fight the Landlord

* ftl_bot.py: The Bot class
* simulator.py: simulator and some tools
* ftl_judgement.py: Judgement class
* DQNModel.py: playmodel and kickers model used by Bot
* Network.py:  playmodel and kickers model used by Bot (deprecated)
* botzone.py: bot for botzone
* train.py: to train the bot
* test.py: to test the bot

---

Environment:

* python 3.5+
* tensorflow 1.7+

---
To start a bot training:

> python ./train.py

It will load stored data in `./data/FTL/` and then start training. If there is no stored data, it will just init the network and then start training.

The parameters of network will be stored every 1000 training steps. And the data will be stored in `./data/FTL/`.

To start testing on bot:

> python ./test.py

It will create two kinds of bot. One kind will load data in `./data/best/` and the other will load data in `./data/FTL/`. Then, the script will test them according to 1000 Fight The Land Lord games under the botzone rules. 
