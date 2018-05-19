#!/usr/bin/python
# -*- coding: UTF-8 -*-

import simulator
import json
import random
from Network import PlayModel
#from PolicyNetwork import PlayModel

# Fight The Landlord executor

# Initialization using the JSON input
class FTLBot:
    def __init__(self, playmodel,kickersmodel, data, dataType = "Judge"):
        self.dataType = dataType
        self.playmodel = playmodel
        #self.valuemodel = valuemodel
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
            sim = simulator.FTLSimulator(myBotID, nowTurn, publicCard)
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
            sim = simulator.FTLSimulator(data["ID"], data["nowTurn"], data["publicCard"])
            sim.deal(data["deal"])
            sim.setHistory(data["history"])
            self.simulator = sim

    def makeData(self, data):
        if self.dataType == "JSON":
            return json.dumps({"response": data})
        elif self.dataType == "Judge":
            return data

    # Return the decision based on type
    def makeDecision(self):
        lastHand = simulator.Hand(self.simulator.cardsToFollow)   
        possiblePlays = simulator.CardInterpreter.splitCard(self.simulator.myCards, lastHand)
        #print(possiblePlays)

        # @TODO You need to modify the following part !!
        # A little messed up ...
        if not len(possiblePlays):
            return self.makeData([])
        
        sim = self.simulator
        net_input = PlayModel.ch2input(sim.nowPlayer,sim.myCards,sim.publicCard,sim.history,sim.lastPlay,sim.lastLastPlay)
        one_hot_t = self.playmodel.hand2one_hot(possiblePlays)
        choice = self.playmodel.getAction(net_input, one_hot_t, 0.5)
        #print(choice)

        # Add kickers, if first element is dict, the choice must has some kickers
        # @TODO get kickers from kickers model
        if choice and isinstance(choice[0],dict):
            tmphand = choice[1:]
            kickers_input = self.kickersmodel.ch2input(net_input,tmphand)
            allkickers = simulator.CardInterpreter.getKickers(sim.myCards, choice[0]["kickerNum"], list(set(tmphand)))
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
        cardChoice = simulator.CardInterpreter.selectCardByHand(self.simulator.myCards, choice)
        
        #self.playmodel.storeSamples(net_input,cardChoice, len(possiblePlays) == 1 and choice == [])
        self.playmodel.storeSamples(net_input,cardChoice,one_hot_t)

        # You need to modify the previous part !!
        return self.makeData(cardChoice)
