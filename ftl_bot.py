#!/usr/bin/python
# -*- coding: UTF-8 -*-

import simulator
import json
import random
#from PolicyNetwork import PlayModel

# Fight The Landlord executor

# Initialization using the JSON input
class FTLBot:
    def __init__(self, playmodel, data, dataType = "Judge"):
        self.dataType = dataType
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
        print(possiblePlays)
        kickerNum = lastHand.kickerNum

        # @TODO You need to modify the following part !!
        # A little messed up ...
        if not len(possiblePlays):
            return self.makeData([])

        choice = random.choice(possiblePlays) # choose a random strategy
        cardChoice = simulator.CardInterpreter.selectCardByHand(self.simulator.myCards, choice)
        handChoice = simulator.Hand(cardChoice)
        for c in cardChoice: # Remove for kickers
            self.simulator.myCards.remove(c)
        #print(cardChoice)
        #print(self.simulator.myCards)

        # Add kickers
        if handChoice.type == "Trio": # Able to add kickers, I won't play shuttle
            if lastHand.type == "Pass":
                kickerNum = random.randint(0,2)
            possibleKickers = simulator.CardInterpreter.getKickers(self.simulator.myCards, kickerNum, choice[0])
            print(possibleKickers)
            if len(possibleKickers) >= lastHand.chain: # choose the smallest kickers
                for i in range(lastHand.chain):
                    kickersChoice = simulator.CardInterpreter.selectCardByHand(self.simulator.myCards, possibleKickers[i])
                    cardChoice.extend(kickersChoice)

        # You need to modify the previous part !!
        return self.makeData(cardChoice)
