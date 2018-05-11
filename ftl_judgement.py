#!/usr/bin/python
# -*- coding: UTF-8 -*-

import simulator
import ftl_bot
import random

# Judge class for Fight The Landlord game
class FTLJudgement:

    # lastCardTable: the card table status of the previous game, influences shuffling
    def __init__(self, lastCardTable = [], isDebug = False):

        self.nowTurn = -1 # Not Played
        self.cardTable = [] # what cards are played now
        #self.isPlayed = [0] * 54 # is card id == k played

        self.playerHistory = [[], [], []] # The play history of three players
        self.last2Plays = [[], []] # Record the most recent previous 2 plays

        self.isDebug = isDebug
        self.cards = lastCardTable
        if not lastCardTable: # no previous information
            self.cards = list(range(0, 54))
            random.shuffle(self.cards)

        self.publicCards = self.cards[0 : 3] # included in Landlord cards
        self.cardsPlayer = [self.cards[0 : 20], self.cards[20 : 37], self.cards[37 : 54]] # Landlord, Farmer 1, Farmer 2
        self.publicCards.sort()
        self.cardsPlayer[0].sort()
        self.cardsPlayer[1].sort()
        self.cardsPlayer[2].sort()
        self.nowCardsPlayer = [self.cardsPlayer[i][:] for i in range(3)] # deep copy

        self.log("Public cards are "+simulator.CardInterpreter.getCardName(self.publicCards))
        for player in range(3):
            self.log("Serve Player %d : "%(player)+simulator.CardInterpreter.getCardName(self.nowCardsPlayer[player]))



    # debug_print
    def log(self, text):
        if self.isDebug:
            print("Turn %d: %s"%(self.nowTurn, text))

    # simulate the game process
    def work(self):
        isGameFinished = False
        while not isGameFinished: # Looping
            self.nowTurn += 1;
            for playerID in range(3): # 3 players
                data = {"ID": playerID, "nowTurn": self.nowTurn, "publicCard": self.publicCards}
                data["history"] = self.playerHistory
                data["deal"] = self.cardsPlayer[playerID]
                player = ftl_bot.FTLBot(data, "Judge")
                cardsPlayed = player.makeDecision()

                self.log("Player %d [%d] card %s"%(playerID, \
                    len(self.nowCardsPlayer[playerID]), \
                    simulator.CardInterpreter.getCardName(cardsPlayed)))
                    
                # checks applied
                for c in cardsPlayed:
                    if not c in self.nowCardsPlayer[playerID]: # player played a card it not owned
                        self.log("Card %s is not owned"%simulator.CardInterpreter.getCardName(c))
                        isGameFinished = True
                        break
                if isGameFinished: break

                hand = simulator.Hand(cardsPlayed)
                if hand.type == "None": # Not avaliable pattern
                    self.log("The pattern is not recognized")
                    isGameFinished = True
                    break

                if hand.chain >= 2 and hand.primal == 12: # 2 in a chain
                    self.log("Card 2 in a chain")
                    isGameFinished = True
                    break

                lastCards = self.last2Plays[1] or self.last2Plays[0]
                lastHand = simulator.Hand(lastCards)
                if not hand.isAbleToFollow(lastHand): # can't cover the last hand
                    self.log("Cards can't follow the last hand %s"%simulator.CardInterpreter.getCardName(lastCards))
                    isGameFinished = True
                    break

                # LEGAL
                self.last2Plays[0] = self.last2Plays[1]
                self.last2Plays[1] = cardsPlayed
                for c in cardsPlayed:
                    self.nowCardsPlayer[playerID].remove(c)
                self.playerHistory[playerID].append(cardsPlayed)
                self.cardTable.extend(cardsPlayed)

                if not len(self.nowCardsPlayer[playerID]): # Finished
                    result = self.report(playerID)
                    isGameFinished = True
                    break

            input("Press <ENTER> to continue...")

        self.log("Game finished")
        for playerID in range(3): # discard cards to table
            self.cardTable.extend(self.nowCardsPlayer[playerID])

        return self.cardTable


    # Report the result
    def report(self, winner):
        self.log("The winner is bot %d"%winner)
        # @TODO Add Score calculation
        # @TODO Add Model Training

ftlJudge = FTLJudgement([], True)
ftlJudge.work()
