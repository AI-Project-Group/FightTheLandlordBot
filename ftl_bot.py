#!/usr/bin/python
# -*- coding: UTF-8 -*-

import simulator
import json
import random, copy
from Network import PlayModel
#from PolicyNetwork import PlayModel

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
        possiblePlays = simulator.CardInterpreter.splitCard(cards, [])
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
            cardsc = simulator.CardInterpreter.selectCardByHand(nextcards, tmpc)
            for c in cardsc:
                nextcards.remove(c)
            hand = simulator.Hand(cardsc)
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
        lastHand = simulator.Hand(sim.cardsToFollow)
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
                    nowHand = simulator.Hand([],p)
                    if nowHand.isAbleToFollow(lastHand):
                        ablelist.append(p)
                possiblePlays = ablelist
                if possiblePlays:
                    possiblePlays.append([])
                    possiblePlays.extend(pbombs)
            #print("Search Human!!!")
            #print(possiblePlays)
        if possiblePlays == []:
            possiblePlays = simulator.CardInterpreter.splitCard(self.simulator.myCards, lastHand)
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
        #print(net_input.shape)
        #print(net_input)
        actidx,val = self.playmodel.get_action(net_input, one_hot_t, self.norand, addNonZero)
        choice = self.playmodel.idx2CardPs(actidx)
        #print(choice)

        # Add kickers, if first element is dict, the choice must has some kickers
        # get kickers from kickers model
        if choice and isinstance(choice[0],dict):
            tmphand = choice[1:]
            allkickers = simulator.CardInterpreter.getKickers(sim.myCards, choice[0]["kickerNum"], list(set(tmphand)))
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
                self.kickersmodel.storeSamples(kickers_input,sim.nowPlayer,k,sim.nowTurn)
            #print(self.kickersmodel.episodeTemp)
            choice = tmphand
        cardChoice = simulator.CardInterpreter.selectCardByHand(self.simulator.myCards, choice)
        
        self.playmodel.storeSamples(net_input,cardChoice,one_hot_t)

        return self.makeData(cardChoice)

if __name__ == "__main__":
    #print(FTLBot.maxValueKickers([[1],[13]],[[2,2]],[1,2],[]))
    print(FTLBot.searchHuman([16,
				15,
				0,
				14,
				19,
				20,
				22,
				49,
				28,
				48,
				17,
				18,
				11,
				9,
				35,
				50,
				41,
				26,
				1,
				12],[],[],0))