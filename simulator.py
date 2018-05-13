#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Fight The Landlord game simulator
# Botzone version according to https://wiki.botzone.org.cn/index.php?title=FightTheLandlord
class FTLSimulator:

    # Simulation Functions
    # Initialization
    def __init__(self, nowPlayer, nowTurn, publicCard):
        # Format Descriptions
        self.nowPlayer = 0 # player 0: Landlord, 1: Farmer A(Jia), 2: Farmer B(Yi)
        self.cardCnt = [0, 0, 0] # the number of card every player now holds
        self.myCards = [] # What cards do I have NOW?
        self.publicCard = [] # 3 public cards owned by the landlord when dealing
        self.nowTurn = -1 # the present gaming turn (3 plays (including pass) each turn), -1 for not played yet
        self.history = [[], [], []] # history[p][k]: the card played by player p in the k-th turn

        # Operation Accessories
        self.isCardPlayed = [0] * 54 # isCardPlayed[i]: is card serial i played by any player
        self.cardsToFollow = [] # which hand shall I follow now ? empty for my turn to choose hand type

        self.nowPlayer = nowPlayer
        self.cardCnt = [20, 17, 17]
        self.nowTurn = nowTurn
        self.publicCard = publicCard
        self.lastPlay = []
        self.lastLastPlay = []
        # If I'm the landlord, public card is already included in my dealing

    # Serve cards
    def deal(self, cards):
        self.myCards.extend(cards)

    # Play cards, record history by sequential order
    def play(self, player, cards):
        self.cardCnt[player] -= len(cards)
        for card in cards:
            self.isCardPlayed[card] = 0
        # It's me playing
        if player == self.nowPlayer:
            for card in cards: # play my cards
                self.myCards.remove(card)
        # Record this play
        self.history[player].append(cards)
        self.lastLastPlay = self.lastPlay
        self.lastPlay = cards

    def setCardsToFollow(self, cards):
        self.cardsToFollow = cards

    # set history directly & retrieve current status
    # no need to set cards to follow any more
    def setHistory(self, historyData):
        lenh = [len(h) for h in historyData]
        for t in range(lenh[0]):
            for player in range(3):
                if t >= lenh[player]:
                    continue
                self.play(player, historyData[player][t])
        self.cardsToFollow = self.lastPlay or self.lastLastPlay
        # after this, self.history will be identical to historyData

class Hand:
    # point:       0 1 2 3 4 5 6 7  8 9 10 11 12 13 14
    # actual card: 3 4 5 6 7 8 9 10 J Q K  A  2  Jb Jr
    @staticmethod
    def getCardPoint(card):
        if isinstance(card, list): # a card list
            return [Hand.getCardPoint(c) for c in card]
        # a single card
        if card == 52:
            return 13
        elif card == 53:
            return 14
        else:
            return card//4

    # distinguish cards as hand patterns
    def __init__(self, cards):
        if isinstance(cards, Hand): # Copy instance
            self.type = cards.type
            self.primal = cards.primal
            self.kickerNum = cards.kickerNum
            self.chain = cards.chain
            return

        # Assume that cards has been sorted:
        cards.sort()
        # And, is a LEGAL type
        #self.cards = cards
        self.type = "None"
        self.primal = -1 # the basic card type, used for judging greater hand
        self.kickerNum = 0 # cards added to the basic type, 0: None, 1: Solo, 2: Pair
        # kicker element format: (point, type), type = 1 or 2
        self.chain = 1 # consecutive count. e.g. trio + solo && chain == 3 : Airplane with wings

        point = Hand.getCardPoint(cards) # the point representing card points

        if len(cards) == 0: # passed
            self.type = "Pass"
        elif len(cards) == 1: # Solo
            self.type = "Solo"
            self.primal = point[0]
        elif len(cards) == 2: # Pair or Rocket
            if point[0] == 13: # Rocket
                self.type = "Rocket"
            elif point[0] == point[1]: # Pair
                self.type = "Pair"
                self.primal = point[0]
        else: # Above, types can be regarded as kickers
            pointU = list(set(point)) # delete duplicate
            pointUCnt = [point.count(p) for p in pointU] # count the number of each point
            pattern = list(set(pointUCnt)) # get the pattern of point
            pattern.sort()
            # distinguish the pattern
            if pattern == [1]: # Solo chain
                self.type = "Solo"
                self.primal = pointU[0]
                self.chain = len(pointU)
            elif pattern == [2]: # Pair chain
                self.type = "Pair"
                self.primal = pointU[0]
                self.chain = len(pointU)
            elif pattern == [3]: # Trio, including airplane
                self.type = "Trio"
                self.primal = min(pointU)
                self.chain = len(pointU)
            elif pattern == [1, 3] or pattern == [2, 3]: # Trio + Solo/Pair, including airplane
                self.type = "Trio"
                self.primal = min([c for i, c in enumerate(pointU) if pointUCnt[i] == 3])
                self.chain = len(pointU) // 2
                self.kickerNum = pattern[0]
            elif pattern == [4]: # Bomb or Four chain (Shuttle)
                if len(pointU) == 1: # Only 1 point: Bomb
                    self.type = "Bomb"
                    self.primal = pointU[0]
                else: # Four chain
                    self.type = "Four"
                    self.primal = min(pointU)
                    self.chain = len(pointU)
            elif pattern == [1, 4] or (pattern == [2, 4] and len(pointU) % 3 == 0): # Four + Dual Solo/Pair, including shuttle
                # originally, error when cards = [0,1,2,3,4,5]
                self.type = "Four"
                self.primal = min([c for i, c in enumerate(pointU) if pointUCnt[i] == 4])
                self.chain = len(pointU) // 3
                self.kickerNum = pattern[0]

    # get report string
    def report(self):
        return "%s From %d Len = %d" % (self.type, self.primal, self.chain)

    # compare two hands, is this pattern able to follow the other one : T / F
    def isAbleToFollow(self, other):
        if self.type == "Pass" and other.type == "Pass": # Must follow sth.
            return False
        if self.type == "Pass" or other.type == "Pass": # always can follow
            return True
        if other.type == "Rocket": # Nothing can follow
            return False
        if self.type == "Rocket": # surpasses every hand
            return True
        if self.type == "Bomb": # dispose bomb separately
            if other.type == "Bomb":
                return self.primal > other.primal
            return True # surpasses other hands
        if self.type != other.type: # mixed patterns
            return False
        return self.chain == other.chain and self.primal > other.primal
        
    def getHandScore(self):
        score = 0
        if self.type == "Pass":
            score = 0
        elif self.type == "Solo" and self.chain == 1:
            score = 1
        elif self.type == "Pair" and self.chain == 1:
            score = 2
        elif self.type == "Trio" and self.chain == 1:
            score = 4
        elif self.type == "Solo" and self.chain >= 5:
            score = 6
        elif self.type == "Pair" and self.chain >= 3:
            score = 6
        elif self.type == "Trio" and self.chain >= 2:
            score = 8
        elif self.type == "Four" and self.chain == 1:
            score = 8
        elif self.type == "Bomb":
            score = 10
        elif self.type == "Four" and self.chain == 2:
            score = 10
        elif self.type == "Rocket":
            score = 16
        elif self.type == "Four" and self.chain > 2:
            score = 20
        return score #/ 100.0


class CardInterpreter:
    # get the written name of the card
    @staticmethod
    def getCardName(cid):
        if isinstance(cid, list): # a card list
            return str([CardInterpreter.getCardName(c) for c in cid])
        if cid == 52:
            return "Black Joker"
        elif cid == 53:
            return "Red Joker"
        else:
            return ("Heart","Diamond","Spade","Club")[cid%4] + " " + \
                ("10" if cid//4 == 7 else "3456789_JQKA2"[cid//4])

    # parse the written name into id, for debug only
    @staticmethod
    def getCardID(cardName):
        cardName.lower()
        cardName.replace(" ","")
        if cardName == "jb" or cardName == "blackjoker":
            return 52
        if cardName == "jr" or cardName == "redjoker":
            return 53
        lastNum = {"3":0,"4":1,"5":2,"6":3,"7":4,"8":5,"9":6,"0":7,"j":8,"q":9,"k":10,"a":11,"2":12}[cardName[-1]]
        colorNum = {"h":0,"d":1,"s":2,"c":3}[cardName[0]]
        return lastNum * 4 + colorNum

    # get all possible hands in cards, meanwhile get all solos and pairs
    # cards: [card_s], cardsToFollow: the card set to follow
    # assume that cards is already sorted
    # return the point representation, may NOT sorted
    # when the hand has kickers, like "Trio" or "Four", the first element will be kickers
    @staticmethod
    def splitCard(cards, cardsToFollow = []):
        allHands = [[]] # all legal cards in points, originally 'pass' is a legal actions
        # *** colors, kickers, difference between bomb and four are not considered
        lastHand = Hand(cardsToFollow)
        # solos records all solo kickers
        # pairs records all pair kickers
        if lastHand.type == "Rocket": # can't follow
            return allHands

        # deal with Rocket
        if 52 in cards and 53 in cards: # Rocket
            allHands.append([13,14])

        # No Joker from now on, deal with bombs
        point = Hand.getCardPoint(cards) # get the point expressions
        pointU = list(set(point)) # delete duplicate
        pointUCnt = [point.count(p) for p in pointU] # count the number of each point

        # including card 2 (point == 12)
        soloRec = pointU[:] # including joker cards
        pairRec = [p for i, p in enumerate(pointU) if pointUCnt[i] >= 2] # p >= 13: no pair/trio/four
        trioRec = [p for i, p in enumerate(pointU) if pointUCnt[i] >= 3]
        fourRec = [p for i, p in enumerate(pointU) if pointUCnt[i] >= 4] # Bombs or Four + Dual Solo/Pair

        if lastHand.type == "Bomb": # append specific bombs
            allHands.extend([[pr]*4 for pr in fourRec if pr > lastHand.primal])
            return allHands

        allHands.extend([[pr]*4 for pr in fourRec]) # append bombs

        if lastHand.type == "Four": # kickers appended
            # if lastHand is a shuttle, then here all bombs are listed
            for i,c in enumerate(fourRec):
                if lastHand.chain == 1 and c > lastHand.primal:
                    kickers = CardInterpreter.getKickers(cards, lastHand.kickerNum, [c])
                    if len(kickers) >= 2:
                        allHands.append([kickers,c,c,c,c])
            return allHands

        # relative to xxxRec lists
        soloChainCnt = [1]
        pairChainCnt = [1] if pairRec else []
        trioChainCnt = [1] if trioRec else [] # airplane with no wings
        # do not consider four chains (shuttle)

        # card 2 (point 12), joker cards only in chain 1
        for i, p in enumerate(soloRec):
            if i: # start from the second place
                soloChainCnt.append(soloChainCnt[i-1] + 1 if p == soloRec[i-1] + 1 and p < 12 else 1)
        for i, p in enumerate(pairRec):
            if i: # start from the second place
                pairChainCnt.append(pairChainCnt[i-1] + 1 if p == pairRec[i-1] + 1 and p < 12 else 1)
        for i, p in enumerate(trioRec):
            if i: # start from the second place
                trioChainCnt.append(trioChainCnt[i-1] + 1 if p == trioRec[i-1] + 1 and p < 12 else 1)

        # Record Chain
        if lastHand.type == "Solo":
            for i, c in enumerate(soloRec):
                if soloChainCnt[i] >= lastHand.chain and c-soloChainCnt[i]+1 > lastHand.primal: # able to follow
                    allHands.append(list(range(c-lastHand.chain+1, c+1)))
            return allHands
        if lastHand.type == "Pair":
            for i, c in enumerate(pairRec):
                if pairChainCnt[i] >= lastHand.chain and c-pairChainCnt[i]+1 > lastHand.primal: # able to follow
                    allHands.append(list(range(c-lastHand.chain+1, c+1))*2)
            return allHands
        if lastHand.type == "Trio":
            for i, c in enumerate(trioRec):
                if trioChainCnt[i] >= lastHand.chain and c-trioChainCnt[i]+1 > lastHand.primal: # able to follow
                    kickers = CardInterpreter.getKickers(cards, lastHand.kickerNum, list(range(c-lastHand.chain+1, c+1)))
                    if len(kickers) >= lastHand.chain:
                        allHands.append([kickers])
                        allHands[-1].extend(list(range(c-lastHand.chain+1, c+1))*3)
            return allHands

        # Here, lastHand.type = "Pass". you can play any type you want
        allHands.remove([]) # remove "Pass" actions, you must take out some cards
        for i, c in enumerate(soloRec): # record solo and solo chains (>=5)
            allHands.append([c])
            if soloChainCnt[i] >= 5: # able to play
                for length in range(5, soloChainCnt[i]+1):
                    allHands.append(list(range(c-length+1, c+1)))
        for i, c in enumerate(pairRec): # record pair and pair chains (>=3)
            allHands.append([c]*2)
            if pairChainCnt[i] >= 3: # able to play
                for length in range(3, pairChainCnt[i]+1):
                    allHands.append(list(range(c-length+1, c+1))*2)
        for i, c in enumerate(trioRec): # record trio and trio chains i.e. airplane (>=2)
            allHands.append([c]*3)
            for knum in range(1,3):
                kickers = CardInterpreter.getKickers(cards, knum, [c])
                if len(kickers) > 0:
                    allHands.append([kickers])
                    allHands[-1].extend([c]*3)
            if trioChainCnt[i] >= 2: # able to play
                for length in range(2, trioChainCnt[i]+1):
                    for knum in range(1,3):
                        kickers = CardInterpreter.getKickers(cards, knum, list(range(c-length+1, c+1)))
                        if len(kickers) >= length:
                            allHands.append([kickers])
                            allHands[-1].extend(list(range(c-length+1, c+1))*3)
        for i, c in enumerate(fourRec):
            for knum in range(1,3):
                kickers = CardInterpreter.getKickers(cards, knum, [c])
                if len(kickers) >= 2:
                    allHands.append([kickers])
                    allHands[-1].extend([c]*4)

        return allHands
    
    @staticmethod
    def getKickers(cards, kickerNum, primCards):
        point = Hand.getCardPoint(cards) # get the point expressions
        pointU = list(set(point)) # delete duplicate
        res = []

        # including card 2 (point == 12)
        if kickerNum == 1:
            res = [[p] for p in pointU] # including joker cards
            for p in primCards:
                if [p] in res:
                    res.remove([p])

        if kickerNum == 2:
            pointUCnt = [point.count(p) for p in pointU] # count the number of each point
            res = [[p]*2 for i, p in enumerate(pointU) if pointUCnt[i] >= 2] # p >= 13: no pair/trio/four
            for p in primCards:
                if [p,p] in res:
                    res.remove([p,p])
        return res

    # Given a hand of point set, select out a set of card of this pattern
    @staticmethod
    def selectCardByHand(cards, pointSet):
        cardsNotUsed = cards[:]
        pointNotUsed = Hand.getCardPoint(cards)
        cardsSelected = []
        for p in pointSet:
            index = pointNotUsed.index(p)
            cardsSelected.append(cardsNotUsed[index])
            cardsNotUsed.pop(index)
            pointNotUsed.pop(index)

        return cardsSelected