import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import typing as T
from bot_bets import preflop_bet, calculate_bet


def classify(hand: T.List[T.List]) -> int:
    """
    Classifies a given hand using rules described at https://archive.ics.uci.edu/ml/datasets/Poker+Hand.

    Parameter:
    hand - Poker hand to classify. 

    Return:
    The integer code for the hand classification. 
    """
    suit = {}
    rank = {}
    r = 0
    res = 0
    straight = 0
    for index, atr in enumerate(hand):
        if index % 2 == 0:
            if atr in suit:
                suit[atr] += 1
            else:
                suit[atr] = 1
        else:
            if atr in rank:
                rank[atr] += 1
            else:
                rank[atr] = 1
    s_max = max(suit.values())
    if s_max == 5:
        res = max(res, 5)
    r = max(rank.values())
    second = sorted(rank.values())[-2]
    rank_ct_map = {(4, 1): 7, (3, 2): 6, (3, 1): 3,
                   (2, 2): 2, (2, 1): 1, (1, 1): 0}
    res = max(res, rank_ct_map[(r, second)])
    rank_list = sorted(rank.keys())
    if rank_list[0] == 1 and rank_list[-1] == 13:
        seq = [1, 10, 11, 12, 13]
    else:
        seq = [rank_list[0] + i for i in range(5)]
    if rank_list == seq:
        straight = 1
        res = max(res, 4)
    if s_max == 5 and straight == 1:
        if seq[0] == 1 and seq[4] == 13:
            res = 9
        else:
            res = max(res, 8)
    return res


def deal_x_cards(x: int, seen: T.Set):
    """
    Deal x cards to the game, making sure not to deal cards already seen.

    Parameters:
    x - number of cards to deal
    seen - set of already seen cards 
    """
    deck = set()
    res = []
    for suit in range(1, 5):
        for rank in range(1, 14):
            deck.add(str(suit) + "_" + str(rank))
    deck = deck - seen
    drawn = random.sample(list(deck), k=x)
    for card in drawn:
        res.append([int(card[0]), (card[2:])])
    return res, seen | set(drawn)


def pf_bet(rank, curr_bet, pot):
    """
    Bot 
    """

    size = preflop_bet(rank, curr_bet, pot)
    return ['Bet', size]

def in_game_bet(cards, com_cards, curr_bet, pot):
    """
    Bot 
    """
    return calculate_bet(cards, com_cards, curr_bet, pot)


class GameEngine:

    """
    Initialize a game with a set initial stack amount per player, 
    number of players, and small/big blind values.

    """

    def __init__(self, init_stack, player_ct, sm, big):
        self.player_ct = player_ct
        self.hand_ct = player_ct
        self.play_status = [True for i in range(player_ct)]
        self.bot_status = self.play_status[0]

        self.dealer = random.randrange(player_ct)
        self.sm_blind = sm
        self.big_blind = big
        self.curr_bet = big
        self.pot = sm + big

        self.player_stacks = [init_stack for i in range(player_ct)]
        self.bot_stack = self.player_stacks[0]

        self.player_stacks[self.dealer +
                           1] = max(0, self.player_stacks[self.dealer] - sm)
        self.player_stacks[(self.dealer + 2) % player_ct] = max(0,
                                                                self.player_stacks[(self.dealer + 2) % player_ct] - big)

        self.player_cards = [(None, None) for i in range(player_ct)]
        self.bot_card = self.player_cards[0]
        self.com_cards = []
        self.seen = set()

        self.pf_range = {
            1: {'1_1o', '13_1s', '12_1s', '11_1s', '10_1s', '9_1s', '8_1s', '7_1s', '6_1s', '5_1s', '4_1s', '3_1s', '2_1s',
                '13_1o', '12_1o', '11_1o', '10_1o',
                '13_13o', '13_12s', '13_11s', '13_10s', '13_12o',
                '12_12o', '12_11s', '12_10s',
                '11_11o', '11_10s',
                '10_10o', '10_9s',
                '9_9o'},
            2: {'13_9s', '13_8s', '13_11o', '12_9s', '12_11o', '11_9s', '9_8s', '8_7s', '7_6s', '5_4s'},
            3: {'13_7s', '13_6s', '13_5s', '13_10o', '12_8s', '12_10o', '11_8s', '11_10o',
                '10_8s', '9_7s', '8_6s', '7_5s', '3_3o', '2_2o'},
            4: {'13_4s', '13_3s', '13_2s', '12_7s', '12_6s', '12_5s', '11_7s', '10_7s', '10_6s',
                '9_1o', '8_1o', '7_1o', '6_1o', '5_1o', '4_1o',
                '13_9o', '12_9o', '11_9o', '10_9o', '9_6s', '9_8o', '8_5s', '6_4s', '5_3s', '4_3s'},
            5: {'3_1o', '2_1o', '12_4s', '12_3s', '12_2s', '13_8o', '13_7o', '13_6o', '13_5o', '13_4o',
                '11_6s', '11_5s', '11_4s', '11_3s', '11_2o', '11_2s', '12_8o', '12_7o', '12_6o', '12_5o',
                '10_5s', '10_4s', '10_3s', '10_2s', '9_5s', '9_4s', '11_8o', '11_7o', '10_8o', '10_7o', '9_7o',
                '8_4s', '8_7o', '8_6o', '7_4s', '7_3s', '7_6o', '7_5o', '6_3s', '6_5o', '6_4o',
                '5_2s', '5_4o', '4_2s', '3_2s'}
        }

    def bet(self, player, amt):
        assert amt >= 2*self.curr_bet
        assert self.player_stacks[player] >= amt
        self.player_stacks[player] -= amt
        self.pot += amt
        self.curr_bet = amt

    def call(self, player):
        self.bet(player, amt=self.curr_bet)

    def fold(self, player):
        self.play_status[player] = False
        self.hand_ct -= 1

    def deal_hands(self):
        players = self.player_ct
        self.com_cards, self.seen = deal_x_cards(3, self.seen)
        target = self.dealer
        if self.play_status[target] == True:
            self.player_cards[target], self.seen = deal_x_cards(2, self.seen)
        target = (target + 1) % self.player_ct
        while target != self.dealer:
            if self.play_status[target] == True:
                self.player_cards[target], self.seen = deal_x_cards(
                    2, self.seen)
            target = (target + 1) % self.player_ct

    def profile_pf_bet(self, i):
        """
        Preflop betting for non-AI players (only used during training phase)
        Logic:
        We use random numbers to introduce a non-deterministic quality so the 
        bot doesn't become predictable. i represents the range of the preflop
        quality that the current player's hand is in, with 1 being top and 4 being
        bottom.  
        """
        num = np.random.uniform(0, 100, 1)
        if i == 1:
            if num <= 95:
                return ['Bet', self.curr_bet]
        elif i == 2:
            if num <= 75:
                return ['Bet', self.curr_bet]
        elif i == 3:
            if num <= 50:
                return ['Bet', self.curr_bet]
        elif i == 4:
            if num <= 20:
                return ['Bet', self.curr_bet]    
        

    def pf_play(self):
        target = (self.dealer + 3) % self.player_ct
        position = 1
        while target - 1 != self.dealer:
            hand = self.player_cards[target]
            c1 = int(hand[0][1])
            c2 = int(hand[1][1])
            if hand[0][0] == hand[1][0]:
                encode = str(max(c1, c2)) + '_' + str(min(c1, c2)) + 's'
            else:
                encode = str(max(c1, c2)) + '_' + str(min(c1, c2)) + 'o'
            for i in range(1, position+1):
                options = self.pf_range[i]
                if encode in options:
                    if target == 0:
                        decision = pf_bet(i, self.curr_bet, self.pot)
                    else:
                        decision = self.profile_pf_bet(i)
                    if decision[0] == 'Bet':
                        self.bet(target, decision[1])
                    else:
                        self.fold(target)
                        self.hand_ct -= 1
                elif i == position:
                    self.fold(target)
                    self.hand_ct -= 1
            target = (target + 1) % self.hand_ct
            position += 1
        self.play()

    def play(self):
        return 0



first_game = GameEngine(25, 6, 0.25, 0.5)
first_game.deal_hands()
print(first_game.com_cards)
print(first_game.player_stacks)
print(first_game.player_cards)
print(first_game.seen)
