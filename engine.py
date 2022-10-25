import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def classify(hand):
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
    rank_ct_map = {(4, 1): 7, (3, 2): 6, (3, 1): 3, (2, 2): 2, (2, 1): 1, (1, 1): 0}
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

def deal_x_cards(x, seen):
    deck = set()
    res = []
    for suit in range(1, 5):
        for rank in range(1, 14):
            deck.add(str(suit) + "_" + str(rank))
    drawn = random.sample(list(deck), k=x)
    for card in drawn:
        res.append([int(card[0]), (card[2:])])
    return res, seen | set(drawn)


class GameEngine:
    
    def __init__(self, init_stack, player_ct, sm, big):
        self.player_ct = player_ct
        self.play_status = [True for i in range(player_ct)]
        self.bot_status = self.play_status[0]
        
        self.dealer = random.randrange(player_ct)
        self.sm_blind = sm
        self.big_blind = big
        self.curr_bet = big
        self.pot = sm + big
        
        self.player_stacks = [init_stack for i in range(player_ct)]
        self.bot_stack = self.player_stacks[0]
        
        self.player_stacks[self.dealer] = max(0, self.player_stacks[self.dealer] - sm)
        self.player_stacks[(self.dealer + 1) % player_ct] = max(0, self.player_stacks[(self.dealer + 1) % player_ct] - big)
        
        self.player_cards = [(None, None) for i in range(player_ct)]
        self.bot_card = self.player_cards[0]
        self.com_cards = []
        self.seen = set()
        
    def bet(self, player, amt):
        assert amt >= self.curr_bet
        assert self.player_stacks[player] >= amt
        self.player_stacks[player] -= amt
        self.pot += amt
        self.curr_bet = amt
    
    def call(self, player):
        self.call(player, amt=self.curr_bet)
    
    def fold(self, player):
        self.play_status[player] = False
    
    def deal_hands(self):
        players = self.player_ct
        self.com_cards, self.seen = deal_x_cards(3, self.seen)
        target = self.dealer
        if self.play_status[target] == True:
                self.player_cards[target], self.seen = deal_x_cards(2, self.seen)
        target = (target + 1) % players
        while target != self.dealer:
            if self.play_status[target] == True:
                self.player_cards[target], self.seen = deal_x_cards(2, self.seen)
            target = (target + 1) % players
    
first_game = GameEngine(25, 4, 0.25, 0.5)
first_game.deal_hands()
print(first_game.com_cards)
print(first_game.player_stacks)
print(first_game.player_cards)
    