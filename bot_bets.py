import numpy as np
import math
import itertools
import random

def deal_x_cards(x, seen):
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


def classify(hand) -> int:
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
    return res, suit, rank, set(seq)

def flatten(l):
    return [int(item) for sublist in l for item in sublist]

def hand_probs(cards, com_cards):
    acc = 10000
    prob_vector = np.zeros(10)

    seen_cards = set([str(i[0]) + '_' + i[1] for i in (cards + com_cards)])
    current_hand, _, _, _ = classify(flatten(cards + com_cards))
    # return 0
    "simulate 2 more cards being dealt acc times"
    for i in range(acc):
        remainder, _ = deal_x_cards(2, seen_cards)

        "get all possible 3 card subsets out of 5 total community cards"
        com_subsets = list(itertools.combinations(set([str(i[0]) + '_' + i[1] for i in (remainder + com_cards)]), 3))
        
        "get classfication of 2 cards in hand and a 3 card subset"
        for trip in com_subsets:
            trip_list = [[int(card[0]), (card[2:])] for card in trip]
            possible_5 = flatten(trip_list + cards)
            res, _, _, _ = classify(possible_5)

            "record instance in prob_vector"
            prob_vector[res] += 1
    
    "normalize prob_vector"
    prob_vector = prob_vector/(10*acc)
    
    "update prob_vector if some hand is already existing"
    prob_vector[current_hand] = 1
    print(prob_vector)

    return prob_vector



def preflop_bet(rank, curr_bet, pot):
    pot_odds = curr_bet/(curr_bet + pot)
    num = np.random.uniform(0, 100, 1)
    if rank == 1:
        if num <= 95:
            return math.floor((1/pot_odds)*curr_bet)
    elif rank == 2:
        if num <= 75:
            return math.floor((1/pot_odds)*curr_bet)
    elif rank == 3:
        if num <= 50:
            return math.floor((1/pot_odds)*curr_bet)
    elif rank == 4:
        if num <= 20:
            return math.floor((1/pot_odds)*curr_bet)

    

def calculate_bet(cards, com_cards, curr_bet, pot):
    pot_odds = curr_bet/(curr_bet + pot)
    probs = hand_probs(cards, com_cards)
    return 0

hand_probs([[3, '10'], [4, '7']], [[4, '8'], [2, '9'], [3, '11']])