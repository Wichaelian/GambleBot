import numpy as np
import math
import itertools
import random
from scipy import stats

np.seterr(all='raise')


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
    acc = 5
    prob_vector = np.zeros(10)

    seen_cards = set([str(i[0]) + '_' + i[1] for i in (cards + com_cards)])
    com_cards_set = set([str(i[0]) + '_' + i[1] for i in com_cards])
    current_subsets = list(itertools.combinations(seen_cards, 5))
    curr_hand = 0
    for quint in current_subsets:
        if quint != com_cards_set:
            possible_5 = flatten([[int(card[0]), (card[2:])]
                                 for card in quint])
            res, _, _, _ = classify(possible_5)
            curr_hand = max(res, curr_hand)
    # current_hand, _, _, _ = classify(flatten(cards + com_cards))
    "simulate (5 - # dealt) more cards being dealt acc times"
    for i in range(acc):
        remainder, _ = deal_x_cards(7-len(com_cards), seen_cards)

        non_player_cards = set([str(i[0]) + '_' + i[1]
                               for i in (remainder + com_cards)])

        all_cards = seen_cards | non_player_cards

        "get all possible 5 card subsets out of 7 total cards"
        com_subsets = list(itertools.combinations(all_cards, 5))

        "get classfication of 5 cards (as long as its not all community cards"
        res_max = 0
        for quint in com_subsets:
            if quint != non_player_cards:
                possible_5 = flatten([[int(card[0]), (card[2:])]
                                     for card in quint])
                res, _, _, _ = classify(possible_5)

                "update best possible result"
                res_max = max(res, res_max)

        prob_vector[res_max] += 1

    "normalize prob_vector"
    prob_vector = prob_vector/acc

    "update prob_vector if some hand is already existing"
    prob_vector[curr_hand] = 1

    return prob_vector, seen_cards


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
    else:
        return 0
    return 0


def score(cards, com_cards):
    probs, _ = hand_probs(cards, com_cards)
    # hands = np.array([2**i for i in range(10)])
    hands = np.array([math.exp(i) for i in range(10)])

    one_dex = np.where(probs == 1)[0]
    sub_probs = probs[one_dex[-1]:]
    sub_hands = hands[one_dex[-1]:]
    score = np.average(hands, weights=probs)

    return score

def calculate_bet(prob_dict_obj, cards, com_cards, curr_bet, pot, neighbors):
    pot_odds = curr_bet/(curr_bet + pot)
    hand_score = score(cards, com_cards)
    hand_percentile = stats.percentileofscore(
        prob_dict_obj.score_list, hand_score, 'weak')

    top_probs = np.empty(0)
    top_scores = np.empty(0)
    for hand in sorted(prob_dict_obj.hand_prob_dict, key=prob_dict_obj.hand_prob_dict.get, reverse=True)[:neighbors]:
        top_probs = np.append(top_probs, prob_dict_obj.hand_prob_dict[hand])
        top_scores = np.append(top_scores, prob_dict_obj.score_dict[hand])

    avg_score = np.average(top_scores, weights=top_probs)
    avg_score_percentile = stats.percentileofscore(
        prob_dict_obj.score_list, avg_score, 'weak')

    win_prob = hand_percentile / (hand_percentile + avg_score_percentile)

    if win_prob < pot_odds:
        decision = 'Fold'
        size = 0
    elif win_prob > min(0.7, pot_odds + 0.2 + prob_dict_obj.adjust):
        decision = 'Raise'
        size = (2*curr_bet) + pot
    else:
        rng = np.random.uniform(0, 100, 1)
        if rng >= 100*win_prob:
            decision = 'Call'
            size = curr_bet
        else:
            decision = 'Raise'
            size = (2*curr_bet)/win_prob

    return decision, size, (avg_score_percentile - hand_percentile)


def decision_maker(dict_obj_list, my_cards, com_cards, curr_bet, pot, neighbors):
    decision_dict = {'Raise': 0, 'Call': 0, 'Fold': 0}
    size_avg = 0
    diff_avg = 0
    players = len(dict_obj_list)
    for obj in dict_obj_list:
        decision, size, differential = calculate_bet(
            obj, my_cards, obj.known_cards, curr_bet, pot, neighbors)
        decision_dict[decision] += 1
        size_avg += size/players
        diff_avg += differential/players

    ultimate_decision = max(decision_dict, key=decision_dict.get)
    if ultimate_decision == 'Raise':
        return ['Raise', max(2*curr_bet, size_avg), diff_avg]
    if ultimate_decision == 'Call':
        return ['Call', curr_bet, diff_avg]
    else:
        return ['Fold', 0, diff_avg]


class prob_dictionary:

    def __init__(self, my_cards, known_cards, action_ct, raise_scale=1, risk_adjust=0):

        self.known_cards = known_cards
        deck = set()
        for suit in range(1, 5):
            for rank in range(1, 14):
                deck.add(str(suit) + "_" + str(rank))
        deck = deck - set([str(i[0]) + '_' + i[1]
                          for i in (my_cards + known_cards)])

        probs_hrcf = {}
        hands_list = sorted(list(itertools.combinations(deck, 2)))
        hands_ct = len(hands_list)

        prob_sum = 0
        raise_sum = 0
        call_sum = 0
        fold_sum = 0

        scores = []
        self.score_dict = {}
        hand_prob_dict = {}

        for hand in hands_list:

            hand_score_format = [[int(card[0]), (card[2:])] for card in hand]
            hand_score = score(hand_score_format, known_cards)
            self.score_dict[hand] = hand_score
            scores.append(hand_score)

        scores_percentiles = [stats.percentileofscore(
            scores, s, 'weak') for s in scores]

        for index, hand in enumerate(hands_list):

            hand_prob = 100/hands_ct
            hand_prob_dict[hand] = hand_prob
            prob_sum += 100

            hand_percentile = scores_percentiles[index]
            raise_prob = hand_percentile
            raise_sum += raise_prob

            call_prob = (100 - hand_percentile) * (hand_percentile / 100)
            call_sum += call_prob

            fold_prob = 100 - (raise_prob + call_prob)
            fold_sum += fold_prob

            probs_hrcf[hand] = [hand_prob, raise_prob, call_prob, fold_prob]

        self.prob_dict = probs_hrcf
        self.hand_prob_dict = dict(sorted(hand_prob_dict.items()))
        self.score_list = scores

        self.prob_sum = prob_sum
        self.raise_sum = raise_sum
        self.call_sum = call_sum
        self.fold_sum = fold_sum

        self.prob_raise = self.raise_sum/self.prob_sum
        self.prob_call = self.call_sum/self.prob_sum
        self.prob_fold = self.fold_sum/self.prob_sum

        self.action_ct = action_ct
        self.raise_scale = raise_scale
        self.adjust = risk_adjust

    def update_probs_action(self, action, curr_bet, prev_bet):
        self.prob_raise = max(0.001, self.prob_raise)
        prob_dictionary = self.prob_dict
        action_map = {'Raise': 1, 'Call': 2, 'Fold': 3}
        action_code = action_map[action]
        self.action_ct[action_code - 1] += 1

        for hand, prob_list in prob_dictionary.items():
            hand_prob = prob_list[0]

            r_prob = prob_list[1]
            c_prob = prob_list[2]
            f_prob = prob_list[3]

            if action_code == 1:
                hand_prob_given_action = min(
                    1, self.raise_scale*(curr_bet/prev_bet) * (r_prob * hand_prob) / (self.prob_raise * 200))
                self.hand_prob_dict[hand] = hand_prob_given_action
                if hand_prob_given_action == 0:
                    raise_prob_given_hand = r_prob
                else:
                    raise_prob_given_hand = (
                        hand_prob * r_prob) / hand_prob_given_action

                remainder = 100 - raise_prob_given_hand
                other_prob_sum = c_prob + f_prob
                if other_prob_sum == 0:
                    call_prob_given_hand = 0
                    fold_prob_given_hand = 0
                else:
                    call_prob_given_hand = (
                        c_prob * remainder) / other_prob_sum
                    fold_prob_given_hand = (
                        f_prob * remainder) / other_prob_sum

                self.prob_sum += 100 * (hand_prob_given_action - hand_prob)
                self.raise_sum += (raise_prob_given_hand - r_prob)
                self.call_sum += (call_prob_given_hand - c_prob)
                self.fold_sum += (fold_prob_given_hand - f_prob)
                self.prob_raise = self.raise_sum/self.prob_sum
                self.prob_call = self.call_sum/self.prob_sum
                self.prob_fold = self.fold_sum/self.prob_sum

            elif action_code == 2:
                hand_prob_given_action = (
                    c_prob * hand_prob) / (self.prob_call * 100)
                self.hand_prob_dict[hand] = hand_prob_given_action
                if hand_prob_given_action == 0:
                    call_prob_given_hand = c_prob
                else:
                    call_prob_given_hand = (
                        hand_prob * c_prob) / hand_prob_given_action

                remainder = 100 - call_prob_given_hand
                other_prob_sum = r_prob + f_prob
                if other_prob_sum == 0:
                    raise_prob_given_hand = 0
                    fold_prob_given_hand = 0
                else:
                    raise_prob_given_hand = (
                        r_prob * remainder) / other_prob_sum
                    fold_prob_given_hand = (
                        f_prob * remainder) / other_prob_sum

                self.prob_sum += (hand_prob_given_action - hand_prob)
                self.raise_sum += (raise_prob_given_hand - r_prob)
                self.fold_sum += (fold_prob_given_hand - c_prob)
                self.fold_sum += (fold_prob_given_hand - f_prob)
                self.prob_raise = self.raise_sum/self.prob_sum
                self.prob_call = self.call_sum/self.prob_sum
                self.prob_fold = self.fold_sum/self.prob_sum

            elif action_code == 3:
                hand_prob_given_action = (
                    f_prob * hand_prob) / (self.prob_fold * 100)
                self.hand_prob_dict[hand] = hand_prob_given_action
                if hand_prob_given_action == 0:
                    fold_prob_given_hand = f_prob
                else:
                    fold_prob_given_hand = (
                        hand_prob * f_prob) / hand_prob_given_action

                remainder = 100 - fold_prob_given_hand
                other_prob_sum = r_prob + c_prob
                if other_prob_sum == 0:
                    raise_prob_given_hand = 0
                    call_prob_given_hand = 0
                else:
                    raise_prob_given_hand = (
                        r_prob * remainder) / other_prob_sum
                    call_prob_given_hand = (
                        c_prob * remainder) / other_prob_sum

                self.prob_sum += (hand_prob_given_action - hand_prob)
                self.raise_sum += (raise_prob_given_hand - r_prob)
                self.call_sum += (call_prob_given_hand - c_prob)
                self.fold_sum += (fold_prob_given_hand - f_prob)
                self.prob_raise = self.raise_sum/self.prob_sum
                self.prob_call = self.call_sum/self.prob_sum
                self.prob_fold = self.fold_sum/self.prob_sum

            prob_dictionary[hand] = [hand_prob_given_action,
                                     raise_prob_given_hand, call_prob_given_hand, fold_prob_given_hand]

        self.prob_dict = prob_dictionary
        self.hand_prob_dict = dict(sorted(self.hand_prob_dict.items()))

    def update_probs_ncard(self, card):
        self.known_cards.append(card)
        if len(self.known_cards) == 7:
            return
        prob_dictionary = self.prob_dict
        hand_prob_dict = self.hand_prob_dict
        new_overall = {}
        new_hand_probs = {}
        new_score_dict = {}
        for hand, prob_list in prob_dictionary.items():
            if card in hand:
                self.prob_sum -= prob_list[0]
                self.raise_sum -= prob_list[1]
                self.call_sum -= prob_list[2]
                self.fold_sum -= prob_list[3]

                self.prob_raise = self.raise_sum/self.prob_sum
                self.prob_call = self.call_sum/self.prob_sum
                self.prob_fold = self.fold_sum/self.prob_sum

            else:
                new_overall[hand] = prob_list
                new_hand_probs[hand] = prob_list[0]
                hand_score_format = [[int(card[0]), (card[2:])]
                                     for card in hand]
                hand_score = score(hand_score_format, self.known_cards)
                new_score_dict[hand] = hand_score

        self.prob_dict = new_overall
        self.hand_prob_dict = dict(sorted(new_hand_probs.items()))
        self.score_dict = new_score_dict
        self.score_array = np.array([val for _, val in self.score_dict])

    def update_winner(self):
        actions_this_round = self.action_ct
        self.raise_scale -= ((actions_this_round[0]
                             * 0.02) + (actions_this_round[1] * 0.01))
        self.adjust += 0.02

    def update_loser(self):
        actions_this_round = self.action_ct
        self.raise_scale += ((actions_this_round[0]
                             * 0.02) + (actions_this_round[1] * 0.01))
        self.adjust -= 0.01

    # def round_complete(self):
    #     return self.raise_scale, self.adjust


opp_ct = 3
obj_list = []
for opp in range(opp_ct):
    obj_list.append(prob_dictionary([[2, '7'], [2, '8']], [
                    [3, '12'], [1, '5'], [4, '6']], np.array([0, 0, 0]), 1, 0))

obj_list[0].update_probs_action("Call", 0, 0)
obj_list[1].update_probs_action("Call", 0, 0)
obj_list[2].update_probs_action('Call', 0, 0)

print(decision_maker(obj_list, [[2, '7'], [2, '8']], [
      [3, '12'], [1, '5'], [4, '6']], 5, 40, 100))

# obj_list[0].update_probs_action("Call", 186, 186)
# obj_list[1].update_probs_action("Fold", 0, 186)
# obj_list[2].update_probs_action('Raise', 372, 186)

# print(decision_maker(obj_list, [[2, '7'], [2, '8']], [[3, '12'], [2, '5'], [2, '6']], 372, 648, 100))
