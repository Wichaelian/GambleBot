import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import typing as T
from bot_bets import preflop_bet, calculate_bet, prob_dictionary, prob_dictionary, decision_maker
from itertools import combinations
from collections import defaultdict


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
    for suit_c, rank_c in hand:
        if suit_c in suit:
            suit[suit_c] += 1
        else:
            suit[suit_c] = 1
        if rank_c in rank:
            rank[rank_c] += 1
        else:
            rank[rank_c] = 1
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
        seq = [rank_list[0] + str(i) for i in range(5)]
    if rank_list == seq:
        straight = 1
        res = max(res, 4)
    if s_max == 5 and straight == 1:
        if seq[0] == 1 and seq[4] == 13:
            res = 9
        else:
            res = max(res, 8)
    return res


def deal_x_cards(x: int, seen: T.Set) -> T.Tuple[T.List[T.List[int]], T.Set[str]]:
    """
    Deal x cards to the game, making sure not to deal cards already seen and update
    the seen set.

    Parameters:
    x: number of cards to deal
    seen: set of already seen cards

    Return: resulting dealt cards and updated set of seen cards
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


def preflop_bet_bot(rank: int, curr_bet: float, pot: float) -> T.List:
    """
    Preflop betting for the bot. Given an amount for the pot, the current bet, and
    the rank of the hand, calculate whether to call or raise.

    Parameters:
    rank: hand rank
    curr_bet: current bet in this round
    pot: amount of money in the pot

    Return:
    List with [raise/call, amount to raise or call]
    """
    curr = curr_bet
    size = preflop_bet(rank, curr_bet, pot)
    print("size: ", str(size))
    if (size >= 2*curr):
        return ['Raise', size]
    return ['Call', curr_bet]


def postflop_bet_bot(bet_in_play: bool, rank: int, curr_bet: float, pot: float) -> T.List:
    """
    Postflop betting for the bot. Given an amount for the pot, the current bet, and
    the rank of the hand, calculate whether to bet, call, raise, or check.

    Parameters:
    betinplay: if there is a bet currently in play
    rank: hand rank
    curr_bet: current bet in this round
    pot: amount of money in the pot

    Return:
    List with [raise/call, amount to raise or call]
    """
    if not bet_in_play:
        return ['Check', 0]
    else:
        curr = curr_bet
        size = preflop_bet(rank, curr_bet, pot)
        print("size: ", str(size))
        if (size >= 2*curr):
            return ['Raise', size]


def in_game_bet(cards, com_cards, curr_bet, pot):
    """
    Bot

    ****** NOT CURRENTLY IN USE *******
    """
    return calculate_bet(cards, com_cards, curr_bet, pot)


class GameEngine:
    """
    Poker game engine to track game state throughout play and update
    game play.

    ******
    Class Variables
    ******

    player_ct: number of players
    hand_ct: number of active players, ie, have not folded
    play_status: list of which players have folded, in order of player index
    moveleft_status: list tracking whether player i has a move left in this round of betting
    bot_status: whether the bot is in play. bot is player 0 by default

    dealer: player number representing the dealer
    sm_blind: amount that small blind bets
    big_blind: amount that big blind bets
    curr_bet: current betting amount, starts off as big blind value
    pot: amount of money in the pot. Initially small blind bet + big blind bet

    player_stacks: amount of money each player has
    bot_stack: amount of money bot has

    player_cards: list containing tuples of each player's hand
    bot_card: the bot's hand
    com_cards: community cards currently visible
    seen: set of seen cards

    pf_range: preflop ranges for hands
    """

    def __init__(self, init_stack: float, player_ct: int, sm: float, big: float):
        """
        Initialize a game engine according to the class specification with a given
        number of players, stack per player, and big and small blind bet amounts.
        """
        self.efficiency = 0
        self.differential_sum = 0
        self.efficiency_list = []
        self.bot_PnL_list = []
        self.player_ct = player_ct
        self.hand_ct = player_ct
        self.play_status = [True for i in range(player_ct)]
        self.moveleft_status = [True for i in range(player_ct)]
        self.bot_status = self.play_status[0]
        self.buy_in = init_stack

        self.dealer = random.randrange(player_ct)

        self.sm_blind = sm
        self.big_blind = big
        self.curr_bet = big
        self.pot = sm + big

        self.round_in_play = 0

        self.prev_bet = self.big_blind

        self.player_stacks = [init_stack for i in range(player_ct)]
        self.bot_stack = self.player_stacks[0]

        self.player_stacks[(self.dealer +
                           1) % player_ct] = max(0, self.player_stacks[self.dealer] - sm)
        self.player_stacks[(self.dealer + 2) % player_ct] = max(0,
                                                                self.player_stacks[(self.dealer + 2) % player_ct] - big)

        self.player_cards = [(None, None) for i in range(player_ct)]
        self.bot_card = self.player_cards[0]
        self.com_cards = []
        self.seen = set()

        self.prob_matrices = []
        self.action_statuses = {}
        for i in range(self.player_ct):
            self.action_statuses[i] = (1, 0)

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

    def check(self, player: int) -> None:
        """
        Given player checks on their turn, meaning does nothing. Only permitted
        if there have been no bets in the round yet.

        Parameter:
        player: player that checks. Must be within [0, player_ct)
        """
        assert player in range(self.player_ct)
        pass

    def bet(self, player: int, amt: float, postflop=False) -> None:
        """
        Bet a given amount for a given player. Update game state accordingly

        Parameters:
        player: player that is betting. Must be within range [0, player_ct)
        amt: amount player is betting. Must be either a call or raise amount

        Precondition:
        player must have at least amt money in its stack
        """
        if self.player_stacks[player] < amt:
            amt = self.player_stacks[player]
        self.player_stacks[player] -= amt
        self.pot += amt
        self.curr_bet = amt
        if postflop:
            self.prob_matrices[player-1-1].update_probs_action(
                "Raise", amt, self.prev_amt)
            self.prev_amt = amt

    def call(self, player: int, postflop=False) -> None:
        """
        Given player calls, or bets current amount.

        Parameters:
        player: player that calls. Must be within range [0, player_ct)
        """
        assert player in range(self.player_ct)
        self.bet(player, amt=self.curr_bet)
        if postflop:
            self.prob_matrices[player-1-1].update_probs_action(
                "Call", self.curr_bet, self.prev_amt)
            self.prev_amt = self.curr_bet

    def fold(self, player: int) -> None:
        """
        Given player folds. Update game state by marking this player inactive and
        reducing number of active hands.

        Parameters:
        player: player that folds. Must be within range [0, player_ct)
        """
        assert player in range(self.player_ct)
        self.play_status[player] = False
        self.hand_ct -= 1

    def deal_hands(self) -> None:
        """
        Deal hands for each player.
        """
        target = self.dealer
        if self.play_status[target] == True:
            self.player_cards[target], self.seen = deal_x_cards(2, self.seen)
        target = (target + 1) % self.player_ct
        while target != self.dealer:
            if self.play_status[target] == True:
                self.player_cards[target], self.seen = deal_x_cards(
                    2, self.seen)
            target = (target + 1) % self.player_ct
        self.com_cards, self.seen = deal_x_cards(3, self.seen)

    def profile_preflop_bet(self, i: int) -> T.List:
        """
        Preflop betting for non-AI players (only used during training phase)
        Logic:
        We use random numbers to introduce a non-deterministic quality so the
        bot doesn't become predictable. i represents the range of the preflop
        quality that the current player's hand is in, with 1 being top and 4 being
        bottom.

        Parameters:
        i: rank of player hand that we're betting on
        """

        num = random.uniform(0, 100)
        if i == 1:
            if num <= 95:
                raiseorcall = random.uniform(0, 100)
                if raiseorcall <= 60:
                    return ['Raise', 2*self.curr_bet]
                else:
                    return ['Call', self.curr_bet]
        elif i == 2:
            if num <= 75:
                raiseorcall = random.uniform(0, 100)
                if raiseorcall <= 60:
                    return ['Raise', 2*self.curr_bet]
                else:
                    return ['Call', self.curr_bet]
        elif i == 3:
            if num <= 50:
                raiseorcall = random.uniform(0, 100)
                if raiseorcall <= 60:
                    return ['Raise', 2*self.curr_bet]
                else:
                    return ['Call', self.curr_bet]
        elif i == 4:
            if num <= 20:
                raiseorcall = random.uniform(0, 100)
                if raiseorcall <= 60:
                    return ['Raise', 2*self.curr_bet]
                else:
                    return ['Call', self.curr_bet]
        return ['Fold', 0]

    def check_if_no_moves_left(self, arr: T.List[T.List[bool]]) -> bool:
        """
        Check if any players have moves left based on the matrix tracking
        player statuses.

        Return:
        True if no moves are left,
        False if there are possible moves left
        """
        for i in range(len(arr)):
            if arr[i]:
                return False
        return True

    def preflop_play(self) -> None:
        """
        Preflop betting rounds.
        """
        position = (self.dealer + 3) % self.player_ct
        while self.hand_ct > 1 and self.check_if_no_moves_left(self.moveleft_status) == False:
            print("STILL IN PLAY ARRAY IS ", str(self.play_status))
            print("MOVE LEFT ARRAY IS ", str(self.moveleft_status))

            print("dealer: ", str(self.dealer), "position: ", str(position))

            if position >= self.player_ct:
                position = 0

            print(self.play_status)

            if self.play_status[position] == False:
                position += 1
                continue

            if self.moveleft_status[position] == False:
                position += 1
                continue

            hand = self.player_cards[position]
            c1 = int(hand[0][1])
            c2 = int(hand[1][1])
            if hand[0][0] == hand[1][0]:
                encode = str(max(c1, c2)) + '_' + str(min(c1, c2)) + 's'
            else:
                encode = str(max(c1, c2)) + '_' + str(min(c1, c2)) + 'o'
            print("hand encodes to ",  encode)
            diff = abs(self.dealer - position)
            for i in range(1, diff + 1):
                options = self.pf_range[i]
                if encode in options:
                    # SET BOT TO BE POSITION 0.
                    if position == 0:
                        print("Bot")
                        decision = preflop_bet_bot(i, self.curr_bet, self.pot)
                    else:
                        decision = self.profile_preflop_bet(i)
                    print("decision: ", str(decision))
                    if decision[0] == 'Call':
                        self.bet(position, self.curr_bet)

                    elif decision[0] == 'Raise':
                        for i in range(self.player_ct):
                            if self.play_status[i]:
                                self.moveleft_status[i] = True
                        self.bet(position, decision[1])
                    else:
                        # self.fold(position) tmp for testing
                        self.bet(position, self.curr_bet)
                elif i == position:
                    self.fold(position)
            self.moveleft_status[position] = False

            print("end_for")

            print("loop end")
            self.prev_amt = self.curr_bet
            position += 1

    def flop(self, x: int) -> T.List[T.List]:
        """
        Deal x amount of communty cards and print them to the output
        Return the new cards/card 
        """
        new_cards, self.seen = deal_x_cards(x, self.seen)
        print("new", str(new_cards))
        self.com_cards += new_cards
        print("Flop: ", str(self.com_cards))
        return new_cards

    def check_if_no_moves_left(self, arr: T.List[T.List[bool]]) -> bool:
        """
        Check if any players have moves left based on the matrix tracking
        player statuses.

        Return:
        True if no moves are left,
        False if there are possible moves left
        """
        for i in range(len(arr)):
            if arr[i]:
                return False
        return True

    def profile_postflop_bet(self, rank: int, bet_in_play: bool, i: int) -> T.List:
        """
        Preflop betting for non-AI players (only used during training phase)
        Logic:
        We use random numbers to introduce a non-deterministic quality so the
        bot doesn't become predictable. i represents the range of the preflop
        quality that the current player's hand is in, with 1 being top and 4 being
        bottom.

        Parameters:
        i: rank of player hand that we're betting on
        """
        num = random.uniform(0, 100)
        if bet_in_play:
            if rank == 0 or rank == 1:
                if num <= 10:
                    return ['Raise', 2 * self.curr_bet]
                elif num <= 50:
                    return ['Call', self.curr_bet]
                else:
                    return ['Fold', 0]
            elif rank == 2 or rank == 3:
                if num <= 20:
                    return ['Raise', 2 * self.curr_bet]
                elif num <= 80:
                    return ['Call', self.curr_bet]
                else:
                    return ['Fold', 0]
            else:
                if num <= 50:
                    return ['Raise', 2 * self.curr_bet]
                else:
                    return ['Call', self.curr_bet]
        else:
            betorcheck = np.random.uniform(0, 100)
            if rank == 0 or rank == 1:
                if betorcheck <= 15:
                    bet_variation = random.uniform(0, 0.25)
                    return ['Bet', self.big_blind * (1 + bet_variation)]
                else:
                    return ['Check', 0]
            elif rank == 2 or rank == 3:
                if betorcheck <= 60:
                    bet_variation = random.uniform(0, 1)
                    return ['Bet', self.big_blind * (1 + bet_variation)]
                else:
                    return ['Check', 0]
            else:
                if num <= 95:
                    if betorcheck <= 90:
                        bet_variation = random.uniform(0, 2)
                        return ['Bet', self.big_blind * (1 + bet_variation)]
                    else:
                        return ['Check', 0]
        # return ['Fold', 0]

    def postflop_play(self) -> None:
        """
        Postflop game play
        **** TODO: ADD CHECK LOGIC *****
        """
        print("****************PF")
        position = (self.dealer + 2) % self.player_ct
        bet_in_play = False
        for i in range(self.player_ct):
            if self.play_status[i]:
                self.moveleft_status[i] = True
        while self.hand_ct > 1 and self.check_if_no_moves_left(self.moveleft_status) == False:

            print("STILL IN PLAY ARRAY IS ", str(self.play_status))
            print("MOVE LEFT ARRAY IS ", str(self.moveleft_status))

            if position >= self.player_ct:
                position = 0
                print("BOT HAS HAND OF ", self.player_cards[0])

            print("dealer: ", str(self.dealer), "position: ", str(position))

            print(self.play_status)

            if self.player_stacks[position] == 0:
                print("DIS PLAYA BROKE ASF: ", position)
                self.moveleft_status[position] = False
                print("DIS PLAYA BROKE ASF: ", position)
                self.moveleft_status[position] = False
                position += 1
                continue

            if self.play_status[position] == False:
                position += 1
                continue

            if self.moveleft_status[position] == False:
                position += 1
                continue
            if position == 0:
                if not bet_in_play:
                    decision = decision_maker(
                        self.prob_matrices, self.player_cards[0], self.com_cards, self.curr_bet/2, self.pot, 100)
                else:
                    decision = decision_maker(
                        self.prob_matrices, self.player_cards[0], self.com_cards, self.curr_bet, self.pot, 100)
                self.differential_sum += decision[2]
            rank = self.classify_player_best_hand(position)
            decision = self.profile_postflop_bet(rank, bet_in_play, position)
            print("player ", str(position), " decision ", str(decision))
            if decision[0] == 'Bet':
                bet_in_play = True
                for i in range(self.player_ct):
                    if self.play_status[i]:
                        self.moveleft_status[i] = True
                self.bet(position, decision[1], postflop=True)

            elif decision[0] == 'Call':
                self.call(position, postflop=True)

            elif decision[0] == 'Raise':
                for i in range(self.player_ct):
                    if self.play_status[i]:
                        self.moveleft_status[i] = True
                self.bet(position, decision[1], postflop=True)
            else:
                print("betinplay ", str(bet_in_play))
                if bet_in_play == True:
                    self.fold(position)
            self.moveleft_status[position] = False
            print("moves left ", self.moveleft_status)

            print("loop end")
            position += 1
        print("loop over -- 1 player", str(self.hand_ct))

    def classify_player_best_hand(self, player: int) -> int:
        """
        For a given player, return the ranking of their best possible hand
        using the revealed community cards and their cards.
        """
        res_max = -1
        all_cards = self.com_cards + self.player_cards[player]
        com_tuple = tuple(self.com_cards)
        com_subsets = list(combinations(all_cards, 5))
        if len(self.com_cards) > 4:
            com_subsets.remove(com_tuple)
        for quint in com_subsets:
            res = classify(quint)
            if res > res_max:
                res_max = res
        return res_max

    def game_end(self, round_count) -> int:
        """
        Evaluate remaining players' hands and return winner
        """
        scores = [-1 for i in range(self.player_ct)]
        max_score = -1
        winner = -1
        for i in range(self.player_ct):
            if self.play_status[i]:
                res_max = self.classify_player_best_hand(i)
                scores[i] = res_max
                if res_max > max_score:
                    max_score = res_max
                    winner = i
            else:
                if i != 0:
                    self.prob_matrices[i-1].update_loser()
                if i != 0:
                    self.prob_matrices[i-1].update_loser()

        self.player_stacks[winner] += self.pot
        print(scores)
        print("WINNER IS ", winner)
        if round_count > 0:
            d_avg_aggregate = self.differential_sum/round_count
        else:
            d_avg_aggregate = 0

        if i != 0:
            self.prob_matrices[i-1].update_winner()
        self.round_in_play += 1
        
        return d_avg_aggregate, winner

    def play(self) -> None:
        """
        Main gameplay loop.
        """
        self.player_cards = [(None, None) for i in range(self.player_ct)]
        self.bot_card = self.player_cards[0]
        self.com_cards = []

        self.player_cards = [(None, None) for i in range(self.player_ct)]
        self.bot_card = self.player_cards[0]
        self.com_cards = []

        if self.round_in_play >= 1:
            self.hand_ct = 6
            for i in range(self.player_ct):
                if not self.play_status[i]:
                    self.player_stacks[i] = self.buy_in
                    self.pot += self.buy_in
                    self.play_status[i] = True
            self.dealer = (self.dealer + 1) % self.player_ct
            self.curr_bet = self.big_blind
            self.moveleft_status = [True for i in range(self.player_ct)]
            self.moveleft_status = [True for i in range(self.player_ct)]
            self.prev_bet = self.big_blind
        self.seen = set()
        self.deal_hands()
        self.seen = set()
        self.deal_hands()

        print("game 2 reset")

        initial_bot_stack = self.player_stacks[0]

        self.preflop_play()
        if self.hand_ct == 1:
            self.round_in_play += 1
            self.round_in_play += 1
            return

        if self.round_in_play == 0:
            for i in range(0, self.player_ct-1):
                self.prob_matrices.append(prob_dictionary(
                    self.player_cards[i].copy(), self.com_cards.copy(), np.array([0, 0, 0])))

        if self.round_in_play >= 1:
            new_mat = []
            for i in range(0, self.player_ct-1):
                a, e = self.prob_matrices[i].raise_scale, self.prob_matrices[i].adjust

                new_mat.append(prob_dictionary(
                    self.player_cards[i].copy(), self.com_cards.copy(), np.array([0, 0, 0])))

        if self.round_in_play >= 1:
            new_mat = []
            for i in range(0, self.player_ct-1):
                a, e = self.prob_matrices[i].raise_scale, self.prob_matrices[i].adjust

                new_mat.append(prob_dictionary(
                    self.player_cards[i].copy(), self.com_cards.copy(), np.array([0, 0, 0]), raise_scale=a, risk_adjust=e))
            self.prob_matrices = new_mat


        round_count = 0

        if self.play_status[0]:
            round_count += 1
        self.postflop_play()      
        if self.hand_ct == 1:
            self.round_in_play += 1
            self.round_in_play += 1
            return

        n_card = self.flop(1)
        for i in range(0, self.player_ct - 1):
            self.prob_matrices[i-1].update_probs_ncard(n_card[0].copy())
        if self.play_status[0]:
            round_count += 1
        self.postflop_play()
        if self.hand_ct == 1:
            self.round_in_play += 1
            self.round_in_play += 1
            return

        n_card_2 = self.flop(1)
        for i in range(0, 0, self.player_ct-1-1):
            self.prob_matrices[i-1-1].update_probs_ncard(n_card_2[0].copy())
        if self.play_status[0]:
            round_count += 1
        self.postflop_play()
        if self.hand_ct == 1:
            self.round_in_play += 1
            self.round_in_play += 1
            return
        print("status ", str(self.play_status))
        
        d_avg_aggregate, winner = self.game_end(round_count)

        final_bot_stack = self.player_stacks[0]
        
        if d_avg_aggregate != 0:
            if d_avg_aggregate < 0 and winner == 0:
                self.efficiency = self.efficiency - 2 * ((final_bot_stack - initial_bot_stack)/self.big_blind)/d_avg_aggregate
            elif d_avg_aggregate > 0 and winner == 0:
                self.efficiency = self.efficiency + ((final_bot_stack - initial_bot_stack)/self.big_blind)/d_avg_aggregate
            elif d_avg_aggregate < 0 and winner != 0:
                self.efficiency = self.efficiency - ((final_bot_stack - initial_bot_stack)/self.big_blind)/d_avg_aggregate
            elif d_avg_aggregate > 0 and winner != 0:
                self.efficiency = self.efficiency + 2 * ((final_bot_stack - initial_bot_stack)/self.big_blind)/d_avg_aggregate
            else:
                self.efficiency = 0

        print("EFFICIENCY OF BOT IS: ", self.efficiency)
        self.efficiency_list.append(self.efficiency)
        self.bot_PnL_list.append(final_bot_stack - initial_bot_stack)
        print("Efficiency list currently is: ", self.efficiency_list)
        print("PnL list currently is: ", self.bot_PnL_list)


first_game = GameEngine(1000, 6, 5, 10)
# print(first_game.com_cards)
# print(first_game.player_stacks)
# print(first_game.player_cards)
# print(first_game.seen)
# print("dealer", str(first_game.dealer))
# first_game.play()
# print("*******************game 2*************")
# first_game.play()


for i in range(4):
    print("game, " ,str(i))
    first_game.play()