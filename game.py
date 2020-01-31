'''MoneyPit game module
'''

import random
import player


class Game():
    '''Game class containing all game relevant objects
    '''
    def __init__(self, player_methods: tuple):
        self.__deck = create_deck()

        assert len(player_methods) >= 3, "At least 3 players needed"
        self.__players = []
        for i, method in enumerate(player_methods):
            self.__players += [player.Player(i, method)]

        self.__game_cards = self.__get_game_cards()

        self.__next_card()
        self.__next_player()

    def __next_card(self):
        '''Gets next card from deck and sets it as current_card
        '''
        self.__active_card = self.__deck.pop()
        self.__active_chips = 0

    def __next_player(self):
        '''Gets next player and sets it as current player, then rotates
        player list
        '''
        self.__active_player = self.__players[0]
        self.__players += [self.__players.pop(0)]

    def __get_game_cards(self) -> tuple:
        '''Returns tuple of tuple of every players cards

        Returns:
            tuple: Every players cards ending with active player
        '''
        game_cards = {}
        for plr in self.__players:
            game_cards[plr.name] = plr.cards
        return game_cards

    def __get_game_state(self):
        game_state = {}
        for plr in self.__players:
            game_state[plr.name] = (0, 0, plr.score)
        return game_state

    def step(self) -> tuple:
        '''Step game by asking active player to decide on current state

        Returns:
            tuple: Returns game state and
        '''
        # print("Player: " + str(self.__active_player.name), #DEBUG
        #       "Card: " + str(self.__active_card), #DEBUG
        #       "Chips: " + str(self.__active_chips)) #DEBUG

        if self.__active_player.decide(self.__active_card, self.__active_chips,
                                       self.__game_cards):
            # print("Takes card") #DEBUG
            try:
                self.__next_card()
                self.__game_cards = self.__get_game_cards()
            except IndexError:
                # print("Game over!") #DEBUG
                return (True, self.__get_game_state())
        else:
            self.__active_chips += 1
            self.__next_player()
        return (False, None)


def create_deck() -> list:
    '''Creates a starting deck with 24 randomly shuffled numbers in the range
    from 3 to 35 (inclusive)

    Returns:
        []: starting deck
    '''
    deck = list(range(3, 36))
    random.shuffle(deck)
    deck = deck[:24]
    return deck
