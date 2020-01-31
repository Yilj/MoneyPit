'''MoneyPit player module
'''


class Player():
    '''Player class
    '''
    def __init__(self, name: int, player_method):
        assert callable(player_method), "player_method has to be a function"

        self.__name = name  # Player id
        self.__method = player_method  # Players decision method
        self.__chips = 11  # Number of chips the player has
        self.__cards = []  # Cards a player has

    @property
    def name(self) -> int:
        '''Get player name

        Returns:
            int: Player name
        '''
        return self.__name

    @property
    def chips(self) -> int:
        '''Get player chips

        Returns:
            int: Amount of chips the player has
        '''
        return self.__chips

    @property
    def cards(self) -> list:
        '''Get player cards

        Returns:
            []: List of player cards
        '''
        return tuple(self.__cards)

    @property
    def score(self) -> int:
        '''Calculate and return current player score

        Returns:
            int: current player score
        '''
        cards = self.__cards.copy()
        score = 0

        if len(cards) > 0:
            cards.sort(reverse=True)
            while len(cards) >= 2:
                if cards[0] - 1 != cards[1]:
                    score -= cards[0]
                cards.pop(0)
            score -= cards[0]

        score += self.__chips
        return score

    def decide(self, game_card: int, game_chips: int, game_cards) -> bool:
        '''Run decision method and return true if player takes card

        Args:
            card (int): Active card to decide on
            game_chips (int): Current number of chips on card
            game_cards (tuple): Tuple of tuple of all players cards ending with
                                active player

        Returns:
            bool: Returns true if player takes card
        '''
        player_chips = self.__chips
        if player_chips == 0:
            decision = True
        else:
            decision = self.__method(game_card,
                                     game_chips,
                                     player_chips,
                                     game_cards)
        if decision:
            self.__chips += game_chips
            self.__cards = sorted(self.__cards + [game_card])
        else:
            self.__chips -= 1
        return decision
