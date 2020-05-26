'''MoneyPit solve module
'''

from __future__ import print_function

import os
import pickle
import multiprocessing

import neat

import game


THREADS = multiprocessing.cpu_count()

GENERATIONS = 1000
GAMES_PER_EVAL = 20
PLAYERS_PER_GAME = 4
RUN_MULTITHREADED = True


def genome_player_function(card, game_chips, chips, game_cards, net):
    '''Funktion that uses the neural network to let the genome decide, wether
    to take the card ord not.

    Args:
        card (int): Card to decide on
        game_chips (int): Number of chips currently "on the card"
        chips (int): Number of chips on players posession
        game_cards (tuple): Tuple of tuple of all players cards, ending with
            active player
        net: Neural network for decision making

    Returns:
        bool: True if player wants to take card
    '''

    def score_change(cards, card, chips) -> int:
        '''Calculate and return score from cards

        Returns:
            int: current player score
        '''

        if len(cards) == 0:
            return -card + chips

        cards = list(cards)

        score_without = 0

        cards.sort(reverse=True)
        while len(cards) >= 2:
            if cards[0] - 1 != cards[1]:
                score_without -= cards[0]
            cards.pop(0)
        score_without -= cards[0]

        score_with = 0

        cards.append(card)

        cards.sort(reverse=True)
        while len(cards) >= 2:
            if cards[0] - 1 != cards[1]:
                score_with -= cards[0]
            cards.pop(0)
        score_with -= cards[0]

        return score_with - score_without + chips

    inputs = []
    inputs += [game_chips / (PLAYERS_PER_GAME * 11)]
    inputs += [chips / (PLAYERS_PER_GAME * 11)]

    for _, cards in game_cards.items():
        score = score_change(cards, card, game_chips)
        inputs += [score / 35]

    output = net.activate(inputs)
    return output[0] < output[1]


def eval_genomes(genomes, config):
    '''Helper function to run eval genome in sigle threaded mode

    Args:
        genomes : List of genomes
        config : he genome class configuration data
    '''
    for _, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def eval_genome(genome, config) -> float:
    '''This function will be run in parallel by ParallelEvaluator. It is used
    to evaluate a genomes fitness.

    Args:
        genome: A single genome
        config: The genome class configuration data

    Returns:
        float: That genome's fitness
    '''

    def genome_player_function_helper(card, game_chips, chips, game_cards):
        return genome_player_function(card, game_chips, chips, game_cards, net)
        # return False

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    player_methods = (
        [genome_player_function_helper] +
        [lambda card, game_chips, chips, _: False] * (PLAYERS_PER_GAME - 1)
    )

    fitness = []

    for _ in range(GAMES_PER_EVAL):
        session = game.Game(player_methods)

        while 1:
            step = session.step()
            if step[0]:
                fit = 1
                for i in range(1, PLAYERS_PER_GAME):
                    if step[1][i][2] >= step[1][0][2]:
                        fit = 0
                fitness.append(fit)
                break

    return sum(fitness)
    # return sum(fitness) / GAMES_PER_EVAL


def run(config_file):
    '''Run neat using config_file

    Args:
        config_file (os.path): Path to config file
    '''
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    pop = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    if RUN_MULTITHREADED:
        # Run for up to 300 generations.
        pop_eval = neat.ParallelEvaluator(THREADS, eval_genome)
        winner = pop.run(pop_eval.evaluate, GENERATIONS)
    else:
        # Run for up to 300 generations.
        winner = pop.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the winner.
    with open('winner-feedforward', 'wb') as winner_file:
        pickle.dump(winner, winner_file)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    LOCAL_DIR = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(LOCAL_DIR, 'config-feedforward.ini')
    # CONFIG_PATH = os.path.join(LOCAL_DIR, 'config-feedforward.ini')
    run(CONFIG_PATH)


# def cards_to_binlist(card) -> list:
#         '''Converts card number or list of card numbers to binary list

#         Args:
#             card (int/list): Card number or list of card numbers

#         Returns:
#             list: binary list (of length 33) with 1 if card and 0 if not
#         '''
#         binlist = [0] * 33

#         if isinstance(card, int):
#             binlist[card - 3] = 1

#         else:
#             for crd in card:
#                 binlist[crd - 3] = 1

#         return binlist
