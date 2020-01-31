'''MoneyPit solve module
'''

from __future__ import print_function

import multiprocessing
import os
import pickle
import neat

import game

GAMES_PER_EVAL = 5
PLAYERS_PER_GAME = 3


def eval_genome(genome, config) -> float:
    '''This function will be run in parallel by ParallelEvaluator. It is used
    to evaluate a genomes fitness.

    Args:
        genome: A single genome
        config: The genome class configuration data

    Returns:
        float: That genome's fitness
    '''

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    def genome_player_function(card, game_chips, chips, game_cards):

        inputs = []
        inputs += [game_chips / (PLAYERS_PER_GAME * 11)]
        inputs += [chips / (PLAYERS_PER_GAME * 11)]
        inputs += cards_to_binlist(card)
        inputs += cards_to_binlist(game_cards[PLAYERS_PER_GAME - 1])

        return net.activate(inputs)[0] > 0.5

    player_methods = (
        [genome_player_function] +
        [lambda card, game_chips, chips, _: False] * (PLAYERS_PER_GAME - 1)
    )

    fitness = []

    for _ in range(GAMES_PER_EVAL):
        session = game.Game(player_methods)

        while 1:
            step = session.step()
            if step[0]:
                fitness.append(step[1][0][2])
                break

    return min(fitness)
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

    # Run for up to 300 generations.
    pop_eval = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pop_eval.evaluate, 300)

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
    run(CONFIG_PATH)


def cards_to_binlist(card) -> list:
    '''Converts card number or list of card numbers to binary list

    Args:
        card (int/list): Card number or list of card numbers

    Returns:
        list: binary list (of length 33) with 1 if card and 0 if not card
    '''
    binlist = [0] * 33

    if isinstance(card, int):
        binlist[card - 3] = 1

    else:
        for crd in card:
            binlist[crd - 3] = 1

    return binlist
