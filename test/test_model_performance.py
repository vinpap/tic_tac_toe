"""Contains tests designed to make sure that our new model is better than the 
original one.
In order to do that, we train both models from scratch during the same amount of time, 
then we have both of them play 1000 games against the baseline AI and compare their results."""

import os
import time
from copy import deepcopy

import pytest

from improved_q_learning import Improved_q_learning
from q_learning import QLearningAI
from game_system import Game_system
from random_ai import Random_AI


@pytest.fixture
def new_model():
    """
    Returns a trained instance of our new model.
    """

    # Training time in seconds
    training_time = 500

    new_model = Improved_q_learning()
    opponent = Improved_q_learning()

    training_game_system = Game_system(new_model, opponent)

    # First we delete any existing Q-table to ensure we really train the model from scratch
    if os.path.exists("training.json"):
        os.remove("training.json")
    
    starting_time = time.time()

    max_time_reached = False
    games_played = 0

    while not max_time_reached and training_game_system.play_a_game((3, 3)):
        games_played += 1
        if games_played % 100 == 0:
            print(f"New model - {games_played} games played")
        if time.time() - starting_time > training_time:
            max_time_reached = True
        
    return new_model

@pytest.fixture
def original_model():
    """
    Returns a trained instance of the original Q-learning model.
    """

    # Training time in seconds
    training_time = 500

    original_model = QLearningAI()
    opponent = QLearningAI()

    training_game_system = Game_system(original_model, opponent)

    # First we delete any existing Q-table to ensure we really train the model from scratch
    if os.path.exists("training_data/q_learning.pkl"):
        os.remove("training_data/q_learning.pkl")
    
    starting_time = time.time()

    max_time_reached = False
    games_played = 0

    while not max_time_reached and training_game_system.play_a_game((3, 3)):
        games_played += 1
        if games_played % 100 == 0:
            print(f"Original model - {games_played} games played")
        if time.time() - starting_time > training_time:
            max_time_reached = True
        
    return original_model

def test_model_performance(new_model, original_model):
    """
    Makes sure that:
    - the new model's score is higher than 0, meaning it wins more than it loses
    - the new model's score is higher than the original model's.
    """

    new_model.explore = False
    original_model.learning = False
    test_games_count = 200
    
    # Computing new model's score
    game_system_1 = Game_system(new_model, Random_AI())
    
    for i in range(test_games_count):
        game_system_1.play_a_game((3, 3))

    wins = game_system_1.player_1_scores["WINS"]
    losses = game_system_1.player_1_scores["LOSSES"]
    new_model_score = wins - losses
    assert new_model_score >= 0


    game_system_2 = Game_system(original_model, Random_AI())
    for i in range(test_games_count):
        game_system_2.play_a_game((3, 3))

    wins = game_system_2.player_1_scores["WINS"]
    losses = game_system_2.player_1_scores["LOSSES"]
    original_model_score = wins - losses
    print(f"Original model_score: {original_model_score}")
    print(f"New model score: {new_model_score}")

    assert new_model_score > original_model_score


