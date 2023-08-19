"""
Measures the performance of several models against the baseline AI
for different training times and display a visualization of it.
"""

import itertools
import os
from time import time

import pandas as pd
import matplotlib.pyplot as plt

from random_ai import Random_AI
from game_system import Game_system
from graphics import Graphics
from original_ai import Original_AI
from q_learning import QLearningAI

def plot_results(results: pd.DataFrame):
    plt.plot("training_time", "score", data=results)
    plt.title("Score over training time")
    plt.xlabel("Training time (sec)")
    plt.ylabel("Score (wins minus losses)")
    plt.show()

def plot_all_results():
    """
    Plots all results in the training metrics folder side-by-side.
    """
    filenames = os.listdir("training_metrics")
    rows_count = len(filenames)//5 + 1
    columns_count = 5

    fig, axes = plt.subplots(rows_count, columns_count)

    data_index = 0
    for ax, filename in zip(axes.flat, filenames):
        ax.set(xlabel="Training time (sec)", ylabel="Score (wins minus losses)", title=filename)
        df = pd.read_csv(f"training_metrics/{filename}")
        ax.plot("training_time", "score", data=df)
        data_index += 1
    plt.show()


def train_model(model="original", games_count=1000, **hyperparameters):

    if model == "original":
        player_1 = Original_AI()
        player_2 = Original_AI()
    elif model == "q-learning":
        player_1 = QLearningAI(**hyperparameters)
        player_2 = QLearningAI(**hyperparameters)


    player_3 = Random_AI()

    no_display = True
    if no_display:
        training_game_system = Game_system(player_1, player_2)
        test_game_system = Game_system(player_1, player_3)

    else:
        graphics = Graphics()
        training_game_system = Game_system(player_1, player_2, graphics)
        test_game_system = Game_system(player_1, player_3, graphics)


    results = {"wins":[],
                "draws": [],
                "losses": [],
                "score": [],
                "training_time": []} # A dictionary that stores the training results

    # Deleting previous training data
    with open("training.json", 'w') as _:
        pass

    games_played = 0
    starting_time = time()
    total_time = 0

    # This is the main loop. It will keep playing new games until you close the
    # game window. The tuple inside brackets is the shape of the board.
    # In order to train the model, we keep following this cycle:
    # - first the AI plays 500 training games against itself and updates its
    # training data accordingly
    # - Then it plays 100 games against the baseline AI. The score obtained
    # over these 100 games is then saved.
    while training_game_system.play_a_game((3, 3)) and games_played < games_count:
        games_played += 1
        if games_played % 10 == 0:
            print(f"Training games played: {games_played}")
        if games_played % 200 == 0:
            player_1.save_training_data()
            player_2.save_training_data()
            # Updating the training time
            total_time = total_time + (time() - starting_time)
            results["training_time"].append(total_time)

            # Playing 100 games against the baseline model
            # During the tests, exploration is disabled so the AI always prioritizes
            # the best known move.
            player_1.learning = False
            test_game_system.reset_games_counter()
            print("Testing...")
            for i in range(100):
                test_game_system.play_a_game((3, 3))
            player_1.learning = True

            results["wins"].append(test_game_system.player_1_scores["WINS"])
            results["losses"].append(test_game_system.player_1_scores["LOSSES"])
            results["draws"].append(test_game_system.player_1_scores["DRAWS"])
            results["score"].append(test_game_system.player_1_scores["WINS"] - test_game_system.player_1_scores["LOSSES"])
            test_game_system.reset_games_counter()
            starting_time = time()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"training_metrics/{player_1}_results.csv", index=False)

def finetune_q_learning():
    """
    Grid-search function that looks for optimal hyperparameters
    for Q-learning.
    """

    alpha_values = (0.05,)
    gamma_values = (0.5, 0.8)
    epsilon_values = (0.2, 0.6)

    for alpha, gamma, epsilon in itertools.product(alpha_values, gamma_values, epsilon_values):
        print("Trying with the following hyperparameters:")
        print(f"alpha = {alpha}")
        print(f"gamma = {gamma}")
        print(f"epsilon = {epsilon}")
        train_model(model="q-learning", games_count=1401, alpha=alpha, gamma=gamma, epsilon=epsilon)
        os.remove("training_data/q_learning.pkl")



#finetune_q_learning()
plot_all_results()
"""results_df = pd.read_csv("training_metrics/q_learning_results.csv")
plot_results(results_df)"""
