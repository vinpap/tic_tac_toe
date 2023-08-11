"""
Measures the performance of the original AI against the baseline AI
for different training times and display a visualization of it.
"""

from time import time

import pandas as pd
import matplotlib.pyplot as plt

from ai import AI
from random_ai import Random_AI
from game_system import Game_system
from human_player import Human_player
from graphics import Graphics

def plot_results(results: pd.DataFrame):
    plt.plot("training_time", "score", data=results)
    plt.title("Score over training time")
    plt.xlabel("Training time (sec)")
    plt.ylabel("Score (wins minus losses)")
    plt.show()

def train_model():

    player_1 = AI()
    player_2 = AI()
    player_3 = Random_AI()

    no_display = False
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

    games_count = 10000
    games_played = 0
    starting_time = time()
    total_time = 0

    # This is the main loop. It will keep starting new games until you close the
    # game window. The tuple inside brackets is the shape of the board
    while training_game_system.play_a_game((3, 3)) and games_played < games_count:
        games_played += 1
        if games_played % 500 == 0:
            print(f"Training games played: {games_played}")
            # Every 500 games, we play 100 test games while setting the exploration rate to
            # 0 in order to maximize the chances of success
            total_time = total_time + (time() - starting_time)
            results["training_time"].append(total_time)
            player_1.exploration_rate = 0
            test_game_system.reset_games_counter()
            for i in range(100):
                test_game_system.play_a_game((3, 3))
            player_1.exploration_rate = 0.5 # Resetting the exploration rate to its previous value

            results["wins"].append(test_game_system.player_1_scores["WINS"])
            results["losses"].append(test_game_system.player_1_scores["LOSSES"])
            results["draws"].append(test_game_system.player_1_scores["DRAWS"])
            results["score"].append(test_game_system.player_1_scores["WINS"] - test_game_system.player_1_scores["LOSSES"])
            test_game_system.reset_games_counter()
            starting_time = time()

    results_df = pd.DataFrame(results)
    results_df.to_csv("original_model_results.csv")


#train_model()
results_df = pd.read_csv("original_model_results.csv")
plot_results(results_df)
