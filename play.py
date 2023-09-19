"""This is the main script. Launch it to try this project!
Read the comments below to understand how the different parameters work"""
from time import time

from improved_q_learning import Improved_q_learning
from testing_ai import Testing_AI
from random_ai import Random_AI
from game_system import Game_system
from human_player import Human_player
from graphics import Graphics

# If this value is set to False, you play against the AI
# If it is set to True, the AI plays agains itself. In that case, the data from
# both players is used to train the model.
both_players_are_AI = False
test_ai = True

# Only usable if both_players_are_AI is set to True
no_display = True
if no_display and not both_players_are_AI:
    no_display = False

if both_players_are_AI:
    player_1 = Improved_q_learning()
    if test_ai:
        player_2 = Random_AI()
    else:
        player_2 = Improved_q_learning()
else:
    player_1 = Human_player()
    if test_ai:
        player_2 = Random_AI()
    else:  
        player_2 = Improved_q_learning()

# graphics handles everything related to the display of the game
# game_system sets the rules and oversees the game

if no_display:
    game_system = Game_system(player_1, player_2)

else:
    graphics = Graphics()
    game_system = Game_system(player_1, player_2, graphics)

# This is the main loop. It will keep starting new games until you close the
# game window. The tuple inside brackets is the shape of the board
games_count = 2000
games_played = 0
t1 = time()
while game_system.play_a_game((3, 3)) and games_played < games_count-1:
    games_played += 1
print(f"training time: {time()-t1} s")
