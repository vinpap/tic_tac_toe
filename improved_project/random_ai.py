"""
A baseline AI that plays randomly each turn.
"""

import numpy as np

from interfaces import Player_interface

class Random_AI(Player_interface):

    def __init__(self):
        self.is_AI = True

    def play(self, current_state: np.array) -> np.array:
        """
        Selects the based move given the state of the board given by current_state.

        given_state: 2D numpy array that contains the current state of the board.
        Each box on the board is given a value:
        1: this player already checked this box
        -1: the opponent checked this box
        0: no one has checked this box yet.

        Returns the new state of the board after the AI played as a numpy array.  
        """
        
        empty_boxes = np.where(current_state == 0)
        ind = np.random.randint(low=0, high=len(empty_boxes[0]))
        current_state[empty_boxes[0][ind], empty_boxes[1][ind]] = 1
        return current_state


    def notify_game_result(self, result) -> None:
        return

