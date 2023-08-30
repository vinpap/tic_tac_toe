"""
Implementation of the deep Q-Learning model for tic-tac-toe.
"""


import numpy as np
import keras

from interfaces import Player_interface

class DeepQLearningAI(Player_interface):

    dnn = None

    def __init__(self, alpha=0.001):
        self.is_AI = True
        self.learning = True
        self.alpha = alpha # Learning rate
        self.games_played = 0 # Count of games played since the game was launched

        if not DeepQLearningAI.dnn:
            try:
                # Try to load a previously savec model
                pass
            except FileNotFoundError:
                # If no file was found, we generate a new network
                DeepQLearningAI.dnn = self.setup_neural_network()
                


    
    def __str__(self):

        return f"deep_q_learning_alpha{self.alpha}"


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



        move = []
        return move
    
    def setup_neural_network(self):
        """
        Set ups a neural network and returns it.
        """
        init = keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_shape=9, activation="relu", kernel_initializer=init))
        model.add(keras.layers.Dense(12, activation="relu", kernel_initializer=init))
        # Ici, on prédit la valeur d'un état (d'où le fait qu'il n'y a qu'un neurone en sortie)
        model.add(keras.layers.Dense(1, activation="linear", kernel_initializer=init))
        model.compile(loss="mean_absolute_error", optimizer=keras.optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model
    
    def notify_game_result(self, result) -> None:
        """
        Overrided from Player_interface.
        """
        return
    


    