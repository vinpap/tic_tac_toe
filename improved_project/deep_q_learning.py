"""
Implementation of the deep Q-Learning model for tic-tac-toe.
"""
import random

import numpy as np
import keras

from interfaces import Player_interface

class DeepQLearningAI(Player_interface):

    dnn = None

    def __init__(self, alpha=0.001, gamma=0.5, epsilon=0.5):
        self.is_AI = True
        self.learning = True
        self.alpha = alpha # Learning rate
        self.epsilon = epsilon # Exploration rate epsilon (epsilon greedy algo)
        self.gamma = gamma # discount rate

        self.games_played = 0 # Count of games played since the game was launched
        self.batch_size = 100 # Defines how often we should retrain the network (in games count)

        self.replayed_memory_buffer = [] # Memory buffer used during training
        self.game_history = []

        if not DeepQLearningAI.dnn:
            try:
                # Try to load a previously saved model
                DeepQLearningAI.dnn = keras.models.load_model('training_data/deep_q_learning.keras')
            except OSError:
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

        # First, compute a random number between 0 and 1 and compare it to epsilon
        # to decide if we should exploit or explore.
        # Moreover, the AI never explores if its 'learning' attribute is set to False.
        if random.random() < self.epsilon and self.learning:
            # Explore
            # Get a list of all possible moves and pick one randomly
            move = self.get_random_move(current_state)
        else: 
            # Exploit
            # Play the best known move according to the model,
            # after filtering out any illegal move.
            current_state_vector = current_state.flatten()
            q_values = DeepQLearningAI.dnn.predict(np.array([current_state_vector]))[0]
            ordered_q_values = np.argsort(q_values)
            for index in ordered_q_values:
                if self.move_is_valid(current_state, index):
                    move = current_state_vector
                    move[index] = 1
                    move = np.reshape(move, (3, 3))
                    break
                    


        current_state_vector = current_state.flatten()
        new_state_vector = move.flatten()
        action = np.flatnonzero(current_state_vector != new_state_vector)[0]
        move_tuple = (current_state_vector, action, new_state_vector) 
        self.game_history.append(move_tuple)
        return move

    def get_all_possible_moves(self, current_board):
        """
        Returns all possible moves given the current state.
        """
        # Appending arrays corresponding to every possible move in a list called
        # candidate_moves
        candidate_moves = []
        values = []
        for index, value in np.ndenumerate(current_board):
            if value == 0:
                possible_move = current_board.copy()
                possible_move[index] = 1
                candidate_moves.append(possible_move)
        
        
        return candidate_moves

    def get_random_move(self, current_board):
        """
        Returns a random move among all moves that are available.
        """
        possible_moves = self.get_all_possible_moves(current_board)
        random_index = random.randint(0, len(possible_moves)-1)
        return possible_moves[random_index]
    
    def setup_neural_network(self):
        """
        Sets up a neural network and returns it.
        """
        init = keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=9, activation="relu", kernel_initializer=init))
        model.add(keras.layers.Dense(24, activation="relu", kernel_initializer=init))
        model.add(keras.layers.Dense(9, activation="linear", kernel_initializer=init))
        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=self.alpha))
        return model
    
    def train_model(self):
        """
        Trains the model using the replayed memory buffer as training data.
        """

        DeepQLearningAI.dnn.save("training_data/deep_q_learning.keras")
    
    def move_is_valid(self, current_state: np.array, move: int):
        """
        Returns True if the move passed as a parameter is valid given the
        current state, False otherwise.

        current_state: current board given as a 2D numpy array
        move: index of the box to play on the flattened board vector.
        """
        if current_state.flatten()[move] == 0:
            return True
        return False

    def notify_game_result(self, result) -> None:
        """
        Overrided from Player_interface.
        """
        # Here we loop starting from the last state
        history_from_end = list(reversed(self.game_history))
        for state_index in range(len(history_from_end)):
            game_move =  history_from_end[state_index]    
            reward = result * (self.gamma**state_index)
            tuple_to_append = (game_move[0],
                               game_move[1],
                               game_move[2],
                               reward)       
            self.replayed_memory_buffer.append(tuple_to_append)    
        self.game_history = []

        self.games_played += 1
        if self.games_played % self.batch_size == 0:
            print("Training...")
            self.train_model()
    


    