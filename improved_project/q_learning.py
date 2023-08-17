"""
Implementation of the Q-Learning model for tic-tac-toe.
"""
import pickle
import itertools
import random

import numpy as np

from interfaces import Player_interface

class QLearningAI(Player_interface):

    q_table = [] # The Q-table contains the values associated to each
    # state. It is a 2D list of shape [[np.array...], [value...]].
    # The q-table is a static attribute as it needs to be shared
    # between all instances of QLearningAI models.

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.is_AI = True
        self.history = [] # List of all successive moves in the current game
        # Used at the end of each game to update the q-table
        self.explore = True # The AI will never explore if this is set to False
        # (used in test games)

        # Hyperparameter definitions:
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount rate
        self.epsilon = epsilon # 'greediness rate' that defines whether the model exploits or explores
        # The higher the rate, the more likely the model is to explore, i.e. playing a random move.


        if QLearningAI.q_table == []:
            # Try to load from a pkl file if list is empty
            try:
                with open("training_data/q_learning.pkl", "rb") as training_data:
                    QLearningAI.q_table = pickle.load(training_data)
            except FileNotFoundError:
                QLearningAI.q_table = self.generate_initial_q_table()
                self.save_training_data()
    
    def __str__(self):

        return f"q_learning_alpha{self.alpha}_gamma{self.gamma}_epsilon{self.epsilon}"


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
        # Moreover, the AI never explores if its 'explore' attribute is set to False.
        if random.random() < self.epsilon and self.explore:
            # Explore
            # Get a list of all possible moves and pick one randomly
            move = self.get_random_move(current_state)
        else: 
            # Exploit
            # Play the best known move
            move = self.get_best_move(current_state)

        self.history.append(move)
        return move

    
    def notify_game_result(self, result) -> None:
        """
        Overrided from Player_interface.
        """
        self.update_q_table(result)
        self.save_training_data()
        self.history = []
    
    def update_q_table(self, reward):
        """
        Updates the values associated with each state based on the reward obtained.

        Called at the end of a game, this function updates the values associated
        with each move played during this game, starting from the last one and 
        going back from there.
        """

        for state_index in reversed(range(len((self.history)))):
            for index in range(len(QLearningAI.q_table[0])):
                if np.array_equal(self.history[state_index], QLearningAI.q_table[0][index]):
                    current_value = QLearningAI.q_table[1][index]
                    delta_t = len(self.history) - (state_index+1)
                    new_value = (
                        (1-self.alpha) * current_value +
                        self.alpha * (reward * self.gamma**delta_t)
                        )
                    


    
    def get_all_possible_moves(self, current_board):
        """
        Returns all possible moves given the current state, along with their associated values.
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
        
        # Retrieving the values associated with each possible move
        for move in candidate_moves:
            for index in range(len(QLearningAI.q_table[0])):
                if np.array_equal(move, QLearningAI.q_table[0][index]):
                    values.append(QLearningAI.q_table[1][index])
                    break
        
        return [candidate_moves, values]

    
    def get_best_move(self, current_board):
        """
        Returns the best move in the q-table considering the current state of the
        board.
        """
        possible_moves = self.get_all_possible_moves(current_board)
        best_move_index = possible_moves[1].index(max(possible_moves[1]))
        return possible_moves[0][best_move_index]
    
    def get_random_move(self, current_board):
        """
        Returns a random move among all moves that are available.
        """
        possible_moves = self.get_all_possible_moves(current_board)
        random_index = random.randint(0, len(possible_moves[0])-1)
        return possible_moves[0][random_index]

    def check_for_endgame(self, board) -> int:
        """
        Checks if a state is terminal, i.e. if it depicts a loss, victory or draw.

        Returns an integer:
        -1 if it is a loss
        0 if it is a draw
        1 if it is a victory
        Returns 2 if the game is still going.
        """

        no_space_left = True
        testing_x = testing_y = range(-1, 2)

        for index, value in np.ndenumerate(board):
            if value == 0:
                no_space_left = False

            elif (value == -1) or (value == 1):
                for i in testing_x:
                    for j in testing_y:
                        if i == 0 and j == 0:
                            continue
                        try:
                            if (
                                board[index[0] + i, index[1] + j] == value
                            ) and (
                                board[index[0] + 2 * i, index[1] + 2 * j]
                                == value
                            ):
                                if (
                                    index[0] + i >= 0
                                    and index[0] + 2 * i >= 0
                                    and index[1] + j >= 0
                                    and index[1] + 2 * j >= 0
                                ):
                                    return value
                        except IndexError:
                            continue

        if no_space_left:
            return 0
        return 2
    
    def generate_initial_q_table(self):
        """
        Generates a Q-table where each state is assigned a random value.

        Called when no q-table is available. Note that the terminal states
        (winning or losing positions) are set to 0 or -1.
        """

        # Generating all combinations of 3x3 boards first
        boards = [np.reshape(np.array(i), (3, 3)) for i in itertools.product([-1, 0, 1], repeat = 3*3)]

        # Filtering all the combinations by removing the illegal ones
        valid_states = []
        values = []
        for board in boards:
            # excluding all boards with an impossible number of moves by either player
            if np.sum(board) in (0, 1):
                valid_states.append(board)

                # if the board depicts a terminal state, we give it a predefined value:
                # 0 if it is a loss, 1 if it is a victory, 0.5 if it is a draw.
                # any other state is given a random value between 0 and 1.
                board_status = self.check_for_endgame(board)
                if board_status == -1:
                    values.append(0)
                elif board_status == 0:
                    values.append(0.5)
                elif board_status == 1:
                    values.append(1)
                else:
                    values.append(random.random())

        return [valid_states, values]

    def save_training_data(self):
        """
        Saves the Q table in a pickle file.
        """
        with open("training_data/q_learning.pkl", "wb") as training_data:
            pickle.dump(QLearningAI.q_table, training_data)



    