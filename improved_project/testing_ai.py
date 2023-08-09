"""
A rules-based AI used to evaluate the performance of a model by playing against it.

This AI is meant to be "perfect", i.e. it should play tic-tac-toe using 
optimal techniques to win.
"""

import numpy as np

from interfaces import Player_interface

class Testing_AI(Player_interface):

    def __init__(self):
        self.is_AI = True
        self.strategy = None

    def play(self, current_state):

        
        instant_win = self.__check_instant_wins(current_state)
        if type(instant_win) == np.array:
            return instant_win
        
        instant_loss = self.__check_instant_loss(current_state)
        if type(instant_loss) == np.array:
            return instant_loss
        
        # We find out how many moves have been made so far
        # (necessary to decide our move)
        moves_count = np.count_nonzero(current_state)
        # SI ON COMMENCE :
        # Jouer dans un coin
        if moves_count == 0:
            current_state[0, 0] = 1
            return current_state
        
        if moves_count in (1, 2) and not self.strategy:
            self.strategy = self.__pick_strategy(current_state, moves_count)
        print(self.strategy)
        move = self.__follow_strategy(current_state, moves_count)
        return move

        # Si l'adversaire joue au milieu :
        # Continuer à jouer. À chaque tour :
        # Faire un alignement de 3 si possible.
        # Si pas possible, bloquer un alignement de 3 de l'adversaire
        # Si pas possible, faire un alignement de 2 qui pourrait permettre une victoire au tour suivant
        # Si pas possible, faire un alignement de 2 quelconque

        # Sinon :
        # Jouer au centre

        # Si on peut finir la diagonale : 
        # FInir la diagonale, on a gagné
        # SInon :
        # Si l'adversaire va faire un alignement de 3, on le neutralise


    def __pick_strategy(self, board, turn):
        """
        Used on the testing AI's second move in order to pick an optimal
        strategy based on the current state of the board.
        """
        if (turn == 2 and
            board[1, 1] == -1 and
            (board[0, 0] == 1 or
             board[0, 2] == 1 or
             board[2, 0] == 1 or
             board[2, 2] == 1)):
            return "FIRST_opponent_played_center"
        
        if (turn == 2 and
            board[1, 1] != -1 and
            (board[0, 0] == 1 or
             board[0, 2] == 1 or
             board[2, 0] == 1 or
             board[2, 2] == 1)):
            return "FIRST_center_is_free"

        if (turn == 1 and
            board[1, 1] == -1):
            return "SECOND_opponent_played_center"

        elif (turn == 1 and 
            (board[0, 0] == -1 or
            board[2, 0] == -1 or
            board[0, 2] == -1 or
            board[2, 2] == -1)):
            return "SECOND_opponent_played_corner"
        else:
            return "SECOND_opponent_played_side"

    def __follow_strategy(self, board, turn_count):
        """
        Returns the move advised by the current strategy.
        """

        match self.strategy:
            case "FIRST_opponent_played_center":
                return self.__first_opponent_played_center(board, turn_count)
            case "FIRST_center_is_free":
                return self.__first_center_is_free(board, turn_count)
            case "SECOND_opponent_played_center":
                return self.__second_opponent_played_center(board, turn_count)
            case "SECOND_opponent_played_corner":
                return self.__second_opponent_played_corner(board, turn_count)
            case "SECOND_opponent_played_side":
                return self.__second_opponent_played_side(board, turn_count)
            case "block_opponent":
                return self.__block_opponent(board)
            case _:
                raise ValueError("Invalid strategy name for the testing AI")
            
    def __first_opponent_played_center(self, board, turn):
        """
        Covers the case where the testing AI started and the opponent reacted
        by playing in the center box.
        """
        match turn:
            case 2:
                board[2, 2] = 1
            case 4:
                if board[0, 2] == -1:
                    board[2, 0] = 1
                elif board[2, 0] == -1:
                    board[0, 2] = 1
            
                else:
                    self.strategy = "block_opponent"
                    return self.__block_opponent(board)
            case _:
                self.strategy = "block_opponent"
                return self.__block_opponent(board)


        return board
        
    
    def __first_center_is_free(self, board, turn):
        """
        Covers the case where the testing AI started and the opponent did not
        play in the center box in reaction.
        """
        match turn:
            case 2:
                if (board[1, 0] != -1 and
                    board[2, 0] != -1):
                    board[2, 0] = 1
                else:
                    board[0, 2] = 1
            case 4:
                if (board[1, 0] != -1 and
                    board[2, 0] == 1):
                    board[1, 0] = 1
                elif (board[0, 1] != -1 and
                    board[0, 2] == 1):
                    board[0, 1] = 1       
                else:
                    if board[0, 2] == 1:
                        if board[0, 1] == 0:
                            board[0, 1] = 1
                        elif (board[1, 0] != -1 and
                            board[2, 0] != -1):
                            board[2, 0] = 1
                        else:
                            if board[1, 0] == 0:
                                board[1, 0] = 1
                            board[2, 2] = 1
                    else:
                        if (board[0, 1] != -1 and
                            board[0, 2] != -1):
                            board[0, 2] = 1
                        else:
                            board[2, 2] = 1      

        return board
    
    def __second_opponent_played_center(self, board, turn):
        """
        Covers the case where the AI played in second and the opponent started
        with playing in the middle
        """

        if turn == 1:
            board[0, 0] = 1
        self.strategy = "block_opponent"

        return board
    
    def __second_opponent_played_corner(self, board, turn):
        """
        Covers the case where the AI played in second and the opponent started
        with playing in a corner
        """

        if turn == 1:
            board[1, 1] = 1
        elif turn == 3:
            # Checking if the opponent is about to fill a row
            # or a column
            row_sums = np.sum(board, axis=1)
            col_sums = np.sum(board, axis=0)
            if -2 in row_sums or -2 in col_sums:
                board = self.__block_opponent(board)
            else:
                if board[0, 1] != -1:
                    board[0, 1] = 1
                elif board[1, 0] != -1:
                    board[1, 0] = 1
                elif board[2, 1] != -1:
                    board[2, 1] = 1
                else:
                    board[1, 2] = 1

            self.strategy = "block_opponent"
        return board
    
    def __second_opponent_played_side(self, board, turn):
        
        if turn == 1:
            board[1, 1] = 1
        
        elif turn == 3:
            if ((board[1, 0] == -1 and board[1, 2] == -1) or
                (board[0, 1] == -1 and board[2, 1] == -1)):
                board[0, 0] = 1
            else:
                self.strategy = "block_opponent"
                board = self.__block_opponent(board)

        elif turn == 5:
            if board[2, 2] != -1:
                board[2, 2] = 1
            else:
                board[0, 2] = 1

        return board
    
    def __check_instant_wins(self, board):
        """
        Checks if there is a way to win immediately.
        """
        row_sums = np.sum(board, axis=1)
        col_sums = np.sum(board, axis=0)

        # Checking if we can win right now, i.e. if the opponent made a big mistake
        for row_idx, total in enumerate(row_sums):
            if total == 2:
                for col_idx, value in enumerate(board[row_idx, :]):
                    if value == 0:
                        board[row_idx, col_idx] = 1
                        return board

        # Same with colums
        for col_idx, total in enumerate(col_sums):
            if total == 2:
                for row_idx, value in enumerate(board[: ,col_idx]):
                    if value == 0:
                        board[row_idx, col_idx] = 1
                        return board

        # Checking diagonals
        if board[0, 0] + board[1, 1] + board[2, 2] == 2:
            for i in range(3):
                if board[i, i] == 0:
                    board[i, i] = 1
                    return board
        
        elif board[0, 2] + board[1, 1] + board[2, 0] == 2:
            for i in range(3):
                if board[i, 2-i] == 0:
                    board[i, 2-i] = 1
                    return board
        else: 
            return False
    
    def __check_instant_loss(self, board):

        row_sums = np.sum(board, axis=1)
        col_sums = np.sum(board, axis=0)
        # Checking if a row is about to be filled by the opponent, and blocking
        # him if so
        for row_idx, total in enumerate(row_sums):
            if total == -2:
                for col_idx, value in enumerate(board[row_idx, :]):
                    if value == 0:
                        board[row_idx, col_idx] = 1
                        return board

        # Same with columns
        for col_idx, total in enumerate(col_sums):
            if total == -2:
                for row_idx, value in enumerate(board[: ,col_idx]):
                    if value == 0:
                        board[row_idx, col_idx] = 1
                        return board

        # Checking diagonals
        if board[0, 0] + board[1, 1] + board[2, 2] == -2:
            for i in range(3):
                if board[i, i] == 0:
                    board[i, i] = 1
                    return board
        
        elif board[0, 2] + board[1, 1] + board[2, 0] == -2:
            for i in range(3):
                if board[i, 2-i] == 0:
                    board[i, 2-i] = 1
                    return board
        
        # If the opponent is not about to fill any line/column/diagonal,
        # the AI plays randomly (it cannot win anymore anyway)
        

        return False

    
    def __block_opponent(self, board):
        """
        Plays in a way that prevents the opponent from winning when it
        has become impossible for the AI itself to win (except if there 
        is a big mistake)
        """

        row_sums = np.sum(board, axis=1)
        col_sums = np.sum(board, axis=0)
        # Checking if a row is about to be filled by the opponent, and blocking
        # him if so
        for row_idx, total in enumerate(row_sums):
            if total == -2:
                for col_idx, value in enumerate(board[row_idx, :]):
                    if value == 0:
                        board[row_idx, col_idx] = 1
                        return board

        # Same with columns
        for col_idx, total in enumerate(col_sums):
            if total == -2:
                for row_idx, value in enumerate(board[: ,col_idx]):
                    if value == 0:
                        board[row_idx, col_idx] = 1
                        return board

        # Checking diagonals
        if board[0, 0] + board[1, 1] + board[2, 2] == -2:
            for i in range(3):
                if board[i, i] == 0:
                    board[i, i] = 1
                    return board
        
        elif board[0, 2] + board[1, 1] + board[2, 0] == -2:
            for i in range(3):
                if board[i, 2-i] == 0:
                    board[i, 2-i] = 1
                    return board
        
        # If the opponent is not about to fill any line/column/diagonal,
        # the AI plays randomly (it cannot win anymore anyway)
        
        else:
            print("CHECK")
            for idx_row, row in enumerate(board):
                for idx_col, value in enumerate(row):
                    print(value)
                    if value == 0:
                        board[idx_row, idx_col] = 1
                        return board
        return board

    def notify_game_result(self, result):
        # Emptying the list of winning moves (if any) for the next game
        self.next_winning_moves = []
        self.strategy = None

