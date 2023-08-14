"""
Contains the unit tests for the random AI, i.e. the AI used as a baseline.
It is an AI that plays each move randomly.
"""
import pytest
import numpy as np

from random_ai import Random_AI

@pytest.fixture
def test_ai():
    """
    Returns an instance of the random AI to use for testing.
    """
    return Random_AI()

def test_play(test_ai):
    """
    Tests the 'play' method of the class.
    """

    initial_state = np.array([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]])
    
    new_state = test_ai.play(initial_state)

    # Checking that the AI played once and only once
    assert new_state.sum() == 1
    assert np.count_nonzero(new_state == 1) == 2
    assert np.count_nonzero(new_state == -1) == 1
    # Making sure that the values previously stored in the array were 
    # not changed
    assert new_state[0, 0] == -1
    assert new_state[1, 1] == 1
