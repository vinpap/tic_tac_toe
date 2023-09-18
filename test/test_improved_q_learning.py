"""
Contains the unit tests for the improved Q-learning, 
which is a hand-coded model-free reinforcement model.
"""
import os

import pytest
import numpy as np

from improved_q_learning import Improved_q_learning

@pytest.fixture
def test_ai():
    """
    Returns an instance of the AI to use for testing.
    """
    return Improved_q_learning()

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



def test_should_explore(test_ai):
    """
    Tests the 'should_explore' method.
    """

    # We pick 0.5 as a state value, i.e. an average move.
    state_value = 0.5
    
    # Making sure the AI never decides to explore when its exploration
    # rate is 0.
    test_ai.exploration_rate = 0
    for i in range(100):
        assert not test_ai.should_explore(state_value)
    
    # Checking that the AI will always explore if the state_value is 0,
    # i.e. if the best possible move has always led to defeat in the past,
    # and the exploration rate is 1
    test_ai.exploration_rate = 1
    state_value = 0
    for i in range(100):
        assert test_ai.should_explore(state_value)


def test_load_training_data(test_ai):
    """
    Tests the 'load_training_data' method.
    """
    if os.path.exists("training.json"):
        # Checking the format of the training data.
        training_data = test_ai.load_training_data()

        assert isinstance(training_data, list)
        if training_data != []:
            state = training_data[-1]
            assert isinstance(state, dict)
            assert set(("array", "value", "occurences")) <= state.keys()
            assert type(state["array"]) == np.ndarray
            assert isinstance(state["value"], float) and 0 <= state["value"] <= 1
            assert isinstance(state["occurences"], int) and state["occurences"] > 0


    else:
        # Making sure an empty list is returned if the training file 
        # was not found.
        assert test_ai.load_training_data() == []


def test_update_training_data(test_ai):
    """
    Tests the 'update_training_data' method.
    """

    state = np.array([[1, 1, -1],
                    [0, -1, 1],
                    [0, 0, -1]])

    dummy_training_data = [
        {
            "array": state,
            "value": 0.5,
            "occurences": 2
        }
    ]

    test_ai.training_data = dummy_training_data
    test_ai.update_training_data(state, 0.2)

    # Making sure that state value and occurences count have been properly updated
    assert test_ai.training_data[0]["occurences"] == 3
    assert round(test_ai.training_data[0]["value"], 1) == 0.4

def test_write_training_data(test_ai):
    """
    Tests the 'write_training_data' method.
    """

    state = np.array([[1, 1, -1],
                    [0, -1, 1],
                    [0, 0, -1]])

    dummy_training_data = [
        {
            "array": state,
            "value": 0.5,
            "occurences": 2
        }
    ]
    test_ai.training_data = dummy_training_data
    test_ai.write_training_data()

    # Making sure that the training data have not been altered after 
    # saving them in the JSON ant opening them again.
    test_ai.load_training_data()

    assert dummy_training_data == test_ai.training_data

