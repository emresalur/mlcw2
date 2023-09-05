# mlLearningAgents.py

# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
from collections import defaultdict



class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        self.currentState = state
        self.pacmanPosition = state.getPacmanPosition()
        self.ghostPositions = state.getGhostPositions()
        self.foodGrid = state.getFood()
        self.capsules = state.getCapsules()
        

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # Empty dictionary of Q-values
        self.q_value = defaultdict(int)

        # Empty dictionary of counts
        self.count = defaultdict(int)

        # Previous state and action (for learning)
        self.lastState = None
        self.lastAction = None

        self.score = 0

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """

        return self.q_value[(state.currentState, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # Get legal actions pacman can take excluding STOP
        actions = [action for action in state.currentState.getLegalPacmanActions() if action != Directions.STOP]
        
        # Return action that outputs highest q value
        return max((self.getQValue(state, action) for action in actions), default=0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # Get the current Q-value
        currentQValue = self.getQValue(state, action)

        # Get the maximum Q-value
        maxNextQValue = self.maxQValue(nextState)

        # Q(s, a) = Q(s, a) + alpha * (reward + gamma * maxQ(s', a') - Q(s, a))
        newQValue = currentQValue + self.alpha * (reward + self.gamma * maxNextQValue - currentQValue)

        # Update the Q-value using the Q-learning update rule
        self.q_value[(state.currentState, action)] = newQValue


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.count[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.count[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # Get legal actions except stop
        legal = state.getLegalPacmanActions()
        legal.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state)

        # Set lastState to random value at beginning of game
        if self.lastState is None:

            self.lastAction = random.choice(legal)
            self.lastState = state
            return self.lastAction

        # Extract features from previous state
        old_stateFeatures = GameStateFeatures(self.lastState)

        # Compute the reward between the previous state and the current state
        reward = self.computeReward(self.lastState, state)
        # Update q values
        self.learn(old_stateFeatures, self.lastAction, reward, stateFeatures)

        # Choose a random action with probability epsilon for epsilon-greedy exploration otherwise pick best action
        if util.flipCoin(self.epsilon):
            return random.choice(legal)
        else:
            self.lastAction = self.getGreedyAction(legal, stateFeatures)

        # Update parameters for next action
        self.lastState = state
        return self.lastAction

    def getGreedyAction(self, legal: list[str], stateFeatures: GameStateFeatures) -> str:
        """
        Choose the greedy action for the given state features.

        Args:
            legal: The list of legal actions in the current state.
            stateFeatures: The features of the current state.

        Returns:
            The action to take.
        """

        # Create list of all q values for legal moves
        q_values = [self.getQValue(stateFeatures, action) for action in legal]

        # Place actions with maximum value in max_actions list
        max_q = max(q_values)

        # If there is more than one action with the maximum value, choose one at random
        max_actions = [action for action in legal if self.getQValue(stateFeatures, action) == max_q]
        
        return random.choice(max_actions)

    def final(self, state: GameState):
        """Handle the end of episodes. This is called by the game after a win or a loss."""

        # Compute the reward between the previous state and the current state
        reward = self.computeReward(self.lastState, state)

        # Update q values
        self.learn(GameStateFeatures(self.lastState), self.lastAction, reward, GameStateFeatures(state))

        # Reset variables
        self.lastAction = None
        self.lastState = None

        # Increment the number of episodes
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            print('%s\n%s' % ('Training Done (turning off epsilon and alpha)', '-' * 49))
            self.setAlpha(0)
            self.setEpsilon(0)