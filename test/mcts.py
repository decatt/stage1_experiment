import numpy as np
import copy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import microrts_agent


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors -- output from policy function - a list of tuples of actions
            and their prior probability according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
        Returns:
        A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.

        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search.
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """Arguments:
        policy_value_fn -- policy_value_fn: currently adopted policy function
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
                # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of (action, probability)
        # tuples p and also a score v in [-1, 1] for the current player.
        action_probs, leaf_value = self._policy(state)

        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Runs all playouts sequentially and returns the available actions and their corresponding probabilities
        Arguments:
        state -- the current state, including both game state and the current player.
        temp -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
        the available actions and the corresponding probabilities
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on the visit counts at the root node
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(visits))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, available_actions, state, temp=1e-3, return_prob=0):
        sensible_moves = available_actions
        move_probs = np.zeros(len(available_actions))
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(state, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                self.mcts.update_with_move(move)  # update the root node and reuse the search tree
            else:
                # with the default temp=1e-3, this is almost equivalent to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MoveProbs(nn.Module):
    def __init__(self, action_space):
        super(MoveProbs, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 6 * 6, 256)),
            nn.ReLU(), )
        self.actor = layer_init(nn.Linear(256, action_space), std=0.01)

