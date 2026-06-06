import numpy as np
import copy
import math
import torch
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from tactical import (
    TACTICAL_FORCE_THRESHOLD,
    best_bounded_two_ply_threat_move,
    best_forcing_win_move,
    best_tactical_move,
    opponent_of,
    ranked_tactical_moves,
    winning_moves,
)


def _copy_board(board):
    copier = getattr(board, "copy", None)
    if callable(copier):
        return copier()
    return copy.deepcopy(board)


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def forced_tactical_move(
    board,
    threshold=TACTICAL_FORCE_THRESHOLD,
    return_reason=False,
    two_ply_threats=False,
    two_ply_max_candidates=16,
    two_ply_max_replies=6,
    two_ply_max_followups=12,
):
    return best_tactical_move(
        board,
        threshold=threshold,
        return_reason=return_reason,
        two_ply_threats=two_ply_threats,
        two_ply_max_candidates=two_ply_max_candidates,
        two_ply_max_replies=two_ply_max_replies,
        two_ply_max_followups=two_ply_max_followups,
    )


def _normalized_action_probs(action_probs):
    action_probs = list(action_probs)
    if not action_probs:
        return [], np.array([], dtype=np.float64)

    actions = [int(action) for action, _prob in action_probs]
    probs = np.array([float(prob) for _action, prob in action_probs], dtype=np.float64)
    probs[~np.isfinite(probs)] = 0.0
    probs = np.maximum(probs, 0.0)
    probs_sum = float(np.sum(probs))
    if probs_sum <= 0.0:
        probs = np.full(len(actions), 1.0 / len(actions), dtype=np.float64)
    else:
        probs = probs / probs_sum
    return actions, probs


def _tactical_prior_for_actions(
    board,
    actions,
    temperature=1.0,
    two_ply_bonus=0.0,
    two_ply_max_candidates=16,
    two_ply_max_replies=6,
    two_ply_max_followups=12,
):
    if not actions:
        return np.array([], dtype=np.float64), False

    ranked = ranked_tactical_moves(board)
    scores_by_move = {
        int(item["move"]): max(0.0, float(item["score"]))
        for item in ranked
    }
    two_ply_move = None
    if float(two_ply_bonus or 0.0) > 0.0:
        two_ply_move = best_bounded_two_ply_threat_move(
            board,
            max_candidates=two_ply_max_candidates,
            max_replies=two_ply_max_replies,
            max_followups=two_ply_max_followups,
        )
        if two_ply_move is not None and int(two_ply_move) in set(actions):
            scores_by_move[int(two_ply_move)] = max(
                scores_by_move.get(int(two_ply_move), 0.0),
                float(two_ply_bonus),
            )
            two_ply_hit = True
        else:
            two_ply_hit = False
    else:
        two_ply_hit = False
    logits = np.array(
        [np.log1p(scores_by_move.get(int(action), 0.0)) for action in actions],
        dtype=np.float64,
    )
    logits[~np.isfinite(logits)] = 0.0
    if float(np.max(logits) - np.min(logits)) <= 1e-12:
        return np.full(len(actions), 1.0 / len(actions), dtype=np.float64), two_ply_hit

    temperature = max(float(temperature), 1e-6)
    logits = (logits - np.max(logits)) / temperature
    probs = np.exp(logits)
    probs_sum = float(np.sum(probs))
    if probs_sum <= 0.0 or not np.isfinite(probs_sum):
        return np.full(len(actions), 1.0 / len(actions), dtype=np.float64), two_ply_hit
    return probs / probs_sum, two_ply_hit


def tactical_leaf_value(
    board,
    win_value=1.0,
    loss_value=0.95,
    forcing_value=0.85,
    two_ply_value=0.70,
    two_ply=False,
    max_candidates=16,
    max_replies=6,
    max_followups=12,
    return_reason=False,
):
    if not board.availables:
        result = None
    else:
        current = board.get_current_player()
        opponent = opponent_of(board, current)
        current_wins = winning_moves(board, current)
        opponent_wins = winning_moves(board, opponent)

        if current_wins:
            result = (float(win_value), "win")
        elif len(opponent_wins) >= 2:
            result = (-float(loss_value), "multiple_immediate_losses")
        elif best_forcing_win_move(
            board,
            current,
            max_candidates=max_candidates,
        ) is not None:
            result = (float(forcing_value), "forcing_win")
        elif two_ply and best_bounded_two_ply_threat_move(
            board,
            current,
            max_candidates=max_candidates,
            max_replies=max_replies,
            max_followups=max_followups,
        ) is not None:
            result = (float(two_ply_value), "two_ply_threat")
        else:
            result = None

    if return_reason:
        if result is None:
            return None, None
        return result
    return None if result is None else result[0]


class TreeNode:
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}  # a map from action to TreeNode
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.P = prior_p
        self.virtual_loss = 0
        self.lock = threading.Lock()

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
        according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        parent_scale = math.sqrt(self.n_visits + self.virtual_loss)
        best_action = None
        best_node = None
        best_value = -float("inf")
        for action, node in self.children.items():
            node.u = (
                c_puct * node.P * parent_scale
                / (1 + node.n_visits + node.virtual_loss)
            )
            value = node.Q - node.virtual_loss + node.u
            if value > best_value:
                best_action = action
                best_node = node
                best_value = value
        return best_action, best_node

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
        value Q, and prior probability P, on this node's score.
        """
        self.u = (c_puct * self.P *
                  math.sqrt(self.parent.n_visits + self.parent.virtual_loss) / (1 + self.n_visits + self.virtual_loss))
        return self.Q - self.virtual_loss + self.u

    def apply_virtual_loss(self):
        with self.lock:
            self.virtual_loss += 1

    def remove_virtual_loss(self):
        with self.lock:
            self.virtual_loss -= 1

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
        perspective.
        """
        # Count visit.
        self.n_visits += 1
        # Update Q, a running average of values for all visits.
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MCTS:
    """An implementation of Monte Carlo Tree Search."""
    def __init__(
        self,
        policy_value_fn,
        c_puct=5,
        n_playout=10000,
        policy_value_batch_fn=None,
        tactical_prior_weight=0.0,
        tactical_prior_temperature=1.0,
        tactical_prior_two_ply_bonus=0.0,
        tactical_prior_two_ply_max_candidates=16,
        tactical_prior_two_ply_max_replies=6,
        tactical_prior_two_ply_max_followups=12,
        tactical_leaf_eval=False,
        tactical_leaf_win_value=1.0,
        tactical_leaf_loss_value=0.95,
        tactical_leaf_forcing_value=0.85,
        tactical_leaf_two_ply_value=0.70,
        tactical_leaf_two_ply=False,
        tactical_leaf_max_candidates=16,
        tactical_leaf_max_replies=6,
        tactical_leaf_max_followups=12,
    ):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a parameter controlling the level of exploration.
        n_playout: number of simulations for each search.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._policy_batch = policy_value_batch_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._tactical_prior_weight = max(0.0, min(1.0, float(tactical_prior_weight)))
        self._tactical_prior_temperature = max(float(tactical_prior_temperature), 1e-6)
        self._tactical_prior_two_ply_bonus = max(0.0, float(tactical_prior_two_ply_bonus or 0.0))
        self._tactical_prior_two_ply_max_candidates = int(tactical_prior_two_ply_max_candidates)
        self._tactical_prior_two_ply_max_replies = int(tactical_prior_two_ply_max_replies)
        self._tactical_prior_two_ply_max_followups = int(tactical_prior_two_ply_max_followups)
        self.tactical_prior_applications = 0
        self.tactical_prior_two_ply_hits = 0
        self._tactical_leaf_eval = bool(tactical_leaf_eval)
        self._tactical_leaf_win_value = float(tactical_leaf_win_value)
        self._tactical_leaf_loss_value = float(tactical_leaf_loss_value)
        self._tactical_leaf_forcing_value = float(tactical_leaf_forcing_value)
        self._tactical_leaf_two_ply_value = float(tactical_leaf_two_ply_value)
        self._tactical_leaf_two_ply = bool(tactical_leaf_two_ply)
        self._tactical_leaf_max_candidates = int(tactical_leaf_max_candidates)
        self._tactical_leaf_max_replies = int(tactical_leaf_max_replies)
        self._tactical_leaf_max_followups = int(tactical_leaf_max_followups)
        self.tactical_leaf_evaluations = 0
        self.tactical_leaf_positive = 0
        self.tactical_leaf_negative = 0
        self.tactical_leaf_reasons = {}
        self.batched_policy_batches = 0
        self.batched_policy_positions = 0

    def _root_action_priors(self, board, node, action_probs):
        if self._tactical_prior_weight <= 0.0 or not node.is_root():
            return action_probs

        actions, probs = _normalized_action_probs(action_probs)
        if len(actions) <= 1:
            return list(zip(actions, probs))

        tactical, two_ply_hit = _tactical_prior_for_actions(
            board,
            actions,
            temperature=self._tactical_prior_temperature,
            two_ply_bonus=self._tactical_prior_two_ply_bonus,
            two_ply_max_candidates=self._tactical_prior_two_ply_max_candidates,
            two_ply_max_replies=self._tactical_prior_two_ply_max_replies,
            two_ply_max_followups=self._tactical_prior_two_ply_max_followups,
        )
        weight = self._tactical_prior_weight
        probs = (1.0 - weight) * probs + weight * tactical
        probs = probs / np.sum(probs)
        self.tactical_prior_applications += 1
        if two_ply_hit:
            self.tactical_prior_two_ply_hits += 1
        return list(zip(actions, probs))

    def _tactical_leaf_value(self, board):
        if not self._tactical_leaf_eval:
            return None, None
        value, reason = tactical_leaf_value(
            board,
            win_value=self._tactical_leaf_win_value,
            loss_value=self._tactical_leaf_loss_value,
            forcing_value=self._tactical_leaf_forcing_value,
            two_ply_value=self._tactical_leaf_two_ply_value,
            two_ply=self._tactical_leaf_two_ply,
            max_candidates=self._tactical_leaf_max_candidates,
            max_replies=self._tactical_leaf_max_replies,
            max_followups=self._tactical_leaf_max_followups,
            return_reason=True,
        )
        if value is None:
            return None, None
        self.tactical_leaf_evaluations += 1
        self.tactical_leaf_reasons[reason] = self.tactical_leaf_reasons.get(reason, 0) + 1
        if value > 0:
            self.tactical_leaf_positive += 1
        elif value < 0:
            self.tactical_leaf_negative += 1
        return value, reason

    def get_move_probs(self, board, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        # Use parallel playouts if configured (we'll default to sequential if not)
        # But here we will just run sequential loop for now, or we can switch to parallel
        # if we want to enforce it.
        # For now, let's keep this sequential and add a parallel version.
        for _ in range(self._n_playout):
            state_copy = _copy_board(board)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.n_visits)
                      for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def _select_leaf(self, board, apply_virtual_loss=False):
        node = self._root
        virtual_loss_nodes = []
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            if apply_virtual_loss:
                node.apply_virtual_loss()
                virtual_loss_nodes.append(node)
            board.do_move(action)
        return node, virtual_loss_nodes

    def _terminal_leaf_value(self, board, winner):
        if winner == -1:
            return 0.0
        return 1.0 if winner == board.get_current_player() else -1.0

    def _finish_playout(self, node, board, action_probs, leaf_value, virtual_loss_nodes):
        end, winner = board.game_end()
        if not end:
            tactical_value, _reason = self._tactical_leaf_value(board)
            if tactical_value is not None:
                leaf_value = tactical_value
            action_probs = self._root_action_priors(board, node, action_probs)
            node.expand(action_probs)
        else:
            leaf_value = self._terminal_leaf_value(board, winner)

        for virtual_node in virtual_loss_nodes:
            virtual_node.remove_virtual_loss()
        node.update_recursive(-leaf_value)

    def get_move_probs_batched(self, board, temp=1e-3, batch_size=16):
        """Run MCTS with batched neural leaf evaluation."""
        batch_size = max(1, int(batch_size))
        playouts_done = 0
        if self._root.is_leaf() and self._n_playout > 0:
            self._playout(_copy_board(board))
            playouts_done = 1

        while playouts_done < self._n_playout:
            records = []
            for _ in range(min(batch_size, self._n_playout - playouts_done)):
                state_copy = _copy_board(board)
                node, virtual_loss_nodes = self._select_leaf(
                    state_copy,
                    apply_virtual_loss=True,
                )
                end, winner = state_copy.game_end()
                if end:
                    leaf_value = self._terminal_leaf_value(state_copy, winner)
                    for virtual_node in virtual_loss_nodes:
                        virtual_node.remove_virtual_loss()
                    node.update_recursive(-leaf_value)
                    playouts_done += 1
                    continue
                records.append((node, virtual_loss_nodes, state_copy))
                playouts_done += 1

            if not records:
                continue

            boards = [record[2] for record in records]
            if self._policy_batch is None:
                evaluations = [self._policy(leaf_board) for leaf_board in boards]
            else:
                evaluations = self._policy_batch(boards)
                self.batched_policy_batches += 1
                self.batched_policy_positions += len(boards)

            for (node, virtual_loss_nodes, leaf_board), (action_probs, leaf_value) in zip(records, evaluations):
                self._finish_playout(
                    node,
                    leaf_board,
                    action_probs,
                    leaf_value,
                    virtual_loss_nodes,
                )

        act_visits = [(act, node.n_visits)
                      for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def get_move_probs_parallel(self, board, temp=1e-3, num_threads=8):
        """Run playouts in parallel."""
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for _ in range(self._n_playout):
                state_copy = _copy_board(board)
                futures.append(executor.submit(self._playout_parallel, state_copy))
            
            # Wait for all to complete
            for f in futures:
                f.result()

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.n_visits)
                      for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root.children:
            self._root = self._root.children[last_move]
            self._root.parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def _playout(self, board):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            board.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples and a score for the current player.
        action_probs, leaf_value = self._policy(board)

        # Check for end of game.
        end, winner = board.game_end()
        if not end:
            tactical_value, _reason = self._tactical_leaf_value(board)
            if tactical_value is not None:
                leaf_value = tactical_value
            action_probs = self._root_action_priors(board, node, action_probs)
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == board.get_current_player() else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _playout_parallel(self, board):
        """Run a single playout in parallel with virtual loss."""
        node = self._root
        visited_nodes = []
        
        # Selection
        while True:
            with node.lock:
                if node.is_leaf():
                    break
                # Greedily select next move.
                action, node = node.select(self._c_puct)
                node.apply_virtual_loss()
                visited_nodes.append(node)
            
            board.do_move(action)

        # Evaluation
        # This policy call should be blocking and thread-safe (handled by batching worker)
        action_probs, leaf_value = self._policy(board)

        # Check for end of game.
        end, winner = board.game_end()
        if not end:
            tactical_value, _reason = self._tactical_leaf_value(board)
            if tactical_value is not None:
                leaf_value = tactical_value
            with node.lock:
                # Double check if it's still a leaf (another thread might have expanded it)
                if node.is_leaf():
                    action_probs = self._root_action_priors(board, node, action_probs)
                    node.expand(action_probs)
                else:
                    # If already expanded, we might want to continue selection? 
                    # For simplicity, just backpropagate the value we got.
                    pass
        else:
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == board.get_current_player() else -1.0

        # Backpropagation
        # Remove virtual loss and update
        for n in visited_nodes:
            n.remove_virtual_loss()
        
        node.update_recursive(-leaf_value)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(
        self,
        policy_value_function,
        c_puct=5,
        n_playout=2000,
        is_selfplay=0,
        use_parallel=True,
        policy_value_batch_function=None,
        mcts_batch_size=1,
        mcts_min_batches_per_search=1,
        tactical_threshold=TACTICAL_FORCE_THRESHOLD,
        two_ply_threats=False,
        two_ply_max_candidates=16,
        two_ply_max_replies=6,
        two_ply_max_followups=12,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.25,
        dirichlet_moves=None,
        tactical_prior_weight=0.0,
        tactical_prior_temperature=1.0,
        tactical_prior_two_ply_bonus=0.0,
        tactical_prior_two_ply_max_candidates=16,
        tactical_prior_two_ply_max_replies=6,
        tactical_prior_two_ply_max_followups=12,
        tactical_leaf_eval=False,
        tactical_leaf_win_value=1.0,
        tactical_leaf_loss_value=0.95,
        tactical_leaf_forcing_value=0.85,
        tactical_leaf_two_ply_value=0.70,
        tactical_leaf_two_ply=False,
        tactical_leaf_max_candidates=16,
        tactical_leaf_max_replies=6,
        tactical_leaf_max_followups=12,
    ):
        self.mcts = MCTS(
            policy_value_function,
            c_puct,
            n_playout,
            policy_value_batch_fn=policy_value_batch_function,
            tactical_prior_weight=tactical_prior_weight,
            tactical_prior_temperature=tactical_prior_temperature,
            tactical_prior_two_ply_bonus=tactical_prior_two_ply_bonus,
            tactical_prior_two_ply_max_candidates=tactical_prior_two_ply_max_candidates,
            tactical_prior_two_ply_max_replies=tactical_prior_two_ply_max_replies,
            tactical_prior_two_ply_max_followups=tactical_prior_two_ply_max_followups,
            tactical_leaf_eval=tactical_leaf_eval,
            tactical_leaf_win_value=tactical_leaf_win_value,
            tactical_leaf_loss_value=tactical_leaf_loss_value,
            tactical_leaf_forcing_value=tactical_leaf_forcing_value,
            tactical_leaf_two_ply_value=tactical_leaf_two_ply_value,
            tactical_leaf_two_ply=tactical_leaf_two_ply,
            tactical_leaf_max_candidates=tactical_leaf_max_candidates,
            tactical_leaf_max_replies=tactical_leaf_max_replies,
            tactical_leaf_max_followups=tactical_leaf_max_followups,
        )
        self._is_selfplay = is_selfplay
        self.use_parallel = use_parallel
        self.mcts_batch_size = max(1, int(mcts_batch_size or 1))
        self.mcts_min_batches_per_search = max(1, int(mcts_min_batches_per_search or 1))
        self.effective_mcts_batch_size = self.mcts_batch_size
        self.tactical_threshold = (
            TACTICAL_FORCE_THRESHOLD
            if tactical_threshold is None
            else float(tactical_threshold)
        )
        self.two_ply_threats = bool(two_ply_threats)
        self.two_ply_max_candidates = int(two_ply_max_candidates)
        self.two_ply_max_replies = int(two_ply_max_replies)
        self.two_ply_max_followups = int(two_ply_max_followups)
        self.dirichlet_alpha = max(float(dirichlet_alpha), 1e-6)
        self.dirichlet_frac = max(0.0, min(1.0, float(dirichlet_frac)))
        self.dirichlet_moves = (
            None
            if dirichlet_moves is None
            else max(0, int(dirichlet_moves))
        )
        self.tactical_prior_weight = max(0.0, min(1.0, float(tactical_prior_weight)))
        self.tactical_prior_temperature = max(float(tactical_prior_temperature), 1e-6)
        self.tactical_leaf_eval = bool(tactical_leaf_eval)
        self.player = None
        self.forced_tactical_moves = 0
        self.threat_solver_moves = 0
        self.two_ply_threat_moves = 0
        self.search_moves = 0
        self.search_duration_s = 0.0
        self.selfplay_moves = 0
        self.dirichlet_noise_moves = 0
        self.no_noise_moves = 0
        self.tactical_prior_searches = 0
        self.tactical_prior_two_ply_hits = 0

    @property
    def tactical_leaf_evaluations(self):
        return self.mcts.tactical_leaf_evaluations

    @property
    def tactical_prior_two_ply_applications(self):
        return self.mcts.tactical_prior_two_ply_hits

    @property
    def tactical_leaf_positive(self):
        return self.mcts.tactical_leaf_positive

    @property
    def tactical_leaf_negative(self):
        return self.mcts.tactical_leaf_negative

    @property
    def tactical_leaf_reasons(self):
        return dict(self.mcts.tactical_leaf_reasons)

    @property
    def batched_policy_batches(self):
        return self.mcts.batched_policy_batches

    @property
    def batched_policy_positions(self):
        return self.mcts.batched_policy_positions

    def _effective_mcts_batch_size(self):
        batch_size = self.mcts_batch_size
        if batch_size <= 1 or self.mcts_min_batches_per_search <= 1:
            return batch_size
        remaining_playouts = max(1, self.mcts._n_playout - 1)
        max_batch_size = math.ceil(
            remaining_playouts / self.mcts_min_batches_per_search
        )
        return max(1, min(batch_size, max_batch_size))

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)
        self.selfplay_moves = 0

    def _current_dirichlet_frac(self):
        if not self._is_selfplay:
            return 0.0
        if self.dirichlet_moves is not None and self.selfplay_moves >= self.dirichlet_moves:
            return 0.0
        return self.dirichlet_frac

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            forced = forced_tactical_move(
                board,
                threshold=self.tactical_threshold,
                return_reason=True,
                two_ply_threats=self.two_ply_threats,
                two_ply_max_candidates=self.two_ply_max_candidates,
                two_ply_max_replies=self.two_ply_max_replies,
                two_ply_max_followups=self.two_ply_max_followups,
            )
            forced_move = forced["move"]
            if forced_move is not None:
                move_probs[forced_move] = 1.0
                self.forced_tactical_moves += 1
                if forced["reason"] == "forcing_win":
                    self.threat_solver_moves += 1
                if forced["reason"] == "two_ply_threat":
                    self.two_ply_threat_moves += 1
                if self._is_selfplay:
                    self.selfplay_moves += 1
                    self.no_noise_moves += 1
                self.mcts.update_with_move(forced_move if self._is_selfplay else -1)
                if return_prob:
                    return forced_move, move_probs
                return forced_move

            search_start = time.perf_counter()
            effective_batch_size = self._effective_mcts_batch_size()
            self.effective_mcts_batch_size = effective_batch_size
            if effective_batch_size > 1:
                acts, probs = self.mcts.get_move_probs_batched(
                    board,
                    temp,
                    batch_size=effective_batch_size,
                )
            elif self.use_parallel:
                acts, probs = self.mcts.get_move_probs_parallel(board, temp)
            else:
                acts, probs = self.mcts.get_move_probs(board, temp)
            self.search_duration_s += time.perf_counter() - search_start
            self.search_moves += 1
            if self.tactical_prior_weight > 0.0:
                self.tactical_prior_searches += 1
            
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                noise_frac = self._current_dirichlet_frac()
                if noise_frac > 0.0 and len(probs) > 1:
                    noise = np.random.dirichlet(self.dirichlet_alpha * np.ones(len(probs)))
                    action_probs = (1.0 - noise_frac) * probs + noise_frac * noise
                    action_probs = action_probs / np.sum(action_probs)
                    self.dirichlet_noise_moves += 1
                else:
                    action_probs = probs
                    self.no_noise_moves += 1
                move = np.random.choice(acts, p=action_probs)
                self.selfplay_moves += 1
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
                # location = board.move_to_location(move)
                # print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
