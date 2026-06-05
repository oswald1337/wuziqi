import random
from tactical import best_tactical_move, ranked_tactical_moves


class RandomPlayer:
    def __init__(self, seed=None):
        self.player = None
        self._rng = random.Random(seed)

    def set_player_ind(self, player):
        self.player = player

    def reset_player(self):
        pass

    def get_action(self, board):
        if not board.availables:
            return None
        return self._rng.choice(list(board.availables))

    def __str__(self):
        return f"Random {self.player}"


class HeuristicPlayer:
    def __init__(self, seed=None):
        self.player = None
        self._rng = random.Random(seed)

    def set_player_ind(self, player):
        self.player = player

    def reset_player(self):
        pass

    def get_action(self, board):
        if not board.availables:
            return None

        forced = best_tactical_move(board)
        if forced is not None:
            return forced

        ranked = ranked_tactical_moves(board)
        if not ranked:
            return self._rng.choice(list(board.availables))
        best_score = ranked[0]["score"]
        top_moves = [
            item["move"]
            for item in ranked[:8]
            if abs(item["score"] - best_score) < 1e-6
        ]
        return self._rng.choice(top_moves or [ranked[0]["move"]])

    def __str__(self):
        return f"Heuristic {self.player}"
