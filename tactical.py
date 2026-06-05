import copy
import math


DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
WIN_SCORE = 1_000_000.0
OPEN_FOUR_SCORE = 250_000.0
CLOSED_FOUR_SCORE = 120_000.0
OPEN_THREE_SCORE = 35_000.0
CLOSED_THREE_SCORE = 8_000.0
OPEN_TWO_SCORE = 1_500.0
TACTICAL_FORCE_THRESHOLD = OPEN_FOUR_SCORE


def opponent_of(board, player):
    return board.players[0] if player == board.players[1] else board.players[1]


def _inside(board, x, y):
    return 0 <= x < board.width and 0 <= y < board.height


def _empty(board, x, y):
    return _inside(board, x, y) and (y * board.width + x) not in board.states


def _count_direction(board, x, y, dx, dy, player):
    count = 0
    x += dx
    y += dy
    while _inside(board, x, y) and board.states.get(y * board.width + x) == player:
        count += 1
        x += dx
        y += dy
    return count, x, y


def would_win(board, move, player):
    old_last_move = board.last_move
    board.states[move] = player
    board.last_move = move
    try:
        won, winner = board.has_a_winner()
        return won and winner == player
    finally:
        board.states.pop(move, None)
        board.last_move = old_last_move


def winning_moves(board, player):
    return [move for move in board.availables if would_win(board, move, player)]


def _board_after_move(board, move, player):
    child = copy.deepcopy(board)
    child.current_player = player
    child.do_move(move)
    return child


def _line_score(count, open_ends, n_in_row):
    if count >= n_in_row:
        return WIN_SCORE
    if count == n_in_row - 1:
        if open_ends == 2:
            return OPEN_FOUR_SCORE
        if open_ends == 1:
            return CLOSED_FOUR_SCORE
    if count == n_in_row - 2:
        if open_ends == 2:
            return OPEN_THREE_SCORE
        if open_ends == 1:
            return CLOSED_THREE_SCORE
    if count == n_in_row - 3 and open_ends == 2:
        return OPEN_TWO_SCORE
    return max(0.0, (count - 1) * 80.0 + open_ends * 20.0)


def line_shape(board, move, player, dx, dy):
    y = move // board.width
    x = move % board.width
    forward, fx, fy = _count_direction(board, x, y, dx, dy, player)
    backward, bx, by = _count_direction(board, x, y, -dx, -dy, player)
    open_ends = int(_empty(board, fx, fy)) + int(_empty(board, bx, by))
    return 1 + forward + backward, open_ends


def locality_score(board, move):
    y = move // board.width
    x = move % board.width
    center_y = (board.height - 1) / 2
    center_x = (board.width - 1) / 2
    center_distance = abs(y - center_y) + abs(x - center_x)

    neighbor_count = 0
    own_neighbor_count = 0
    current = board.get_current_player()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ny = y + dy
            nx = x + dx
            if not _inside(board, nx, ny):
                continue
            neighbor = board.states.get(ny * board.width + nx)
            if neighbor is None:
                continue
            neighbor_count += 1
            if neighbor == current:
                own_neighbor_count += 1

    return neighbor_count * 24.0 + own_neighbor_count * 10.0 - center_distance * 1.5


def shape_score(board, move, player):
    if move not in board.availables:
        return -math.inf
    if would_win(board, move, player):
        return WIN_SCORE

    scores = move_line_scores(board, move, player)
    return scores[0] + 0.55 * scores[1] + locality_score(board, move)


def move_line_scores(board, move, player):
    scores = []
    for dx, dy in DIRECTIONS:
        count, open_ends = line_shape(board, move, player, dx, dy)
        scores.append(_line_score(count, open_ends, board.n_in_row))
    scores.sort(reverse=True)
    return scores


def fork_threat_count(board, move, player, threshold=OPEN_FOUR_SCORE):
    if move not in board.availables:
        return 0
    if would_win(board, move, player):
        return 0
    return sum(1 for score in move_line_scores(board, move, player) if score >= threshold)


def creates_unanswerable_threat(board, move, player=None, min_winning_replies=2):
    """Return true when a move creates multiple immediate wins next turn."""
    if move not in board.availables:
        return False

    player = board.get_current_player() if player is None else player
    if would_win(board, move, player):
        return False

    child = _board_after_move(board, move, player)
    opponent = child.get_current_player()
    if winning_moves(child, opponent):
        return False
    return len(winning_moves(child, player)) >= min_winning_replies


def forcing_win_moves(board, player=None, max_candidates=32, min_winning_replies=2):
    player = board.get_current_player() if player is None else player
    moves = []
    for item in ranked_tactical_moves(board, player)[:max_candidates]:
        move = item["move"]
        if creates_unanswerable_threat(
            board,
            move,
            player,
            min_winning_replies=min_winning_replies,
        ):
            moves.append(move)
    return moves


def best_forcing_win_move(board, player=None, max_candidates=32, min_winning_replies=2):
    moves = forcing_win_moves(
        board,
        player=player,
        max_candidates=max_candidates,
        min_winning_replies=min_winning_replies,
    )
    return None if not moves else moves[0]


def _dedupe_available(board, moves):
    seen = set()
    available = set(board.availables)
    deduped = []
    for move in moves:
        if move in seen or move not in available:
            continue
        deduped.append(move)
        seen.add(move)
    return deduped


def _ranked_move_ids(board, player, limit):
    limit = max(0, int(limit))
    if limit <= 0:
        return []
    return [item["move"] for item in ranked_tactical_moves(board, player)[:limit]]


def plausible_reply_moves(board, attacker, defender=None, max_replies=8):
    """Return a bounded set of likely defensive replies for threat-space probes."""
    defender = board.get_current_player() if defender is None else defender
    replies = []
    replies.extend(winning_moves(board, defender))
    replies.extend(winning_moves(board, attacker))
    replies.extend(_ranked_move_ids(board, defender, max_replies))
    return _dedupe_available(board, replies)[: max(0, int(max_replies))]


def _has_forcing_followup(
    board,
    attacker,
    max_candidates=16,
    min_winning_replies=2,
):
    if winning_moves(board, attacker):
        return True
    return best_forcing_win_move(
        board,
        attacker,
        max_candidates=max_candidates,
        min_winning_replies=min_winning_replies,
    ) is not None


def creates_bounded_two_ply_threat(
    board,
    move,
    player=None,
    max_replies=8,
    max_followups=16,
    min_winning_replies=2,
    include_one_ply=False,
):
    """Return true when a move survives plausible replies and still converts.

    This is intentionally a bounded threat-space teacher/search primitive. It
    checks immediate wins, required blocks, and the top ranked tactical replies,
    rather than proving the move against every legal board reply.
    """
    if move not in board.availables:
        return False

    player = board.get_current_player() if player is None else player
    if would_win(board, move, player):
        return False

    if include_one_ply and creates_unanswerable_threat(
        board,
        move,
        player,
        min_winning_replies=min_winning_replies,
    ):
        return True

    child = _board_after_move(board, move, player)
    defender = child.get_current_player()
    if winning_moves(child, defender):
        return False

    replies = plausible_reply_moves(
        child,
        player,
        defender=defender,
        max_replies=max_replies,
    )
    if not replies:
        return False

    for reply in replies:
        reply_board = _board_after_move(child, reply, defender)
        end, winner = reply_board.game_end()
        if end:
            if winner == defender or winner == -1:
                return False
            continue
        if not _has_forcing_followup(
            reply_board,
            player,
            max_candidates=max_followups,
            min_winning_replies=min_winning_replies,
        ):
            return False
    return True


def bounded_two_ply_threat_moves(
    board,
    player=None,
    max_candidates=32,
    max_replies=8,
    max_followups=16,
    min_winning_replies=2,
    include_one_ply=False,
):
    player = board.get_current_player() if player is None else player
    moves = []
    for item in ranked_tactical_moves(board, player)[: max(0, int(max_candidates))]:
        move = item["move"]
        if creates_bounded_two_ply_threat(
            board,
            move,
            player=player,
            max_replies=max_replies,
            max_followups=max_followups,
            min_winning_replies=min_winning_replies,
            include_one_ply=include_one_ply,
        ):
            moves.append(move)
    return moves


def best_bounded_two_ply_threat_move(
    board,
    player=None,
    max_candidates=32,
    max_replies=8,
    max_followups=16,
    min_winning_replies=2,
    include_one_ply=False,
):
    moves = bounded_two_ply_threat_moves(
        board,
        player=player,
        max_candidates=max_candidates,
        max_replies=max_replies,
        max_followups=max_followups,
        min_winning_replies=min_winning_replies,
        include_one_ply=include_one_ply,
    )
    return None if not moves else moves[0]


def ranked_tactical_moves(board, player=None):
    player = board.get_current_player() if player is None else player
    opponent = opponent_of(board, player)
    ranked = []
    for move in board.availables:
        attack = shape_score(board, move, player)
        defense = shape_score(board, move, opponent)
        ranked.append({
            "move": move,
            "score": max(attack, defense * 1.08) + min(attack, defense) * 0.12,
            "attack": attack,
            "defense": defense,
        })
    ranked.sort(key=lambda item: (item["score"], item["attack"], -item["move"]), reverse=True)
    return ranked


def _maybe_reason(move, reason, return_reason):
    if return_reason:
        return {"move": move, "reason": reason}
    return move


def best_tactical_move(
    board,
    include_quiet=False,
    threshold=TACTICAL_FORCE_THRESHOLD,
    return_reason=False,
    two_ply_threats=False,
    two_ply_max_candidates=16,
    two_ply_max_replies=6,
    two_ply_max_followups=12,
):
    if not board.availables:
        return _maybe_reason(None, None, return_reason)

    current = board.get_current_player()
    opponent = opponent_of(board, current)

    current_wins = winning_moves(board, current)
    if current_wins:
        return _maybe_reason(current_wins[0], "win", return_reason)

    opponent_wins = winning_moves(board, opponent)
    if opponent_wins:
        return _maybe_reason(opponent_wins[0], "block_win", return_reason)

    forcing_move = best_forcing_win_move(board, current)
    if forcing_move is not None:
        return _maybe_reason(forcing_move, "forcing_win", return_reason)

    if two_ply_threats:
        two_ply_move = best_bounded_two_ply_threat_move(
            board,
            current,
            max_candidates=two_ply_max_candidates,
            max_replies=two_ply_max_replies,
            max_followups=two_ply_max_followups,
        )
        if two_ply_move is not None:
            return _maybe_reason(two_ply_move, "two_ply_threat", return_reason)

    ranked = ranked_tactical_moves(board, current)
    if not ranked:
        return _maybe_reason(None, None, return_reason)

    best = ranked[0]
    if include_quiet or best["score"] >= threshold:
        return _maybe_reason(best["move"], "shape", return_reason)
    return _maybe_reason(None, None, return_reason)
