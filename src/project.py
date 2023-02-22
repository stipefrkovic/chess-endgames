import chess


def main():
    endgame_fen = "5k2/3Q4/8/4K3/8/8/8/8 w - - 0 1"
    checkmate_fen = "5k2/3Q4/5K2/8/8/8/8/8 w - - 0 1"
    endgame_board = chess.Board(checkmate_fen)
    print(endgame_board)
    print(endgame_board.outcome())
    print("turn: " + str(endgame_board.turn))

    endgame_board.push_san("Qd8")
    print(endgame_board)
    print(endgame_board.outcome())
    print("winner: " + str(endgame_board.outcome().winner))
    print("turn: " + str(endgame_board.turn))


if __name__ == "__main__":
    main()
