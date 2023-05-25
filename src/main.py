from game import GamePlayer, RookEndgameGame, QueenEndgameGame
from learn import ChessMLP
import argparse


def queen_endgame_raw_train():
    chess_mlp = ChessMLP()
    game_player = GamePlayer(chess_mlp)
    game_player.set_up(
        load_weights=True,
        clear_dirs=False
    )

    queen_endgame_game = QueenEndgameGame(
        max_depth=5,
        train=True,
        train_steps=15,
        lambda_value=0.9,
        name="queen_endgame_raw_train"
    )
    game_player.play_game(
        game=queen_endgame_game,
        iterations=1000,
        save_every_n_iter=500
    )

    game_player.wrap_up(
        save_weights=True
    )


def rook_endgame_transfer_train():
    chess_mlp = ChessMLP()
    game_player = GamePlayer(chess_mlp)
    game_player.set_up(
        load_weights=True,
        clear_dirs=True
    )

    rook_endgame_game = RookEndgameGame(
        max_depth=5,
        train=True,
        train_steps=15,
        lambda_value=0.9,
        name="rook_endgame_transfer_train"
    )
    game_player.play_game(
        game=rook_endgame_game,
        iterations=500,
        save_every_n_iter=50
    )

    game_player.wrap_up(
        save_weights=True
    )


def rook_endgame_raw_train():
    chess_mlp = ChessMLP()
    game_player = GamePlayer(chess_mlp)
    game_player.set_up(
        load_weights=False,
        clear_dirs=True
    )

    rook_endgame_game = RookEndgameGame(
        max_depth=5,
        train=True,
        train_steps=15,
        lambda_value=0.9,
        name="rook_endgame_raw_train"
    )
    game_player.play_game(
        game=rook_endgame_game,
        iterations=500,
        save_every_n_iter=50
    )

    game_player.wrap_up(
        save_weights=True
    )


def main():
    experiments = {
        "1": queen_endgame_raw_train,
        "2": rook_endgame_transfer_train,
        "3": rook_endgame_raw_train,
    }
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('-e', '--experiment', choices=('1', '2', '3'), required=True)
    args = parser.parse_args()
    experiments[args.experiment]()


if __name__ == "__main__":
    main()
