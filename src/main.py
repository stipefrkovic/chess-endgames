from experiment import GamePlayer, RookEndgameGame, QueenEndgameGame
from learning import ChessMLP

# TODO log settings

def main():
    chess_mlp = ChessMLP()
    game_player = GamePlayer(chess_mlp)
    game_player.set_up(
        load_weights=False,
        clear_dirs=True
        )
    
    queen_endgame_game = QueenEndgameGame(
        max_depth=3,
        train_steps=10,
        lambda_value=0.9
        )
    game_player.play_game(
        game=queen_endgame_game,
        iterations=10
        )

    rook_endgame_game = RookEndgameGame(
        max_depth=3,
        train_steps=10,
        lambda_value=0.9
        )
    game_player.play_game(
        game=rook_endgame_game,
        iterations=10
        )
    
    game_player.wrap_up(
        save_weights=True
        )

if __name__ == "__main__":
    main()
