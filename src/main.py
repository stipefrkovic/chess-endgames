from experiment import GamePlayer, RookEndgameGame, QueenEndgameGame
from learning import ChessMLP

# TODO log settings

def main():
    chess_mlp = ChessMLP()
    game_player = GamePlayer(chess_mlp)
    game_player.set_up(load_weights=False)

    rook_endgame_game = RookEndgameGame("rook_endgames", 3, 10, 0.9)
    game_player.play_games(rook_endgame_game, 2)
    
    # queen_endgame_game = QueenEndgameGame("queen_endgames", 3, 10, 0.9)
    # game_player.play_games(queen_endgame_game, 2)

    game_player.wrap_up(save_weights=True)

if __name__ == "__main__":
    main()
