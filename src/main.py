from experiment import ExperimentRunner, RookEndgameExperiment, QueenEndgameExperiment
from learning import ChessMLP

# TODO log settings

def main():
    chess_mlp = ChessMLP()
    experiment_runner = ExperimentRunner(chess_mlp)
    experiment_runner.set_up(load_weights=False)

    rook_endgame_experiment = RookEndgameExperiment("rook_endgames", 6, 10, 0.9)
    experiment_runner.run_experiments(rook_endgame_experiment, 10)
    
    # queen_endgame_experiment = QueenEndgameExperiment("queen_endgames", 2, 10, 0.9)
    # experiment_runner.run_experiments(queen_endgame_experiment, 2)

    # experiment_runner.wrap_up(save_weights=True)

if __name__ == "__main__":
    main()
