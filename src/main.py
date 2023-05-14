from experiment import ExperimentRunner, RookEndgamesExperiment
from learning import ChessMLP

# TODO add logging

def main():
    chess_mlp = ChessMLP()
    experiment_runner = ExperimentRunner(chess_mlp)
    experiment_runner.set_up(load_weights=False)

    rook_endgames_experiment = RookEndgamesExperiment(5, 10, 0.9)
    experiment_runner.run_experiments(rook_endgames_experiment, 50)

    experiment_runner.wrap_up(save_weights=True)

if __name__ == "__main__":
    main()
