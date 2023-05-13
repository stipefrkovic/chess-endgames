from experiment import ExperimentRunner, RookEndgamesExperiment

def main():
    experiment_runner = ExperimentRunner()
    experiment_runner.set_up()

    rook_endgames_experiment = RookEndgamesExperiment(3, 20, 0.95, 2)
    experiment_runner.run_experiment(rook_endgames_experiment)

    experiment_runner.wrap_up()

if __name__ == "__main__":
    main()
