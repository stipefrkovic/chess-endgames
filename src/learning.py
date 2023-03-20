from src.util import fen_to_bitboard


def compute_tds(model, variation):
    evaluations = [model.predict(state) for state in variation.moves]
    evaluations.append(variation.evaluation)
    tds = [evaluations[i+1] - evaluations[i] for i in range(len(evaluations)-1)]
    return tds


def compute_td_lambda_update(variation, tds, lambda_value):
    input_data = []
    output_data = []
    for state_idx, state in enumerate(variation.moves):
        input_data.append(fen_to_bitboard(state))
        target_value = 0.0
        for td_idx, td in enumerate(tds[state_idx:]):
            target_value += pow(lambda_value, td_idx) * td
        output_data.append(target_value)
    return input_data, output_data
