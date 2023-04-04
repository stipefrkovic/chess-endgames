from src.util import fen_to_bitboard


def compute_evaluations(moves, model):
    evaluations = []
    for move in moves:
        evaluation = model.predict(move)
        evaluations.append(evaluation)
    return evaluations


def compute_tds(reward, evaluations):
    evaluations = evaluations
    evaluations.append(reward)
    tds = [evaluations[i+1] - evaluations[i] for i in range(len(evaluations)-1)]
    return tds


def compute_lambda_tds(moves, tds, lambda_value):
    lambda_tds = []
    for moves_idx in range(len(moves)):
        target_value = 0.0
        for td_idx, td in enumerate(tds[moves_idx:]):
            target_value += pow(lambda_value, td_idx) * td
        lambda_tds.append(target_value)
    return lambda_tds
