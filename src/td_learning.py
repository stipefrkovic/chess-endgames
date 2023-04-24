def compute_tds(reward, evaluations):
    evaluations = evaluations
    evaluations.append(reward)
    tds = [evaluations[i+1] - evaluations[i] for i in range(len(evaluations)-1)]
    return tds


def compute_lambda_tds(tds, lambda_value):
    lambda_tds = []
    for moves_idx in range(len(tds)):
        lambda_tds.append(compute_lambda_td(tds[moves_idx:], lambda_value))
    return lambda_tds


def compute_lambda_td(tds, lambda_value):
    lambda_td = 0.0
    for td_idx, td in enumerate(tds):
        lambda_td += pow(lambda_value, td_idx) * td
    return lambda_td
