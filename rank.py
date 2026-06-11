# rank.py scaffold
def normalize_scores(score):
    # normalize to [0.50, 0.99] range
    return max(0.50, min(0.99, score))
