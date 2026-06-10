# rank.py scaffold
def load_candidates(path):
    with open(path, "r") as f:
        return json.load(f)
