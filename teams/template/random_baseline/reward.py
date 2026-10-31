def compute_reward(result):
    # Example: reward based on chain count
    chains = result["chains"]
    return float(chains ** 2)

'''
점수중심 보상 예시
def compute_reward(result):
    return float(result["score"])
'''