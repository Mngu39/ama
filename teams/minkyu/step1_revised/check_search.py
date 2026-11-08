"""Integration smoke test against the REAL compiled Ama C++ extension."""

import numpy as np
import torch
from rl_env import puyo_cpp
from .features import encode_afterstate
from .network import AfterstateValueNet
from .search import greedy_search


def main():
    torch.set_num_threads(min(2, torch.get_num_threads()))
    env = puyo_cpp.PuyoEnv()
    if not hasattr(env, 'clone'):
        raise RuntimeError('Missing PuyoEnv.clone(); rebuild C++ pybind extension')
    for seed in (7, 12345, 98765):
        obs = env.reset(seed)
        base = encode_afterstate(obs)
        assert len(base) == 5 and len(base[0]) == 14
        for _ in range(3):
            candidates = env.preview_all()
            assert candidates
            a = int(candidates[0]['action'])
            before = env.observation()
            clone = env.clone()
            simulated = clone.step(a)
            previewed = env.preview(a)
            assert simulated['score'] == previewed['score']
            assert simulated['chains'] == previewed['chains']
            assert simulated['observation'] == previewed['observation']
            assert env.observation() == before, 'Cloned branch mutated real game'
            actual = env.step(a)
            assert actual['observation'] == previewed['observation']
    env.reset(12345)
    model = AfterstateValueNet(5, 8, 1).eval()
    with torch.no_grad():
        one = greedy_search(env, model, torch.device('cpu'), .995, 1000.,
                            -1000., ply=1)
        three = greedy_search(env, model, torch.device('cpu'), .995, 1000.,
                              -1000., ply=3, root_keep=5, second_keep=10)
    legal = [int(c['action']) for c in env.preview_all()]
    assert one.action in legal and three.action in legal
    assert three.evaluated <= 352, three
    assert not env.done() and env.observation()['turn'] == 0
    print('REAL CPP / 1-PLY / 3-PLY INTEGRATION CHECK PASSED')


if __name__ == '__main__':
    main()
