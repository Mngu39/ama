"""Smoke test of the read-only C++ afterstate preview API.

Run from repository root:
    python -m teams.mngu39.td_1step.check_afterstate
"""

from rl_env import puyo_cpp
from .features import CHANNELS, HEIGHT, WIDTH, encode_afterstate


def check(seed: int):
    env = puyo_cpp.PuyoEnv()
    env.reset(seed)
    total = 0
    while not env.done():
        before = env.observation()
        mask_before = env.legal_actions()
        candidates = env.preview_all()

        assert len(candidates) == sum(mask_before), "Wrong preview count"
        assert len({c["action"] for c in candidates}) == len(candidates)
        assert env.observation() == before, "preview_all changed real state"
        assert env.legal_actions() == mask_before

        for c in candidates:
            action = c["action"]
            assert mask_before[action], f"Preview contains illegal action {action}"
            assert env.preview(action) == {
                k: v for k, v in c.items() if k != "action"
            }, "preview vs preview_all mismatch"
            planes = encode_afterstate(c["observation"])
            assert len(planes) == CHANNELS
            assert all(len(p) == HEIGHT for p in planes)
            assert all(len(r) == WIDTH for p in planes for r in p)
            assert env.observation() == before, "preview changed real state"

        # Select deterministically so result is repeatable.
        chosen = candidates[(seed + total * 7) % len(candidates)]
        actual = env.step(chosen["action"])
        expected = {k: v for k, v in chosen.items() if k != "action"}
        assert actual == expected, "Actual step differs from its preview"
        total += 1
    return total


def main():
    for seed in (12345, 100000, 100001):
        turns = check(seed)
        print(f"[PASS] seed={seed}, turns={turns}")
    print("AFTERSTATE PREVIEW TEST PASSED")


if __name__ == "__main__":
    main()
