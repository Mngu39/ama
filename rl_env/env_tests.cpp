
#include "puyo_env.h"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <algorithm>

void check_same(
    const PuyoEnv::Observation& a,
    const PuyoEnv::Observation& b
) {
    assert(a.board == b.board);
    assert(a.row14 == b.row14);
    assert(a.visible == b.visible);
    assert(a.turn == b.turn);
}

int main() {
    PuyoEnv env;

    // 1. All 22 actions on an empty board
    for (int action = 0; action < 22; ++action) {
        env.reset(12345);
        assert(env.legal_actions()[action]);

        auto result = env.step(action);
        assert(result.observation.turn == 1);
        assert(!result.terminated);
        assert(!result.truncated);
    }
    std::cout << "[PASS] All 22 initial actions\n";

    // 2. Equal-color pair: only 11 actions
    env.reset(12345);
    env.step(0);
    env.step(1);

    auto mask = env.legal_actions();
    assert(std::count(mask.begin(), mask.end(), true) == 11);

    for (int i = 0; i < 22; ++i) {
        assert(mask[i] == (i < 11));
    }
    std::cout << "[PASS] Equal-color action mask\n";

    // 3. Illegal actions must not change the state
    auto before = env.observation();

    for (int bad : {-1, 11, 22}) {
        bool caught = false;
        try {
            env.step(bad);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
        check_same(before, env.observation());
    }
    std::cout << "[PASS] Illegal action rejection\n";

    // 4. Compare two environments over 100 seeds
    PuyoEnv a, b;
    int total_turns = 0;
    int terminated_count = 0;
    int truncated_count = 0;

    for (u32 seed = 0; seed < 100; ++seed) {
        check_same(a.reset(seed), b.reset(seed));

        int turn = 0;

        while (!a.done()) {
            assert(!b.done());

            auto ma = a.legal_actions();
            auto mb = b.legal_actions();
            assert(ma == mb);

            int chosen = -1;
            for (int k = 0; k < 22; ++k) {
                int id = (seed + turn * 7 + k) % 22;
                if (ma[id]) {
                    chosen = id;
                    break;
                }
            }
            assert(chosen >= 0);

            auto ra = a.step(chosen);
            auto rb = b.step(chosen);

            check_same(ra.observation, rb.observation);
            assert(ra.chains == rb.chains);
            assert(ra.score == rb.score);
            assert(ra.terminated == rb.terminated);
            assert(ra.truncated == rb.truncated);

            ++turn;
            assert(turn <= PuyoEnv::MAX_TURNS);

            if (ra.terminated) ++terminated_count;
            if (ra.truncated) ++truncated_count;
        }

        assert(b.done());
        assert(terminated_count + truncated_count == int(seed) + 1);

        bool caught = false;
        try {
            a.step(0);
        } catch (const std::logic_error&) {
            caught = true;
        }
        assert(caught);

        total_turns += turn;
    }

    std::cout << "[PASS] 100-seed deterministic replay\n";
    std::cout << "[PASS] Episode termination / truncation\n";
    std::cout << "[PASS] Post-episode action rejection\n";

    // 5. Reset after completed episodes
    auto fresh = a.reset(12345);
    assert(!a.done());
    assert(fresh.turn == 0);

    auto reset_mask = a.legal_actions();
    assert(std::count(
        reset_mask.begin(), reset_mask.end(), true
    ) == 22);

    std::cout << "[PASS] Reset after episode\n";
    std::cout << "Total turns: " << total_turns << "\n";
    std::cout << "Terminated: " << terminated_count << "\n";
    std::cout << "Truncated: " << truncated_count << "\n";
    std::cout << "ALL TESTS PASSED\n";
}
