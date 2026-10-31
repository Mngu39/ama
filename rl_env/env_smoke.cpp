
#include "puyo_env.h"
#include <iostream>
#include <cassert>

int main() {
    PuyoEnv env;
    auto obs = env.reset(12345);

    assert(obs.turn == 0);
    assert(cell::to_char(obs.visible[0].first) == 'R');
    assert(cell::to_char(obs.visible[0].second) == 'Y');

    const int expected_legal[10] = {
        22, 22, 11, 22, 22, 11, 22, 22, 22, 11
    };

    int total_score = 0;

    for (int turn = 0; turn < 10; ++turn) {
        auto mask = env.legal_actions();

        int n = std::count(mask.begin(), mask.end(), true);
        assert(n == expected_legal[turn]);

        // Same action ordering as the earlier engine test
        int nth = turn % n;
        int chosen = -1;

        for (int id = 0; id < PuyoEnv::ACTION_COUNT; ++id) {
            if (mask[id] && nth-- == 0) {
                chosen = id;
                break;
            }
        }

        assert(chosen >= 0);

        auto result = env.step(chosen);
        total_score += result.score;

        std::cout << "Turn " << turn + 1
                  << " | action=" << chosen
                  << " | chain=" << result.chains
                  << " | score=" << result.score
                  << "\n";

        assert(result.observation.turn == turn + 1);
        assert(!result.terminated);
        assert(!result.truncated);
    }

    assert(total_score == 40);
    std::cout << "PuyoEnv smoke PASSED\n";
    return 0;
}
