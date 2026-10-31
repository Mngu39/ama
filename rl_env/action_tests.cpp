
#include "puyo_env.h"
#include <array>
#include <iostream>

int main() {
    PuyoEnv env;

    for (int action = 0; action < 22; ++action) {
        auto initial = env.reset(12345);

        // Seed 12345 starts with (RED, YELLOW)
        if (initial.visible[0] !=
            cell::Pair{cell::Type::RED, cell::Type::YELLOW}) {
            std::cerr << "Unexpected initial pair\n";
            return 1;
        }

        auto mask = env.legal_actions();
        if (!mask[action]) {
            std::cerr << "Action incorrectly masked: "
                      << action << "\n";
            return 1;
        }

        // Independently construct the expected field
        std::array<std::array<char, 6>, 13> expected{};
        for (auto& row : expected) {
            row.fill('.');
        }

        if (action < 6) {
            // UP: R below Y
            int x = action;
            expected[0][x] = 'R';
            expected[1][x] = 'Y';
        }
        else if (action < 11) {
            // RIGHT: R left, Y right
            int x = action - 6;
            expected[0][x] = 'R';
            expected[0][x + 1] = 'Y';
        }
        else if (action < 17) {
            // DOWN: Y below R
            int x = action - 11;
            expected[0][x] = 'Y';
            expected[1][x] = 'R';
        }
        else {
            // LEFT: Y left, R right
            int x = action - 16;
            expected[0][x - 1] = 'Y';
            expected[0][x] = 'R';
        }

        auto result = env.step(action);

        if (result.observation.board != expected ||
            result.observation.row14 != 0 ||
            result.chains != 0 ||
            result.score != 0 ||
            result.observation.turn != 1) {

            std::cerr << "FAILED: action "
                      << action << "\n";
            return 1;
        }
    }

    std::cout << "[PASS] All 22 action placements\n";
    std::cout << "[PASS] Axis/child color ordering\n";
    std::cout << "[PASS] Empty-board observation\n";
    std::cout << "ALL ACTION TESTS PASSED\n";
    return 0;
}
