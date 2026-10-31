
#include <iostream>
#include "../core/move.h"
#include "../core/chain.h"

int main()
{
    // Test 1: Known 1-chain result
    Field check;
    const auto red = cell::Type::RED;

    check.drop_pair(0, direction::Type::UP, {red, red});
    check.drop_pair(0, direction::Type::UP, {red, red});

    auto popped = check.pop();
    auto result = chain::get_score(popped);

    if (result.count != 1 || result.score != 40 ||
        !check.is_empty()) {
        std::cerr << "Chain test FAILED\n";
        return 1;
    }
    std::cout << "Chain test PASSED\n";

    // Test 2: Play 10 moves without any search AI
    const auto queue = cell::create_queue(12345);
    Field field;

    std::cout << "Current / NEXT / NEXT2: ";
    for (int j = 0; j < 3; ++j) {
        std::cout << "("
                  << cell::to_char(queue[j].first)
                  << ","
                  << cell::to_char(queue[j].second)
                  << ") ";
    }
    std::cout << "\n";

    int total_score = 0;

    for (int turn = 0; turn < 10; ++turn) {
        const auto& pair = queue[turn];

        auto legal = move::generate(
            field, pair.first == pair.second
        );

        if (legal.get_size() == 0) {
            std::cout << "No legal moves\n";
            break;
        }

        // Deterministic test action; not an AI policy
        auto action = legal[turn % legal.get_size()];

        field.drop_pair(action.x, action.r, pair);

        auto masks = field.pop();
        auto outcome = chain::get_score(masks);
        total_score += outcome.score;

        std::cout << "Turn " << (turn + 1)
                  << " | legal=" << legal.get_size()
                  << " | x=" << int(action.x)
                  << " | rotation=" << int(action.r)
                  << " | chain=" << outcome.count
                  << " | score=" << outcome.score
                  << "\n";
    }

    std::cout << "\nFinal field:\n";
    field.print();

    std::cout << "Total score: " << total_score << "\n";
    return 0;
}
