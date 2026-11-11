// Standalone parity/performance check (compile via check_rule_based_fast.py).
#include "rl_env/ama_fast_queue.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;
using Triple = std::array<cell::Pair, 3>;

Triple visible_at(const cell::Queue& q, int turn) {
    return {q[size_t(turn)], q[size_t(turn + 1)], q[size_t(turn + 2)]};
}

std::pair<cell::Queue, u32> baseline(u32 fresh_seed, int turn, Triple visible,
                                     u32* trials) {
    std::mt19937 mt(fresh_seed);
    for (u32 i = 1; i <= 2'000'000; ++i) {
        const u32 candidate_seed = static_cast<u32>(mt());
        auto q = cell::create_queue(candidate_seed);
        if (visible_at(q, turn) == visible) {
            *trials = i;
            return {std::move(q), candidate_seed};
        }
    }
    throw std::runtime_error("baseline trial budget exceeded");
}

std::pair<cell::Queue, u32> fast(u32 fresh_seed, int turn, Triple visible,
                                 u32* trials) {
    std::mt19937 mt(fresh_seed);
    cell::Queue out;
    for (u32 i = 1; i <= 2'000'000; ++i) {
        const u32 candidate_seed = static_cast<u32>(mt());
        if (ama_fast_queue::try_candidate(candidate_seed, turn, visible, out)) {
            *trials = i;
            return {std::move(out), candidate_seed};
        }
    }
    throw std::runtime_error("fast trial budget exceeded");
}

int main() {
    // Jump-ahead must be bit-identical to per-call LCG advancement.
    for (u32 seed : {u32(0),u32(1),u32(12345),u32(0xffffffff),u32(0xdeadbeef)}) {
        for (u32 n : {u32(0),u32(1),u32(5),u32(656),u32(661),u32(1317),u32(20000)}) {
            u32 state = seed;
            for (u32 i=0;i<n;++i) ama_fast_queue::next(state);
            assert(state == ama_fast_queue::jump_ahead(seed, n));
        }
    }

    // Accepted candidates must be identical FULL 128-pair queues to Ama.
    // Rejections must also agree exactly across initial/normal/late positions.
    std::mt19937 mt(424242);
    constexpr int TURNS[] = {0,1,2,3,4,10,20,31,49};
    u32 accepted_count = 0, rejected_count = 0;
    for (int i=0;i<4000;++i) {
        auto seed = static_cast<u32>(mt());
        auto original = cell::create_queue(seed);
        for (int turn : TURNS) {
            auto public_visible = visible_at(original, turn);
            cell::Queue fast_queue;
            assert(ama_fast_queue::try_candidate(seed, turn, public_visible, fast_queue));
            assert(fast_queue == original);
            ++accepted_count;
            // A wrong visible triple from an unrelated queue should be
            // accepted iff original create_queue(seed) also matches it.
            auto unrelated = cell::create_queue(static_cast<u32>(mt()));
            auto wrong_visible = visible_at(unrelated, turn);
            cell::Queue discarded;
            bool expected = (public_visible == wrong_visible);
            bool observed = ama_fast_queue::try_candidate(seed,turn,wrong_visible,discarded);
            assert(expected == observed);
            if (observed) assert(discarded == original);
            else assert(discarded.empty());
            ++rejected_count;
        }
    }
    std::cout << "PARITY OK: " << accepted_count << " matched seed/turn cases, "
              << rejected_count << " alternate visible cases, full queues identical\n";

    // Rejection-trial sequence and accepted seed/queue must remain IDENTICAL.
    // Time includes 20 different fresh_seed samples at 0/1/2/10/30 turns.
    const auto sample_queue = cell::create_queue(12345);
    double original_seconds = 0, fast_seconds = 0;
    u64 trial_total = 0;
    for (int turn : {0,1,2,10,30}) {
        for (u32 i=0;i<4;++i) {
            const u32 fresh_seed = 5000 + u32(turn * 13) + i;
            const auto wanted = visible_at(sample_queue, turn);
            u32 n0=0, n1=0;
            const auto st0=Clock::now();
            const auto [q0,s0] = baseline(fresh_seed,turn,wanted,&n0);
            const auto st1=Clock::now();
            const auto [q1,s1] = fast(fresh_seed,turn,wanted,&n1);
            const auto st2=Clock::now();
            assert(n0 == n1 && s0 == s1 && q0 == q1);
            original_seconds += std::chrono::duration<double>(st1-st0).count();
            fast_seconds += std::chrono::duration<double>(st2-st1).count();
            trial_total += n0;
        }
    }
    std::cout << "REJECTION PARITY OK: 20 cases, " << trial_total
              << " total candidate trials, accepted seeds/queues identical\n";
    std::cout << std::fixed << std::setprecision(3)
              << "BENCHMARK baseline=" << original_seconds << " s, fast="
              << fast_seconds << " s, speedup=" << original_seconds / fast_seconds
              << "x (this machine, C++ benchmark; not training throughput)\n";
    return 0;
}
