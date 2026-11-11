#pragma once

// Exact, short-circuit candidate generator for Ama's cell::create_queue.
//
// Used only by the counterfactual rule_based sampler. Never changes the
// original Ama cell::create_queue or the real game's queue generation.
// A rejected candidate does not allocate/construct any full pair vector.
// The accepted candidate is bit-for-bit equivalent to create_queue(seed).

#include "../core/cell.h"
#include <array>
#include <stdexcept>
#include <utility>

namespace ama_fast_queue {

constexpr u32 MULTIPLIER = u32(0x5D588B65);
constexpr u32 INCREMENT  = u32(0x269EC3);
constexpr u32 WARMUP_CALLS = 5;
constexpr u32 CALLS_PER_SHUFFLE = 2 * (15 * 8 + 7 * 16 + 3 * 32); // 656

inline u32 next(u32& state) {
    state = state * MULTIPLIER + INCREMENT; // u32 overflow is intentional
    return state;
}

// Jump exactly n calls in the same modulo-2^32 LCG, without iterating n times.
inline u32 jump_ahead(u32 state, u32 n) {
    u32 acc_mul = 1, acc_add = 0;
    u32 cur_mul = MULTIPLIER, cur_add = INCREMENT;
    while (n != 0) {
        if (n & 1) {
            acc_mul = acc_mul * cur_mul;
            acc_add = acc_add * cur_mul + cur_add;
        }
        cur_add = (cur_mul + 1) * cur_add;
        cur_mul = cur_mul * cur_mul;
        n >>= 1;
    }
    return acc_mul * state + acc_add;
}

// Keep the exact original call order, indices, and swap sequence.
inline void shuffle(std::array<u8, 256>& queue, u32& state) {
    for (i32 col = 0; col < 15; ++col) {
        for (i32 i = 0; i < 8; ++i) {
            i32 n1 = i32(next(state) >> 28) + col * 16;
            i32 n2 = i32(next(state) >> 28) + (col + 1) * 16;
            std::swap(queue[size_t(n1)], queue[size_t(n2)]);
        }
    }
    for (i32 col = 0; col < 7; ++col) {
        for (i32 i = 0; i < 16; ++i) {
            i32 n1 = i32(next(state) >> 27) + col * 32;
            i32 n2 = i32(next(state) >> 27) + (col + 1) * 32;
            std::swap(queue[size_t(n1)], queue[size_t(n2)]);
        }
    }
    for (i32 col = 0; col < 3; ++col) {
        for (i32 i = 0; i < 32; ++i) {
            i32 n1 = i32(next(state) >> 26) + col * 64;
            i32 n2 = i32(next(state) >> 26) + (col + 1) * 64;
            std::swap(queue[size_t(n1)], queue[size_t(n2)]);
        }
    }
}

inline bool same_pair(const std::array<u8, 256>& q, size_t pair_index,
                      const cell::Pair& expected) {
    return cell::Type(q[2 * pair_index]) == expected.first &&
           cell::Type(q[2 * pair_index + 1]) == expected.second;
}

// Returns false when the supplied seed's Ama queue does not match the three
// public pairs at indices turn..turn+2. Only on true is `accepted` populated.
// This is the SAME seed rejection distribution as the original algorithm.
inline bool try_candidate(u32 seed, int turn,
                          const std::array<cell::Pair, 3>& visible,
                          cell::Queue& accepted) {
    if (turn < 0 || turn + 3 > 128)
        throw std::out_of_range("Invalid visible triple position");

    u32 state = jump_ahead(seed, WARMUP_CALLS);
    std::array<u8, 4> initial_three = {};

    if (turn < 2) {
        // Pair 0 and pair 1 come from the 3-color queue. Generate it ONLY
        // while it is among the visible pairs, and reject before 4-color work.
        std::array<u8, 256> q3{};
        for (size_t i = 0; i < q3.size(); ++i) q3[i] = u8(i % 3);
        shuffle(q3, state);
        if (turn == 0 && (!same_pair(q3, 0, visible[0]) ||
                          !same_pair(q3, 1, visible[1]))) return false;
        if (turn == 1 && !same_pair(q3, 1, visible[0])) return false;
        for (size_t i = 0; i < 4; ++i) initial_three[i] = q3[i];
    } else {
        // Mode-0 uses 656 RNG draws before mode-1. Skip the physical
        // 3-color shuffle, but preserve the EXACT RNG state entering mode-1.
        state = jump_ahead(state, CALLS_PER_SHUFFLE);
    }

    std::array<u8, 256> q4{};
    for (size_t i = 0; i < q4.size(); ++i) q4[i] = u8(i % 4);
    shuffle(q4, state);

    // The 5-color shuffle happens later than the 4-color shuffle in Ama
    // and its result is never used in the returned queue: skip it entirely.
    if (turn < 2) {
        for (size_t i = 0; i < 4; ++i) q4[i] = initial_three[i];
    }
    for (int k = 0; k < 3; ++k) {
        if (!same_pair(q4, size_t(turn + k), visible[size_t(k)])) return false;
    }

    if (turn >= 2) {
        // The first 2 pairs cannot affect this rejection decision. Recover
        // them ONLY for an accepted candidate, to return the full EXACT Ama
        // queue (including indices 0 and 1) for parity/audit purposes.
        std::array<u8, 256> q3{};
        for (size_t i = 0; i < q3.size(); ++i) q3[i] = u8(i % 3);
        u32 three_state = jump_ahead(seed, WARMUP_CALLS);
        shuffle(q3, three_state);
        for (size_t i = 0; i < 4; ++i) q4[i] = q3[i];
    }

    accepted.resize(128);
    for (size_t i = 0; i < accepted.size(); ++i)
        accepted[i] = { cell::Type(q4[2 * i]), cell::Type(q4[2 * i + 1]) };
    return true;
}

} // namespace ama_fast_queue
