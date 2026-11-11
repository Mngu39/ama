#!/usr/bin/env python3
"""Narrow, idempotent update to Ama's conditional rule_based fork.

Run at the Ama repository root. This changes only the C++ rule-based method
and inserts one new header include; it preserves all other local edits.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

METHOD = '''    // Rejection-sample independent Ama seeds conditional on the three pairs
    // visible at this turn. The fast candidate generator skips unused shuffles,
    // jumps the LCG ahead, and rejects as early as correctness allows. It
    // produces EXACTLY cell::create_queue(seed) for each accepted candidate.
    // No original hidden pair or past-history information is inspected.
    PuyoEnv fork_with_hidden_rule_seed(u32 fresh_seed) const {
        if (queue_.size() < size_t(turn_ + 3))
            throw std::logic_error("Not enough visible pairs for conditional fork");
        std::mt19937 candidate_rng(fresh_seed);
        constexpr u32 MAX_ATTEMPTS = 2'000'000;
        const std::array<cell::Pair, 3> visible = {
            queue_[size_t(turn_)],
            queue_[size_t(turn_ + 1)],
            queue_[size_t(turn_ + 2)]
        };
        cell::Queue accepted;
        for (u32 trial = 0; trial < MAX_ATTEMPTS; ++trial) {
            const u32 candidate_seed = static_cast<u32>(candidate_rng());
            if (!ama_fast_queue::try_candidate(candidate_seed, turn_, visible, accepted))
                continue;
            PuyoEnv fork = *this;
            for (size_t i = size_t(turn_ + 3); i < queue_.size(); ++i)
                fork.queue_[i] = accepted[i];
            return fork;
        }
        throw std::runtime_error(
            "Ama rule_based conditional seed sampling exceeded 2000000 trials; "
            "no approximate or oracle fallback was used");
    }'''


def replace_method(source: str) -> str:
    signature = r'\bPuyoEnv\s+fork_with_hidden_rule_seed\s*\(\s*u32\s+fresh_seed\s*\)\s*const\s*\{'
    match = re.search(signature, source)
    if match is None:
        raise ValueError('Cannot locate the existing fork_with_hidden_rule_seed(u32) method')
    brace_start = source.index('{', match.start())
    level = 0
    end = None
    for i in range(brace_start, len(source)):
        if source[i] == '{':
            level += 1
        elif source[i] == '}':
            level -= 1
            if level == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError('Malformed method body: unbalanced braces')
    start = source.rfind('\n', 0, match.start()) + 1
    prefix = source[:start].splitlines(keepends=True)
    # Remove comments only when they unmistakably belong to this method.
    idx = len(prefix)
    while idx and prefix[idx - 1].lstrip().startswith('//'):
        idx -= 1
    comment_block = ''.join(prefix[idx:])
    if ('Draw an independent Ama queue conditioned' in comment_block or
            'Rejection-sample independent Ama seeds conditional' in comment_block or
            'Sample Ama\'s exact UNCONDITIONAL queue-generation' in comment_block):
        start = len(''.join(prefix[:idx]))
    source = source[:start] + METHOD + source[end:]
    include = '#include "ama_fast_queue.h"'
    if include not in source:
        source = source.replace('#pragma once', '#pragma once\n' + include, 1)
    return source


def main() -> int:
    root = Path.cwd()
    hdr = root / 'rl_env/puyo_env.h'
    fast = root / 'rl_env/ama_fast_queue.h'
    project = root / 'teams/minkyu/step_comparison'
    if not hdr.is_file() or not fast.is_file() or not project.is_dir():
        print('ERROR: extract ZIP in the Ama root, then run from there.', file=sys.stderr)
        return 1
    raw = hdr.read_bytes()
    newline = '\r\n' if b'\r\n' in raw else '\n'
    try:
        updated = replace_method(raw.decode('utf-8').replace('\r\n', '\n'))
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    updated_bytes = updated.replace('\n', newline).encode('utf-8')
    if updated_bytes == raw:
        print('Already applied: header unchanged')
    else:
        backup = hdr.with_name('puyo_env.h.before_rule_based_fast.bak')
        if not backup.exists():
            backup.write_bytes(raw)
        hdr.write_bytes(updated_bytes)
        print('Updated rule_based fork method only:', hdr)
        print('Backup:', backup)
    print('Rebuild existing engine with: python teams/minkyu/step_comparison/scripts/build_existing_engine.py')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
