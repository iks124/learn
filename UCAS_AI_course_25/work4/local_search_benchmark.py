#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproducible benchmark template for:
- 8-puzzle (Hamming / Manhattan)
- 8-queens (conflict pairs)
Algorithms:
- Hill Climbing: steepest-ascent, first-choice
- Simulated Annealing

Metrics:
- success rate
- search cost (neighbor evaluations)
- best-cost curve vs steps (averaged)

Usage examples:
  python local_search_benchmark.py --problem puzzle --heuristic manhattan --algo all --runs 200 --max_steps 5000 --seed 0
  python local_search_benchmark.py --problem queens --algo all --runs 500 --max_steps 2000 --seed 42
"""

from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Common experiment logging
# -------------------------

@dataclass
class RunResult:
    success: bool
    steps: int
    evals: int  # "search dissipation": number of neighbor evaluations
    final_cost: int
    best_curve: List[int]  # best-so-far cost per step (length=steps+1)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def pad_and_average_curves(curves: List[List[int]], max_len: int) -> np.ndarray:
    """
    curves: list of best_curves, each length varies.
    We pad with last value to max_len, then average.
    """
    arr = np.zeros((len(curves), max_len), dtype=float)
    for i, c in enumerate(curves):
        if len(c) >= max_len:
            arr[i] = np.array(c[:max_len], dtype=float)
        else:
            arr[i, :len(c)] = np.array(c, dtype=float)
            arr[i, len(c):] = float(c[-1])
    return arr.mean(axis=0)


# -------------------------
# 8-Puzzle
# -------------------------
# State representation: tuple of 9 ints, 0 is blank.
# Goal: (1,2,3,4,5,6,7,8,0)

PUZZLE_GOAL = (1, 2, 3,
               4, 5, 6,
               7, 8, 0)

PUZZLE_GOAL_POS = {v: i for i, v in enumerate(PUZZLE_GOAL)}  # tile -> index


def puzzle_is_solvable(state: Tuple[int, ...]) -> bool:
    # For 3x3, solvable iff inversion parity is even.
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            inv += (arr[i] > arr[j])
    return (inv % 2) == 0


def puzzle_neighbors(state: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    z = state.index(0)
    r, c = divmod(z, 3)
    nbrs = []
    swaps = []
    if r > 0: swaps.append(z - 3)
    if r < 2: swaps.append(z + 3)
    if c > 0: swaps.append(z - 1)
    if c < 2: swaps.append(z + 1)
    for s in swaps:
        lst = list(state)
        lst[z], lst[s] = lst[s], lst[z]
        nbrs.append(tuple(lst))
    return nbrs


def puzzle_hamming(state: Tuple[int, ...]) -> int:
    # number of misplaced tiles (excluding blank)
    return sum(1 for i, v in enumerate(state) if v != 0 and v != PUZZLE_GOAL[i])


def puzzle_manhattan(state: Tuple[int, ...]) -> int:
    dist = 0
    for i, v in enumerate(state):
        if v == 0:
            continue
        gi = PUZZLE_GOAL_POS[v]
        r1, c1 = divmod(i, 3)
        r2, c2 = divmod(gi, 3)
        dist += abs(r1 - r2) + abs(c1 - c2)
    return dist


def puzzle_generate_by_reverse_walk(depth: int, rng: random.Random) -> Tuple[int, ...]:
    """
    Generate solvable instance by starting from goal and applying random moves.
    This controls difficulty roughly by depth.
    """
    s = PUZZLE_GOAL
    for _ in range(depth):
        nbrs = puzzle_neighbors(s)
        s = rng.choice(nbrs)
    return s


# -------------------------
# 8-Queens
# -------------------------
# Representation: tuple length N (N=8), index=row, value=col in [0,N-1]
# Cost: number of attacking pairs (0 is solved)

def queens_random_state(n: int, rng: random.Random) -> Tuple[int, ...]:
    return tuple(rng.randrange(n) for _ in range(n))


def queens_cost(state: Tuple[int, ...]) -> int:
    n = len(state)
    conflicts = 0
    for r1 in range(n):
        c1 = state[r1]
        for r2 in range(r1 + 1, n):
            c2 = state[r2]
            if c1 == c2:
                conflicts += 1
            elif abs(r1 - r2) == abs(c1 - c2):
                conflicts += 1
    return conflicts


def queens_neighbors(state: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    n = len(state)
    nbrs = []
    for r in range(n):
        for c in range(n):
            if c == state[r]:
                continue
            lst = list(state)
            lst[r] = c
            nbrs.append(tuple(lst))
    return nbrs


# -------------------------
# Local search algorithms
# -------------------------

def hill_climb_steepest(
    init_state,
    cost_fn: Callable,
    neighbors_fn: Callable,
    max_steps: int,
) -> RunResult:
    state = init_state
    cur_cost = cost_fn(state)
    best = cur_cost
    curve = [best]
    evals = 0

    for step in range(1, max_steps + 1):
        nbrs = neighbors_fn(state)
        # Evaluate all neighbors
        costs = []
        for ns in nbrs:
            evals += 1
            costs.append((cost_fn(ns), ns))
        costs.sort(key=lambda x: x[0])
        best_n_cost, best_n_state = costs[0]

        if best_n_cost < cur_cost:
            state = best_n_state
            cur_cost = best_n_cost
            best = min(best, cur_cost)
        else:
            # local optimum / plateau stop
            curve.append(best)
            return RunResult(success=(best == 0), steps=step-1, evals=evals, final_cost=cur_cost, best_curve=curve)

        curve.append(best)

        if best == 0:
            return RunResult(success=True, steps=step, evals=evals, final_cost=cur_cost, best_curve=curve)

    return RunResult(success=(best == 0), steps=max_steps, evals=evals, final_cost=cur_cost, best_curve=curve)


def hill_climb_first_choice(
    init_state,
    cost_fn: Callable,
    neighbors_fn: Callable,
    max_steps: int,
    max_tries_per_step: int,
    rng: random.Random
) -> RunResult:
    state = init_state
    cur_cost = cost_fn(state)
    best = cur_cost
    curve = [best]
    evals = 0

    for step in range(1, max_steps + 1):
        nbrs = neighbors_fn(state)
        improved = False

        for _ in range(max_tries_per_step):
            ns = rng.choice(nbrs)
            evals += 1
            c = cost_fn(ns)
            if c < cur_cost:
                state = ns
                cur_cost = c
                best = min(best, cur_cost)
                improved = True
                break

        curve.append(best)

        if best == 0:
            return RunResult(success=True, steps=step, evals=evals, final_cost=cur_cost, best_curve=curve)

        if not improved:
            # no improvement found in tries => stop
            return RunResult(success=(best == 0), steps=step, evals=evals, final_cost=cur_cost, best_curve=curve)

    return RunResult(success=(best == 0), steps=max_steps, evals=evals, final_cost=cur_cost, best_curve=curve)


def simulated_annealing(
    init_state,
    cost_fn: Callable,
    neighbors_fn: Callable,
    max_steps: int,
    T0: float,
    alpha: float,
    Tmin: float,
    rng: random.Random
) -> RunResult:
    state = init_state
    cur_cost = cost_fn(state)
    best = cur_cost
    curve = [best]
    evals = 0

    T = T0
    for step in range(1, max_steps + 1):
        if T < Tmin or best == 0:
            return RunResult(success=(best == 0), steps=step-1, evals=evals, final_cost=cur_cost, best_curve=curve)

        nbrs = neighbors_fn(state)
        ns = rng.choice(nbrs)
        evals += 1
        ncost = cost_fn(ns)
        delta = ncost - cur_cost

        if delta <= 0:
            state = ns
            cur_cost = ncost
        else:
            # Metropolis criterion
            p = math.exp(-delta / max(T, 1e-12))
            if rng.random() < p:
                state = ns
                cur_cost = ncost

        best = min(best, cur_cost)
        curve.append(best)

        T *= alpha

    return RunResult(success=(best == 0), steps=max_steps, evals=evals, final_cost=cur_cost, best_curve=curve)


# -------------------------
# Experiment runner
# -------------------------

def run_experiment(
    problem: str,
    heuristic: str,
    algo: str,
    runs: int,
    max_steps: int,
    seed: int,
    puzzle_depth: int,
    queens_n: int,
    fc_tries: int,
    sa_T0: float,
    sa_alpha: float,
    sa_Tmin: float
) -> None:
    set_global_seed(seed)
    base_rng = random.Random(seed)

    # Problem-specific setup
    if problem == "puzzle":
        if heuristic == "hamming":
            cost_fn = puzzle_hamming
        elif heuristic == "manhattan":
            cost_fn = puzzle_manhattan
        else:
            raise ValueError("heuristic must be hamming or manhattan for puzzle")

        neighbors_fn = puzzle_neighbors

        def sample_init(rng: random.Random):
            # reverse walk => guaranteed solvable
            return puzzle_generate_by_reverse_walk(puzzle_depth, rng)

    elif problem == "queens":
        cost_fn = queens_cost
        neighbors_fn = queens_neighbors

        def sample_init(rng: random.Random):
            return queens_random_state(queens_n, rng)
    else:
        raise ValueError("problem must be puzzle or queens")

    algos = ["steepest", "first", "sa"] if algo == "all" else [algo]

    all_results: Dict[str, List[RunResult]] = {a: [] for a in algos}

    for a in algos:
        for i in range(runs):
            rng = random.Random(base_rng.randrange(10**18))
            init = sample_init(rng)

            if problem == "puzzle":
                # sanity check
                assert puzzle_is_solvable(init), "Generated puzzle state should be solvable"

            if a == "steepest":
                res = hill_climb_steepest(init, cost_fn, neighbors_fn, max_steps=max_steps)
            elif a == "first":
                res = hill_climb_first_choice(init, cost_fn, neighbors_fn, max_steps=max_steps,
                                              max_tries_per_step=fc_tries, rng=rng)
            elif a == "sa":
                res = simulated_annealing(init, cost_fn, neighbors_fn, max_steps=max_steps,
                                          T0=sa_T0, alpha=sa_alpha, Tmin=sa_Tmin, rng=rng)
            else:
                raise ValueError("algo must be steepest / first / sa / all")

            all_results[a].append(res)

    # Summary metrics
    print("\n=== Summary ===")
    for a in algos:
        rr = all_results[a]
        success_rate = sum(r.success for r in rr) / len(rr)
        avg_evals = float(np.mean([r.evals for r in rr]))
        avg_steps = float(np.mean([r.steps for r in rr]))
        avg_final_cost = float(np.mean([r.final_cost for r in rr]))
        print(f"[{a:8s}] success={success_rate:.3f} | avg_evals={avg_evals:.1f} | avg_steps={avg_steps:.1f} | avg_final_cost={avg_final_cost:.2f}")

    # Curves (best cost vs steps)
    max_curve_len = max(len(r.best_curve) for a in algos for r in all_results[a])
    max_curve_len = min(max_curve_len, max_steps + 1)  # cap

    plt.figure()
    for a in algos:
        curves = [r.best_curve for r in all_results[a]]
        mean_curve = pad_and_average_curves(curves, max_curve_len)
        plt.plot(np.arange(max_curve_len), mean_curve, label=a)
    plt.xlabel("step")
    plt.ylabel("best-so-far cost")
    plt.title(f"{problem} ({heuristic}) - best cost curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{problem}_{heuristic}_best_cost_curve.png")

    # Optional: efficiency-quality scatter (avg evals vs success) if comparing many settings externally
    # Here we just print, because one run doesn't give a sweep.


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--problem", choices=["puzzle", "queens"], required=True)
    p.add_argument("--heuristic", choices=["hamming", "manhattan"], default="manhattan",
                   help="only for puzzle; ignored for queens")
    p.add_argument("--algo", choices=["steepest", "first", "sa", "all"], default="all")
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)

    # puzzle difficulty
    p.add_argument("--puzzle_depth", type=int, default=25,
                   help="reverse-walk depth from goal, roughly controls difficulty")

    # queens size
    p.add_argument("--queens_n", type=int, default=8)

    # first-choice hill climbing
    p.add_argument("--fc_tries", type=int, default=50,
                   help="max random neighbor attempts per step for first-choice HC")

    # simulated annealing params
    p.add_argument("--sa_T0", type=float, default=10.0)
    p.add_argument("--sa_alpha", type=float, default=0.995)
    p.add_argument("--sa_Tmin", type=float, default=1e-3)

    return p


def main():
    args = build_argparser().parse_args()
    run_experiment(
        problem=args.problem,
        heuristic=args.heuristic,
        algo=args.algo,
        runs=args.runs,
        max_steps=args.max_steps,
        seed=args.seed,
        puzzle_depth=args.puzzle_depth,
        queens_n=args.queens_n,
        fc_tries=args.fc_tries,
        sa_T0=args.sa_T0,
        sa_alpha=args.sa_alpha,
        sa_Tmin=args.sa_Tmin
    )


if __name__ == "__main__":
    main()
