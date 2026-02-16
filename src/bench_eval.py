#!/usr/bin/env python3
"""
Use saved test cases (arcs.csv + meta.json) to compare TWO solution algorithms:
  - objective (reliability + smoothed objective + mean Val)
  - final x (saved to disk per solver + instance)
  - solution time (wall clock)
  - evaluation time and #maxflow calls (optional but useful)

Expected instance folder format:
  arcs.csv columns: tail, head, u, p_fail
  meta.json keys: n, s, t, demand_d

Outputs:
  - results.csv (one row per instance per solver)
  - x_solutions/<instance>/<solver>.npy  (final x vector)
  - optional x_solutions/<instance>/<solver>.csv (human-readable)

Run:
  python compare_solvers.py --suite_dir bench_suite_full --out_csv results.csv --mc 1000 --tau 1.0

How to use with your real algorithms:
  Implement Solver.solve(inst, rng) -> x (shape m,), then register TWO solvers in main().
"""

from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import networkx as nx


# =========================
# Data structures
# =========================

@dataclass
class Instance:
    name: str
    folder: str
    n: int
    m: int
    s: int
    t: int
    demand: float
    edges: List[Tuple[int, int]]
    u: np.ndarray
    p: np.ndarray


@dataclass
class EvalStats:
    maxflow_calls: int = 0
    maxflow_time_sec: float = 0.0


# =========================
# Loading
# =========================

def load_instance(folder: str) -> Instance:
    arcs_path = os.path.join(folder, "arcs.csv")
    meta_path = os.path.join(folder, "meta.json")
    if not os.path.exists(arcs_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing arcs.csv or meta.json in {folder}")

    arcs = pd.read_csv(arcs_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    edges = list(zip(arcs["tail"].astype(int).tolist(),
                     arcs["head"].astype(int).tolist()))
    u = arcs["u"].to_numpy(dtype=float)
    p = arcs["p_fail"].to_numpy(dtype=float)

    n = int(meta["n"])
    s = int(meta["s"])
    t = int(meta["t"])
    d = float(meta["demand_d"])

    name = os.path.basename(folder.rstrip("/\\"))
    return Instance(name=name, folder=folder, n=n, m=len(edges), s=s, t=t, demand=d, edges=edges, u=u, p=p)


def list_instance_folders(suite_dir: str) -> List[str]:
    folders = []
    for entry in os.listdir(suite_dir):
        p = os.path.join(suite_dir, entry)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "arcs.csv")) and os.path.exists(os.path.join(p, "meta.json")):
            folders.append(p)
    folders.sort()
    return folders


# =========================
# Scenario sampling + objectives
# =========================

def sample_failures(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample ℓ_e ~ Bernoulli(p_e). True=failed, False=survives."""
    return rng.random(len(p)) < p


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


# =========================
# Second-stage evaluator: Val(x, κ)
# =========================

def val_maxflow(inst: Instance, x: np.ndarray, failed: np.ndarray, stats: Optional[EvalStats] = None) -> float:
    """
    Compute Val(x,κ) = max s-t flow with capacities cap_e=(1-failed_e)*x_e.
    Uses preflow_push.
    """
    t0 = time.perf_counter()

    G = nx.DiGraph()
    G.add_nodes_from(range(inst.n))

    for (u, v), cap, is_failed in zip(inst.edges, x, failed):
        if (not is_failed) and cap > 0:
            G.add_edge(int(u), int(v), capacity=float(cap))

    try:
        v = nx.maximum_flow_value(G, inst.s, inst.t, capacity="capacity", flow_func=nx.algorithms.flow.preflow_push)
    except Exception:
        v = 0.0

    if stats is not None:
        stats.maxflow_calls += 1
        stats.maxflow_time_sec += (time.perf_counter() - t0)

    return float(v)


def estimate_objectives_mc(
    inst: Instance,
    x: np.ndarray,
    mc_samples: int,
    tau: float,
    rng: np.random.Generator,
    stats: Optional[EvalStats] = None
) -> Dict[str, float]:
    """
    Monte Carlo estimates:
      rel_hat    = Pr(Val>=d)
      smooth_hat = E[sigmoid((Val-d)/tau)]
      val_hat    = E[Val]
    """
    d = inst.demand
    inv_tau = 1.0 / max(1e-12, tau)

    hit = 0
    smooth_sum = 0.0
    val_sum = 0.0

    for _ in range(mc_samples):
        failed = sample_failures(inst.p, rng)
        v = val_maxflow(inst, x, failed, stats)
        val_sum += v
        if v >= d:
            hit += 1
        smooth_sum += float(sigmoid((v - d) * inv_tau))

    return {
        "rel_hat": hit / mc_samples,
        "smooth_hat": smooth_sum / mc_samples,
        "val_hat": val_sum / mc_samples,
    }


# =========================
# Solver interface
# =========================

class Solver:
    name: str = "base"

    def solve(self, inst: Instance, rng: np.random.Generator) -> np.ndarray:
        """
        Return x (shape m,) within [0,u].
        If your formulation requires Nx=0 in stage-1, enforce it in your implementation.
        """
        raise NotImplementedError


# ----- Example solver 1: baseline x=u (fast, deterministic)
class FullCapacitySolver(Solver):
    name = "alg_A_x_equals_u"

    def solve(self, inst: Instance, rng: np.random.Generator) -> np.ndarray:
        return inst.u.copy()


# ----- Example solver 2: random scaling (placeholder for your real alg)
class RandomFractionSolver(Solver):
    name = "alg_B_random_frac"

    def __init__(self, a_min: float = 0.5, a_max: float = 1.0):
        self.a_min = a_min
        self.a_max = a_max

    def solve(self, inst: Instance, rng: np.random.Generator) -> np.ndarray:
        alpha = rng.uniform(self.a_min, self.a_max, size=inst.m)
        return alpha * inst.u


# =========================
# Save x solutions
# =========================

def save_x_solution(out_root: str, inst_name: str, solver_name: str, inst: Instance, x: np.ndarray) -> None:
    """
    Save x as .npy (exact) and optionally .csv (readable with edge list).
    """
    inst_dir = os.path.join(out_root, inst_name)
    os.makedirs(inst_dir, exist_ok=True)

    npy_path = os.path.join(inst_dir, f"{solver_name}.npy")
    np.save(npy_path, x.astype(float))

    # also write a readable csv with edges and x
    csv_path = os.path.join(inst_dir, f"{solver_name}.csv")
    df = pd.DataFrame({
        "tail": [u for (u, v) in inst.edges],
        "head": [v for (u, v) in inst.edges],
        "u_cap": inst.u.astype(float),
        "p_fail": inst.p.astype(float),
        "x": x.astype(float),
    })
    df.to_csv(csv_path, index=False)


# =========================
# Compare two solvers over suite
# =========================

def compare_two_solvers(
    suite_dir: str,
    out_csv: str,
    x_out_dir: str,
    solver_A: Solver,
    solver_B: Solver,
    mc_samples: int,
    tau: float,
    base_seed: int,
    limit: Optional[int] = None
) -> pd.DataFrame:
    folders = list_instance_folders(suite_dir)
    if limit is not None:
        folders = folders[:limit]

    rows: List[Dict[str, Any]] = []
    os.makedirs(x_out_dir, exist_ok=True)

    for folder in folders:
        inst = load_instance(folder)

        for solver_idx, solver in enumerate([solver_A, solver_B]):
            # Separate RNG streams for each solver for fairness and reproducibility
            rng_solve = np.random.default_rng(base_seed + 17 * solver_idx)
            rng_eval = np.random.default_rng(base_seed + 999 + 17 * solver_idx)

            # --- Solve and time it ---
            t0 = time.perf_counter()
            x = solver.solve(inst, rng_solve)
            solve_time = time.perf_counter() - t0

            # Safety clip
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.shape != (inst.m,):
                raise ValueError(f"{solver.name} returned x shape {x.shape}, expected {(inst.m,)}")
            x = np.clip(x, 0.0, inst.u)

            # Save final x
            save_x_solution(x_out_dir, inst.name, solver.name, inst, x)

            # --- Evaluate objective and track maxflow time ---
            stats = EvalStats()
            t1 = time.perf_counter()
            est = estimate_objectives_mc(inst, x, mc_samples=mc_samples, tau=tau, rng=rng_eval, stats=stats)
            eval_time = time.perf_counter() - t1

            rows.append({
                "instance": inst.name,
                "solver": solver.name,
                "n": inst.n,
                "m": inst.m,
                "s": inst.s,
                "t": inst.t,
                "demand": inst.demand,
                "mc_samples": mc_samples,
                "tau": tau,
                "solve_time_sec": solve_time,
                "eval_time_sec": eval_time,
                "maxflow_calls": stats.maxflow_calls,
                "maxflow_time_sec": stats.maxflow_time_sec,
                **est,
                "x_saved_npy": os.path.join(x_out_dir, inst.name, f"{solver.name}.npy"),
                "x_saved_csv": os.path.join(x_out_dir, inst.name, f"{solver.name}.csv"),
            })

            print(
                f"[{solver.name}] {inst.name} | n={inst.n} m={inst.m} | "
                f"rel={est['rel_hat']:.3f} smooth={est['smooth_hat']:.3f} val={est['val_hat']:.3f} | "
                f"solve={solve_time:.3f}s eval={eval_time:.3f}s maxflow={stats.maxflow_calls}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Also provide a pivoted comparison table (A vs B) next to results.csv
    try:
        pivot = df.pivot_table(index="instance", columns="solver",
                               values=["rel_hat", "smooth_hat", "val_hat", "solve_time_sec"],
                               aggfunc="first")
        pivot.to_csv(os.path.splitext(out_csv)[0] + "_pivot.csv")
    except Exception:
        pass

    return df


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
    "--suite_dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "inputs", "bench_suite_full"),
    help="Folder with instance subfolders"
    )
    ap.add_argument("--out_csv", type=str, default="results.csv", help="Output results CSV")
    ap.add_argument("--x_out_dir", type=str, default="x_solutions", help="Folder to save x vectors")
    ap.add_argument("--mc", type=int, default=1000, help="MC samples per solver per instance")
    ap.add_argument("--tau", type=float, default=1.0, help="Sigmoid smoothing temperature")
    ap.add_argument("--seed", type=int, default=123, help="Base random seed")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of instances (quick test)")
    args = ap.parse_args()

    # Register EXACTLY TWO solvers to compare.
    # Replace RandomFractionSolver with your second algorithm implementation.
    solver_A = FullCapacitySolver()
    solver_B = RandomFractionSolver(a_min=0.5, a_max=1.0)

    compare_two_solvers(
        suite_dir=args.suite_dir,
        out_csv=args.out_csv,
        x_out_dir=args.x_out_dir,
        solver_A=solver_A,
        solver_B=solver_B,
        mc_samples=args.mc,
        tau=args.tau,
        base_seed=args.seed,
        limit=args.limit
    )

    print(f"\nSaved results to: {args.out_csv}")
    print(f"Saved x vectors under: {args.x_out_dir}/<instance>/<solver>.(npy|csv)")


if __name__ == "__main__":
    main()