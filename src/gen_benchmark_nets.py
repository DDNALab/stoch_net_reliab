"""
benchmark networks generator: DAG + cyclic random digraph + grid/planar-like digraph
+ per-edge randomness in BOTH capacities u_e and failure probabilities p_e.

Generates instances for:
  n ∈ {10, 20, 50, 100, 200, 500}
  base failure level p0 ∈ {0.2, 0.5, 0.8}
  topology_family ∈ {dag, cyclic, grid}
  seeds ∈ {1,2,3} (editable)

Per-edge randomness:
  - Capacities: derived from edge "importance" counts, then multiplicative lognormal jitter,
               then clipped to [capacity_LB, capacity_UB] and cast to int.
  - Failure probs: p_e = clip(p0 * exp(N(0, p_sigma)), p_min, p_max)
                   (so p_e is positive and centered around p0).
    (If you prefer additive noise, swap to p_e = clip(p0 + N(0, sigma), ...).)

Demand:
  d = demand_frac * intact_maxflow(u)   (default demand_frac=0.8)

Outputs per instance folder:
  arcs.csv: tail, head, u, p_fail
  meta.json: includes generation params and summary statistics

Run:
  python gen_benchmark_nets.py
"""

from __future__ import annotations

import os
import json
import time
import math
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# 0) Utilities
# ============================================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_nx_graph(n: int, edges: List[Tuple[int, int]], caps: np.ndarray) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (u, v), c in zip(edges, caps):
        G.add_edge(int(u), int(v), capacity=float(c))
    return G


def compute_intact_maxflow(G: nx.DiGraph, s: int, t: int) -> float:
    try:
        val, _ = nx.maximum_flow(G, s, t, capacity="capacity")
        return float(val)
    except Exception:
        return 0.0


# ============================================================
# 1) Capacity assignment primitives (with added randomness)
# ============================================================

def capacities_from_edge_counts(
    edges: List[Tuple[int, int]],
    edge_count: Dict[Tuple[int, int], int],
    capacity_LB: int,
    capacity_UB: int,
    STD_MEAN_RATIO: float,
    cap_lognormal_sigma: float
) -> np.ndarray:
    """
    Convert edge usage counts -> base capacities in [LB,UB], then add extra multiplicative jitter.

    Base mean caps:
      mu_e = capacity_UB * count(e)/max_count, clipped to >= LB
    Base noise:
      N(mu_e, STD_MEAN_RATIO * mu_e)
    Extra jitter:
      multiply by exp(N(0, cap_lognormal_sigma))

    Finally clip to [LB,UB] and cast to int.
    """
    counts = np.array([edge_count.get(e, 0) for e in edges], dtype=float)
    max_count = float(np.max(counts)) if len(counts) else 1.0
    if max_count <= 0:
        max_count = 1.0

    mean_caps = capacity_UB * counts / max_count
    mean_caps = np.maximum(mean_caps, float(capacity_LB))

    # Base Gaussian noise around mean
    caps = np.random.normal(mean_caps, STD_MEAN_RATIO * mean_caps)

    # Extra multiplicative jitter (lognormal centered at 1)
    if cap_lognormal_sigma > 0:
        mult = np.exp(np.random.normal(0.0, cap_lognormal_sigma, size=len(caps)))
        caps = caps * mult

    caps = np.rint(caps).astype(int)
    caps = np.clip(caps, capacity_LB, capacity_UB)
    return caps


def edge_counts_from_all_source_sink_paths_dag(
    dag: nx.DiGraph,
    sources: List[int],
    sinks: List[int]
) -> Dict[Tuple[int, int], int]:
    """
    Exact edge counts across ALL simple paths for every (source, sink) pair (DAG only).
    """
    memo_paths: Dict[Tuple[int, int], List[List[int]]] = {}

    def get_paths(source: int, sink: int) -> List[List[int]]:
        key = (source, sink)
        if key not in memo_paths:
            memo_paths[key] = list(nx.all_simple_paths(dag, source=source, target=sink))
        return memo_paths[key]

    edge_count = defaultdict(int)
    for s in sources:
        for t in sinks:
            for path in get_paths(s, t):
                for u, v in zip(path, path[1:]):
                    edge_count[(u, v)] += 1

    return {e: int(edge_count.get(e, 0)) for e in dag.edges()}


def edge_counts_from_k_shortest_simple_paths(
    G: nx.DiGraph,
    s: int,
    t: int,
    K: int,
    max_path_len: Optional[int]
) -> Dict[Tuple[int, int], int]:
    """
    Approximate edge counts across first K shortest simple s→t paths (cyclic / grid).
    """
    edge_count = defaultdict(int)

    try:
        gen = nx.shortest_simple_paths(G, s, t)
        taken = 0
        for path in gen:
            if max_path_len is not None and len(path) > max_path_len:
                continue
            for u, v in zip(path, path[1:]):
                edge_count[(u, v)] += 1
            taken += 1
            if taken >= K:
                break
    except Exception:
        # fallback: single shortest path if possible
        try:
            path = nx.shortest_path(G, s, t)
            for u, v in zip(path, path[1:]):
                edge_count[(u, v)] += 1
        except Exception:
            pass

    return {e: int(edge_count.get(e, 0)) for e in G.edges()}


# ============================================================
# 2) Topology generators
# ============================================================

def get_source_sink_intermediate(dag: nx.DiGraph) -> Tuple[List[int], List[int], List[int]]:
    nodes = list(dag.nodes())
    sources = [n for n in nodes if dag.in_degree(n) == 0]
    sinks = [n for n in nodes if dag.out_degree(n) == 0]
    intermediates = list(set(nodes) - set(sources) - set(sinks))
    return sources, sinks, intermediates


def default_edge_count_dag(n: int) -> int:
    # Conservative (exact all_simple_paths)
    if n <= 20:
        return 3 * n
    if n <= 100:
        return 2 * n
    if n <= 200:
        return int(1.5 * n)
    return n  # n=500 -> 500 edges


def default_edge_count_cyclic(n: int) -> int:
    # Can be denser; still moderate
    if n <= 20:
        return 4 * n
    if n <= 100:
        return 3 * n
    if n <= 200:
        return 2 * n
    return int(1.5 * n)


def gen_connected_dag(num_nodes: int, num_edges: int) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_nodes))

    while dag.number_of_edges() < num_edges:
        u, v = random.sample(range(num_nodes), 2)
        if (not dag.has_edge(u, v)) and (not nx.has_path(dag, v, u)):
            dag.add_edge(u, v)

    comps = list(nx.weakly_connected_components(dag))
    if len(comps) > 1:
        largest = max(comps, key=len)
        for comp in comps:
            if comp == largest:
                continue
            comp_nodes = list(comp)
            tries = 0
            while tries < 200:
                a = random.choice(list(largest))
                b = random.choice(comp_nodes)
                if not nx.has_path(dag, b, a):
                    dag.add_edge(a, b)
                    break
                tries += 1

    return dag


def pick_st_pair_dag(sources: List[int], sinks: List[int]) -> Tuple[int, int]:
    s = int(min(sources)) if sources else 0
    t = int(max(sinks)) if sinks else max(1, s + 1)
    if s == t:
        t = max(1, t)
    return s, t


def gen_cyclic_random_digraph(n: int, m: int) -> nx.DiGraph:
    """
    Cyclic directed graph with guaranteed s→t reachability via backbone chain.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    while G.number_of_edges() < m:
        u, v = random.sample(range(n), 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
    return G


def gen_grid_like_digraph(n: int) -> nx.DiGraph:
    """
    Planar-ish grid, then bidirect edges to create local cycles.
    """
    a = int(math.floor(math.sqrt(n)))
    b = int(math.ceil(n / max(1, a)))

    UG = nx.grid_2d_graph(a, b)
    mapping = {node: idx for idx, node in enumerate(UG.nodes())}
    UG = nx.relabel_nodes(UG, mapping)

    keep = list(range(min(n, UG.number_of_nodes())))
    UG = UG.subgraph(keep).copy()

    G = nx.DiGraph()
    G.add_nodes_from(UG.nodes())
    for u, v in UG.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)
    return G


# ============================================================
# 3) Failure probability generator (per-edge randomness)
# ============================================================

def gen_edge_failure_probs(
    m: int,
    base_p: float,
    p_lognormal_sigma: float,
    p_min: float,
    p_max: float,
    scheme: str = "lognormal_mult"
) -> np.ndarray:
    """
    Generate per-edge failure probabilities p_e with mean around base_p.

    scheme:
      - "lognormal_mult": p_e = clip(base_p * exp(N(0, sigma)), p_min, p_max)
      - "additive":       p_e = clip(base_p + N(0, sigma), p_min, p_max)
    """
    if scheme == "lognormal_mult":
        mult = np.exp(np.random.normal(0.0, p_lognormal_sigma, size=m))
        p = base_p * mult
    elif scheme == "additive":
        p = base_p + np.random.normal(0.0, p_lognormal_sigma, size=m)
    else:
        raise ValueError(f"Unknown p randomness scheme: {scheme}")

    p = np.clip(p, p_min, p_max)
    return p.astype(float)


# ============================================================
# 4) Topology-specific instance builders
# ============================================================

def gen_dag_instance(
    n: int,
    capacity_LB: int,
    capacity_UB: int,
    STD_MEAN_RATIO: float,
    cap_lognormal_sigma: float
) -> Tuple[List[Tuple[int, int]], np.ndarray, int, int, Dict[str, Any], float, float]:
    m = default_edge_count_dag(n)

    t0 = time.time()
    dag = gen_connected_dag(n, m)
    sources, sinks, _ = get_source_sink_intermediate(dag)

    edge_count = edge_counts_from_all_source_sink_paths_dag(dag, sources, sinks)
    edges = list(dag.edges())
    caps = capacities_from_edge_counts(edges, edge_count, capacity_LB, capacity_UB, STD_MEAN_RATIO, cap_lognormal_sigma)
    gen_time = time.time() - t0

    s, t = pick_st_pair_dag(sources, sinks)
    f0 = compute_intact_maxflow(build_nx_graph(n, edges, caps), s, t)

    gen_kwargs = {
        "n": n,
        "m_target": m,
        "capacity_LB": capacity_LB,
        "capacity_UB": capacity_UB,
        "STD_MEAN_RATIO": STD_MEAN_RATIO,
        "cap_lognormal_sigma": cap_lognormal_sigma,
        "capacity_method": "DAG exact all simple paths across all (source,sink) pairs"
    }
    return edges, caps, s, t, gen_kwargs, gen_time, f0


def gen_cyclic_instance(
    n: int,
    capacity_LB: int,
    capacity_UB: int,
    STD_MEAN_RATIO: float,
    cap_lognormal_sigma: float,
    K_paths: int,
    max_path_len: Optional[int]
) -> Tuple[List[Tuple[int, int]], np.ndarray, int, int, Dict[str, Any], float, float]:
    m = default_edge_count_cyclic(n)

    t0 = time.time()
    G = gen_cyclic_random_digraph(n, m)
    s, t = 0, n - 1

    edge_count = edge_counts_from_k_shortest_simple_paths(G, s, t, K=K_paths, max_path_len=max_path_len)
    edges = list(G.edges())
    caps = capacities_from_edge_counts(edges, edge_count, capacity_LB, capacity_UB, STD_MEAN_RATIO, cap_lognormal_sigma)
    gen_time = time.time() - t0

    f0 = compute_intact_maxflow(build_nx_graph(n, edges, caps), s, t)

    gen_kwargs = {
        "n": n,
        "m_target": m,
        "capacity_LB": capacity_LB,
        "capacity_UB": capacity_UB,
        "STD_MEAN_RATIO": STD_MEAN_RATIO,
        "cap_lognormal_sigma": cap_lognormal_sigma,
        "K_paths": K_paths,
        "max_path_len": max_path_len,
        "capacity_method": "Cyclic: edge frequency in K shortest simple s→t paths"
    }
    return edges, caps, s, t, gen_kwargs, gen_time, f0


def gen_grid_instance(
    n: int,
    capacity_LB: int,
    capacity_UB: int,
    STD_MEAN_RATIO: float,
    cap_lognormal_sigma: float,
    K_paths: int,
    max_path_len: Optional[int]
) -> Tuple[List[Tuple[int, int]], np.ndarray, int, int, Dict[str, Any], float, float]:
    t0 = time.time()
    G = gen_grid_like_digraph(n)
    n_eff = G.number_of_nodes()
    s, t = 0, max(G.nodes())

    edge_count = edge_counts_from_k_shortest_simple_paths(G, s, t, K=K_paths, max_path_len=max_path_len)
    edges = list(G.edges())
    caps = capacities_from_edge_counts(edges, edge_count, capacity_LB, capacity_UB, STD_MEAN_RATIO, cap_lognormal_sigma)
    gen_time = time.time() - t0

    f0 = compute_intact_maxflow(build_nx_graph(n_eff, edges, caps), s, t)

    gen_kwargs = {
        "n_requested": n,
        "n_effective": n_eff,
        "m_effective": len(edges),
        "capacity_LB": capacity_LB,
        "capacity_UB": capacity_UB,
        "STD_MEAN_RATIO": STD_MEAN_RATIO,
        "cap_lognormal_sigma": cap_lognormal_sigma,
        "K_paths": K_paths,
        "max_path_len": max_path_len,
        "capacity_method": "Grid: edge frequency in K shortest simple s→t paths (bidirected grid)"
    }
    return edges, caps, s, t, gen_kwargs, gen_time, f0


# ============================================================
# 5) Main benchmark suite
# ============================================================

def generate_suite(
    out_dir: str = "bench_suite_full",
    node_sizes: Tuple[int, ...] = (10, 20, 50, 100, 200, 500),
    base_p_list: Tuple[float, ...] = (0.2, 0.5, 0.8),
    demand_frac: float = 0.75,
    seeds: Tuple[int, ...] = (1, 2, 3),
    topology_families: Tuple[str, ...] = ("dag", "cyclic", "grid"),

    # capacity parameters
    capacity_LB: int = 5,
    capacity_UB: int = 10,
    STD_MEAN_RATIO: float = 0.10,
    cap_lognormal_sigma: float = 0.15,   # extra multiplicative randomness in capacities

    # failure probability randomness
    p_random_scheme: str = "lognormal_mult",  # or "additive"
    p_lognormal_sigma: float = 0.15,
    p_min: float = 0.01,
    p_max: float = 0.99,

    # cyclic/grid capacity estimation (approx)
    K_paths: int = 2000,
    max_path_len_factor: float = 6.0
) -> None:
    """
    demand_frac: d = demand_frac * intact_maxflow(u)
    max_path_len for cyclic/grid path sampling: int(max_path_len_factor*sqrt(n))+10
    """
    ensure_dir(out_dir)

    for seed in seeds:
        set_seeds(seed)

        for n in node_sizes:
            max_path_len = int(max_path_len_factor * math.sqrt(n)) + 10

            for fam in topology_families:
                if fam == "dag":
                    edges, caps, s, t, gen_kwargs, gen_time, f0 = gen_dag_instance(
                        n=n,
                        capacity_LB=capacity_LB,
                        capacity_UB=capacity_UB,
                        STD_MEAN_RATIO=STD_MEAN_RATIO,
                        cap_lognormal_sigma=cap_lognormal_sigma
                    )
                    n_eff = n

                elif fam == "cyclic":
                    edges, caps, s, t, gen_kwargs, gen_time, f0 = gen_cyclic_instance(
                        n=n,
                        capacity_LB=capacity_LB,
                        capacity_UB=capacity_UB,
                        STD_MEAN_RATIO=STD_MEAN_RATIO,
                        cap_lognormal_sigma=cap_lognormal_sigma,
                        K_paths=K_paths,
                        max_path_len=max_path_len
                    )
                    n_eff = n

                elif fam == "grid":
                    edges, caps, s, t, gen_kwargs, gen_time, f0 = gen_grid_instance(
                        n=n,
                        capacity_LB=capacity_LB,
                        capacity_UB=capacity_UB,
                        STD_MEAN_RATIO=STD_MEAN_RATIO,
                        cap_lognormal_sigma=cap_lognormal_sigma,
                        K_paths=K_paths,
                        max_path_len=max_path_len
                    )
                    n_eff = gen_kwargs.get("n_effective", n)

                else:
                    raise ValueError(f"Unknown topology family: {fam}")

                # Demand calibrated to intact maxflow
                d = float(demand_frac * f0)

                # Now create one instance per base failure level, with per-edge randomness
                m_eff = len(edges)

                for base_p in base_p_list:
                    p_vec = gen_edge_failure_probs(
                        m=m_eff,
                        base_p=float(base_p),
                        p_lognormal_sigma=float(p_lognormal_sigma),
                        p_min=float(p_min),
                        p_max=float(p_max),
                        scheme=p_random_scheme
                    )

                    folder = os.path.join(out_dir, f"{fam}_n{n}_p{base_p:.1f}_seed{seed}")
                    ensure_dir(folder)

                    # write arcs with per-edge p
                    arcs = pd.DataFrame({
                        "tail": [int(u) for (u, v) in edges],
                        "head": [int(v) for (u, v) in edges],
                        "u": caps.astype(float),
                        "p_fail": p_vec.astype(float),
                    })
                    arcs.to_csv(os.path.join(folder, "arcs.csv"), index=False)

                    meta = {
                        "topology_family": fam,
                        "n": int(n_eff),
                        "m": int(m_eff),
                        "s": int(s),
                        "t": int(t),
                        "base_p": float(base_p),
                        "p_random_scheme": p_random_scheme,
                        "p_lognormal_sigma": float(p_lognormal_sigma),
                        "p_min": float(p_min),
                        "p_max": float(p_max),
                        "p_summary": {
                            "mean": float(np.mean(p_vec)),
                            "std": float(np.std(p_vec)),
                            "min": float(np.min(p_vec)),
                            "max": float(np.max(p_vec)),
                        },
                        "capacity_summary": {
                            "mean": float(np.mean(caps)),
                            "std": float(np.std(caps)),
                            "min": float(np.min(caps)),
                            "max": float(np.max(caps)),
                        },
                        "capacity_LB": int(capacity_LB),
                        "capacity_UB": int(capacity_UB),
                        "STD_MEAN_RATIO": float(STD_MEAN_RATIO),
                        "cap_lognormal_sigma": float(cap_lognormal_sigma),
                        "demand_d": float(d),
                        "demand_frac_of_intact_maxflow": float(demand_frac),
                        "intact_maxflow": float(f0),
                        "seed": int(seed),
                        "gen_time_sec": float(gen_time),
                        "gen_kwargs": gen_kwargs,
                    }
                    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)

                    print(
                        f"Wrote {folder} | fam={fam} | n={n_eff}, m={m_eff} | "
                        f"s={s}, t={t} | intact_maxflow={f0:.3g} | d={d:.3g} | "
                        f"base_p={base_p} | p_mean={np.mean(p_vec):.3g} | gen_time={gen_time:.2f}s"
                    )


if __name__ == "__main__":
    generate_suite()