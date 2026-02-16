# Benchmark Suite Overview

This benchmark suite is designed for **two-stage stochastic s--t flow
problems under independent arc failures**, where algorithms:

-   Optimize a first-stage decision ( x )
-   Evaluate performance via second-stage max-flow recourse

\[ `\mathrm{Val}`{=tex}(x,`\kappa`{=tex}) \]

(e.g., reliability of meeting demand)

------------------------------------------------------------------------

## Sizes and Regimes

Each benchmark instance is defined by:

-   **Network size ( n )**:\
    10, 20, 50, 100, 200, 500

-   **Base failure probability level ( p_0 )**:\
    0.2, 0.5, 0.8

-   **Topology family**:\
    dag, cyclic, grid

-   **Seed**:\
    Typically 1, 2, 3

Total instances:

\[ (#`\text{families}`{=tex}) `\times `{=tex}(#n) `\times `{=tex}(#p_0)
`\times `{=tex}(#`\text{seeds}`{=tex}) \]

------------------------------------------------------------------------

# Topology Families

## 1) dag (Random Directed Acyclic Graphs)

-   Acyclic directed networks (no directed cycles)
-   Weakly connected during generation
-   Single ( s,t ) pair stored in `meta.json`

**Capacity structure**

Edges on many source→sink paths tend to receive higher capacity.

------------------------------------------------------------------------

## 2) cyclic (Random Directed Graphs with Cycles)

-   Directed graphs with cycles allowed
-   Guaranteed backbone path from ( s ) to ( t )

Capacities estimated from sampled short simple ( s `\rightarrow `{=tex}t
) paths.

------------------------------------------------------------------------

## 3) grid (Planar-ish, Locally Meshed)

-   Derived from a 2D grid graph
-   Both directions added to each edge
-   Produces local loops and many alternative routes

Capacity structure similar to cyclic.

------------------------------------------------------------------------

# Failure Model

Independent per-edge failures:

\[ `\ell`{=tex}\_e `\sim `{=tex}`\mathrm{Bernoulli}`{=tex}(p_e) \]

Each instance has base level:

\[ p_0 `\in `{=tex}{0.2, 0.5, 0.8} \]

Per-edge probabilities are randomized around ( p_0 ).

------------------------------------------------------------------------

# Capacity Generation

Each edge has capacity ( u_e ).

Capacities are structured:

-   dag: exact path counts
-   cyclic/grid: sampled short path counts

Mapped to range 5--10 with small random perturbations.

------------------------------------------------------------------------

# Demand and Terminals

Each instance has:

-   Source ( s )
-   Sink ( t )
-   Demand ( d )

Demand calibrated as:

\[ d = `\alpha `{=tex}`\cdot `{=tex}`\mathrm{MaxFlow}`{=tex}(u) \]

where ( `\alpha `{=tex}`\approx 0.8`{=tex} ).

------------------------------------------------------------------------

# File Format

## arcs.csv

-   tail\
-   head\
-   u (capacity ( u_e ))\
-   p_fail (failure probability ( p_e ))

## meta.json

Contains:

-   n, m\
-   s, t\
-   demand_d\
-   base_p\
-   summary statistics\
-   generation parameters

------------------------------------------------------------------------

# Evaluation Procedure

1.  Compute first-stage decision ( x )
2.  Sample failures:

\[ `\ell`{=tex}\_e `\sim `{=tex}`\mathrm{Bernoulli}`{=tex}(p_e) \]

3.  Evaluate:

\[ `\mathrm{Val}`{=tex}(x,`\kappa`{=tex}) \]

using surviving capacities:

\[ (1 - `\ell`{=tex}\_e) x_e \]

------------------------------------------------------------------------

# Repository Location

`../inputs/bench_suite_full`

------------------------------------------------------------------------

# Suitability

-   dag: cut-structure stress\
-   cyclic: rerouting stress\
-   grid: planar meshed networks
