# Sokoban solver — make.fun sokoban-solver-01-run-20261121

One Node 22 program (`main.ts`, no dependencies) that answers a set of 5
Sokoban levels in a single run: it reads `/input/<id>/level.txt` (or
`$MAKEFUN_INPUT_DIR`) and writes `/output/<id>/solution.txt` (or
`$MAKEFUN_OUTPUT_DIR`) as LURD moves.

## How it works

Every level in this competition is a 1-wide hallway with 1–4 rooms off it,
one door each. A box pushed through a door can never return, so each room is
an independent push-optimal problem and a level's optimum P\* is the sum of
its rooms' optima. The solver:

1. **Decomposes** the level into rooms by cutting the hallway row's dead
   squares (floor from which no box can reach a goal) out of the floor.
2. **Solves each room in push space** with breadth-first search (state =
   sorted box cells + the player's reachable-region representative), so the
   first solved state is push-optimal. Deadlock pruning: live squares
   (reverse-push reachability to a goal) and frozen 2×2 blocks.
3. **Falls back** to weighted A* (g + 2h, min-assignment heuristic over
   reverse push distances) if a breadth-first search exhausts its work slice,
   so a level still gets a valid answer.
4. **Replays** the push path into walks (shortest BFS walk to each push
   square, fixed direction order U/D/L/R) and concatenates the rooms' moves.

All budgeting is by counted work (expansions + flood-fill cells), never by
the wall clock, so two runs produce byte-identical output. No network, no
clock reads.

## Practice results

| level | pushes | P\* |
|-------|--------|----|
| i001  | 13     | 13 |
| i002  | 16     | 16 |
| i003  | 29     | 29 |
| i004  | 47     | 47 |
| i005  | 60     | 60 |

Practice score **1,000,000** (whole practice set in ~1 s on this machine).
