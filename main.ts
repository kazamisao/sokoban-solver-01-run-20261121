// Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// Strategy (written from the competition rules + the generator's structure):
// every level is a 1-wide hallway with 1-4 rooms off it, one door each. The
// decomposition is justified by the cut argument, not by the doors: the
// hallway squares from which no box can ever reach a goal are dead, so no
// optimal box path crosses them, and each remaining component holding boxes
// is an independent room. A level's optimum P* is the sum of its rooms'
// optima. Each room is solved in push space with breadth-first search (a
// state is the sorted box cells plus the player's reachable-region
// representative), so the first solved state is push-optimal. Deadlock
// pruning: live squares (reverse-push reachability to a goal) and frozen 2x2
// blocks. A weighted A* with a min-assignment heuristic over reverse push
// distances is the fallback when a breadth-first search exhausts its slice.
// Every search phase gets its own fresh work budget, so a fallback is never
// dead on arrival. All budgeting is by counted work (expansions + flood-fill
// cells + heuristic calls), never by the wall clock, so two runs produce
// byte-identical output.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const DIRS: { dx: number; dy: number; ch: string; push: string }[] = [
  { dx: 0, dy: -1, ch: 'u', push: 'U' },
  { dx: 0, dy: 1, ch: 'd', push: 'D' },
  { dx: -1, dy: 0, ch: 'l', push: 'L' },
  { dx: 1, dy: 0, ch: 'r', push: 'R' },
];

export type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player). */
export function parseLevel(text: string): Level {
  const rows = text.replace(/\r/g, '').split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * Live squares: cells from which a box can still be pushed onto some goal,
 * found by walking backwards from the goals. A box at y can be pushed in
 * direction d onto x = y + d exactly when the player can stand at y - d, so
 * the reverse step from a live cell x reaches y = x - d when y and y - d are
 * open floor.
 */
export function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue: number[] = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const dir of DIRS) {
      const d = dir.dy * w + dir.dx;
      const y = x - d;
      if (y < 0 || y >= wall.length) continue;
      const behind = y - d;
      if (behind < 0 || behind >= wall.length) continue;
      if (!wall[y] && !wall[behind] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood-fill the player's reachable region from `from` with `occupied` in
 * the way, marking it in `seen` with `stamp`. Returns the region's lowest
 * cell (the state representative) and how many cells the fill visited.
 */
export function floodRegion(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): { low: number; cells: number } {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  let cells = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    cells++;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return { low, cells };
}

/** True when the box just pushed to `t` sits in a frozen 2x2 block. */
export function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b])) return true;
    if ((occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/**
 * A state's key: region representative + sorted box cells, as a string.
 * A numeric base-`size` key would overflow the safe-integer range once the
 * box count grows (the whole-level fallback holds up to 16 boxes), silently
 * colliding distinct states, so the key is a string.
 */
export function keyOf(boxes: number[], low: number): string {
  return `${low}:${boxes.join(',')}`;
}

/** Shortest walk from `from` to `to` around walls and boxes, lower LURD. */
export function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const dir of DIRS) {
      const q = p + dir.dy * w + dir.dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = dir.ch;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** Player cell after replaying `moves` (LURD, case ignored) from `from`. */
export function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

/**
 * Reverse push distances: dist[g][cell] = fewest pushes to move a lone box
 * from `cell` onto goal g (other boxes ignored). Same reverse step as
 * liveSquares, run per goal.
 */
export function reverseDistances(level: Level): number[][] {
  const { w, wall, goals } = level;
  const out: number[][] = [];
  for (const g of goals) {
    const dist = new Array<number>(wall.length).fill(Infinity);
    dist[g] = 0;
    const queue: number[] = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi]!;
      for (const dir of DIRS) {
        const d = dir.dy * w + dir.dx;
        const y = x - d;
        if (y < 0 || y >= wall.length) continue;
        const behind = y - d;
        if (behind < 0 || behind >= wall.length) continue;
        if (!wall[y] && !wall[behind] && dist[y] === Infinity) {
          dist[y] = dist[x]! + 1;
          queue.push(y);
        }
      }
    }
    out.push(dist);
  }
  return out;
}

/** All permutations of 0..n-1 (n <= 5 only; larger n uses the greedy bound). */
export function goalsIndex(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const rec = (): void => {
    if (cur.length === n) {
      out.push([...cur]);
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

/**
 * Min-cost one-to-one assignment of boxes to goals over reverse distances.
 * Admissible: each box needs at least its reverse distance to its assigned
 * goal, ignoring the other boxes. For more than 5 boxes the exact assignment
 * is factorial, so the sum of per-box minima is used instead (still a lower
 * bound). Infinity when no finite assignment exists (dead position: prune).
 * Requires boxes.length === dist.length; callers with unequal counts must
 * not use this heuristic (use h = 0 instead).
 */
export function assignmentHeuristic(boxes: number[], dist: number[][], perms: number[][] | null): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (perms === null) {
    // Greedy bound (n > 5): sum of each box's best goal distance.
    let sum = 0;
    for (let i = 0; i < n; i++) {
      let best = Infinity;
      for (let g = 0; g < dist.length; g++) {
        const d = dist[g]![boxes[i]!]!;
        if (d < best) best = d;
      }
      if (best === Infinity) return Infinity;
      sum += best;
    }
    return sum;
  }
  let best = Infinity;
  for (const p of perms) {
    let sum = 0;
    let ok = true;
    for (let i = 0; i < n; i++) {
      const d = dist[p[i]!]![boxes[i]!]!;
      if (d === Infinity) {
        ok = false;
        break;
      }
      sum += d;
    }
    if (ok && sum < best) best = sum;
  }
  return best;
}

export type SolveResult = { pushes: number; moves: string };
export type SearchOutcome = {
  res: SolveResult | null;
  work: number;
  reason: 'solved' | 'budget' | 'states' | 'unsolved' | 'reconstruct';
};

type Succ = { boxes: number[]; player: number; dir: number; key: string };

/**
 * Push-space search over one (sub-)level with its OWN work budget `cap`
 * (fresh per call, so a fallback never inherits an exhausted budget).
 * mode 'bfs' expands in push order (first solved state is push-optimal);
 * mode 'astar' orders by g + 2*h with the assignment heuristic and keeps the
 * best solved state as incumbent, pruning anything with g + h >= incumbent,
 * and stops once the heap's smallest f reaches 2 * incumbentG. Returns the
 * work spent alongside the result so callers can account for it.
 */
export function solvePush(
  level: Level,
  cap: number,
  mode: 'bfs' | 'astar',
  maxStates = 250_000,
): SearchOutcome {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  const budget = { work: 0, cap };
  if (level.boxes.some((b) => !live[b])) return { res: null, work: 0, reason: 'unsolved' };
  if (solved(level.boxes)) return { res: { pushes: 0, moves: '' }, work: 0, reason: 'solved' };

  const dist = mode === 'astar' ? reverseDistances(level) : null;
  if (dist !== null) budget.work += size * goals.length;
  const useH = dist !== null && level.boxes.length === level.goals.length;
  const perms = useH && level.boxes.length <= 5 ? goalsIndex(level.boxes.length) : null;
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const statesKey: string[] = [''];
  const statesG: number[] = [0];
  const parent: number[] = [-1];
  const index = new Map<string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const startFlood = floodRegion(level, occupied, level.player, seen, ++stamp, stack);
  budget.work += startFlood.cells;
  for (const b of level.boxes) occupied[b] = 0;
  const startKey = keyOf(level.boxes, startFlood.low);
  statesKey[0] = startKey;
  index.set(startKey, 0);

  // Successors of state s: every legal push from its reachable region.
  const successors = (s: number): Succ[] => {
    const boxes = statesBoxes[s]!;
    const out: Succ[] = [];
    for (const b of boxes) occupied[b] = 1;
    const here = ++stamp;
    const region = floodRegion(level, occupied, statesPlayer[s]!, seen, here, stack);
    budget.work += region.cells;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const dir = DIRS[di]!;
        const d = dir.dy * w + dir.dx;
        const t = b + d;
        if (seen[b - d] !== here) continue;
        if (t < 0 || t >= size || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenBlock(level, occupied, goal, t)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const next = boxes.slice();
        next[bi] = t;
        next.sort((x, y) => x - y);
        const after = floodRegion(level, occupied, b, mark, ++stamp, stack);
        budget.work += after.cells;
        const key = keyOf(next, after.low);
        occupied[t] = 0;
        occupied[b] = 1;
        out.push({ boxes: next, player: b, dir: di, key });
      }
    }
    for (const b of boxes) occupied[b] = 0;
    return out;
  };

  // Reconstruct the move string for state `found` by walking its push path.
  // Null when a walk unexpectedly fails: the caller treats it as a search
  // failure and falls back instead of crashing the level.
  const reconstruct = (found: number): SolveResult | null => {
    const path: number[] = [];
    for (let s = found; s > 0; s = parent[s]!) path.push(s);
    path.reverse();
    let moves = '';
    let player = level.player;
    let boxes = level.boxes;
    try {
      for (const s of path) {
        const nextBoxes = statesBoxes[s]!;
        const from = boxes.find((b) => !nextBoxes.includes(b))!;
        const to = nextBoxes.find((b) => !boxes.includes(b))!;
        const d = to - from;
        const di = DIRS.findIndex((dir) => dir.dy * w + dir.dx === d);
        for (const b of boxes) occupied[b] = 1;
        moves += walkTo(level, occupied, player, from - d);
        for (const b of boxes) occupied[b] = 0;
        moves += DIRS[di]!.push;
        player = from;
        boxes = nextBoxes;
      }
    } catch {
      return null;
    }
    return { pushes: path.length, moves };
  };

  if (mode === 'bfs') {
    const queue: number[] = [0];
    for (let head = 0; head < queue.length; head++) {
      if (budget.work > budget.cap) return { res: null, work: budget.work, reason: 'budget' };
      const s = queue[head]!;
      budget.work += 1;
      for (const succ of successors(s)) {
        if (index.has(succ.key)) continue;
        if (statesBoxes.length >= maxStates) return { res: null, work: budget.work, reason: 'states' };
        const t = statesBoxes.length;
        index.set(succ.key, t);
        statesKey.push(succ.key);
        statesBoxes.push(succ.boxes);
        statesPlayer.push(succ.player);
        statesG.push(statesG[s]! + 1);
        parent.push(s);
        if (solved(succ.boxes)) {
          const r = reconstruct(t);
          if (r === null) return { res: null, work: budget.work, reason: 'reconstruct' };
          return { res: r, work: budget.work, reason: 'solved' };
        }
        queue.push(t);
      }
    }
    return { res: null, work: budget.work, reason: 'unsolved' };
  }

  // Weighted A*: heap ordered by (g + 2h, h, insertion). The incumbent is the
  // best solved state found so far; the search ends when the heap's smallest
  // f reaches 2 * incumbentG (no better weighted solution can remain), or
  // when the budget runs out (the incumbent is returned if one exists).
  const hOf = (boxes: number[]): number => {
    if (!useH) return 0;
    budget.work += perms !== null ? perms.length : boxes.length;
    return assignmentHeuristic(boxes, dist!, perms);
  };
  const heap: { f: number; h: number; n: number; s: number }[] = [];
  let counter = 0;
  let incumbent: number | null = null;
  let incumbentG = Infinity;
  const bestG = new Map<string, number>([[startKey, 0]]);
  const heapLess = (
    a: { f: number; h: number; n: number; s: number },
    b: { f: number; h: number; n: number; s: number },
  ): boolean =>
    a.f < b.f || (a.f === b.f && (a.h < b.h || (a.h === b.h && a.n < b.n)));
  const heapPush = (e: { f: number; h: number; n: number; s: number }): void => {
    heap.push(e);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): { f: number; h: number; n: number; s: number } => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  const h0 = hOf(level.boxes);
  heapPush({ f: 2 * h0, h: h0, n: counter++, s: 0 });
  while (heap.length > 0) {
    if (budget.work > budget.cap) {
      if (incumbent === null) return { res: null, work: budget.work, reason: 'budget' };
      const r = reconstruct(incumbent);
      return r === null
        ? { res: null, work: budget.work, reason: 'reconstruct' }
        : { res: r, work: budget.work, reason: 'solved' };
    }
    if (incumbent !== null && heap[0]!.f >= 2 * incumbentG) break;
    const { s } = heapPop();
    const g = statesG[s]!;
    if (bestG.get(statesKey[s]!) !== g) continue;
    budget.work += 1;
    if (solved(statesBoxes[s]!)) {
      if (g < incumbentG) {
        incumbent = s;
        incumbentG = g;
      }
      continue;
    }
    for (const succ of successors(s)) {
      const ng = g + 1;
      const prevG = bestG.get(succ.key);
      if (prevG !== undefined && prevG <= ng) continue;
      const h = hOf(succ.boxes);
      if (h === Infinity) continue;
      if (incumbent !== null && ng + h >= incumbentG) continue;
      if (statesBoxes.length >= maxStates) {
        if (incumbent === null) return { res: null, work: budget.work, reason: 'states' };
        const r = reconstruct(incumbent);
        return r === null
          ? { res: null, work: budget.work, reason: 'reconstruct' }
          : { res: r, work: budget.work, reason: 'solved' };
      }
      const t = statesBoxes.length;
      bestG.set(succ.key, ng);
      statesKey.push(succ.key);
      statesBoxes.push(succ.boxes);
      statesPlayer.push(succ.player);
      statesG.push(ng);
      parent.push(s);
      if (solved(succ.boxes)) {
        if (ng < incumbentG) {
          incumbent = t;
          incumbentG = ng;
        }
        continue;
      }
      heapPush({ f: ng + 2 * h, h, n: counter++, s: t });
    }
  }
  if (incumbent === null) return { res: null, work: budget.work, reason: 'unsolved' };
  const r = reconstruct(incumbent);
  return r === null
    ? { res: null, work: budget.work, reason: 'reconstruct' }
    : { res: r, work: budget.work, reason: 'solved' };
}

export type Room = { squares: number[]; boxes: number[]; goals: number[]; done: boolean };

/**
 * Split a level into rooms: cut the hallway row's dead squares (floor the
 * player walks but from which no box reaches a goal) out of the floor; each
 * remaining component holding boxes is a room. Null when the level is not a
 * set of rooms (a box in the hallway, a box on a wall cell, or a component
 * with boxes != goals).
 */
export function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const dir of DIRS) {
        const q = squares[i]! + dir.dy * w + dir.dx;
        if (q < 0 || q >= wall.length) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) {
    if (part[b]! < 0) return null;
    rooms[part[b]!]!.boxes.push(b);
  }
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * True when the player at `player` can still reach the level's start cell
 * with `occupied` in the way (i.e. it can leave the room it just solved and
 * walk the hallway). The generator promises a finished room never shuts the
 * player in; this is the cheap check that catches it if one ever does.
 */
export function canExit(level: Level, occupied: Uint8Array, player: number): boolean {
  if (player < 0 || player >= level.wall.length) return false;
  const seen = new Int32Array(level.wall.length);
  const stack = new Int32Array(level.wall.length);
  const stamp = 1;
  const { low } = floodRegion(level, occupied, player, seen, stamp, stack);
  void low;
  return seen[level.player] === stamp;
}

/**
 * Solve a level room by room, trying the rooms in scan order and then in
 * reverse order (a room that traps the player behind its solved boxes is
 * retried last). Each room's BFS and A* attempts get their own fresh work
 * budget. Null when no ordering solves every room.
 */
export function solveByRooms(
  level: Level,
  roomBfsCap: number,
  roomAstarCap: number,
): { res: SolveResult | null; work: number } {
  const base = roomsOf(level);
  if (base === null) return { res: null, work: 0 };
  let totalWork = 0;
  const orders: Room[][] = [
    base,
    [...base].reverse(),
  ];
  for (const order of orders) {
    const rooms: Room[] = order.map((r) => ({
      squares: r.squares,
      boxes: [...r.boxes],
      goals: [...r.goals],
      done: false,
    }));
    let player = level.player;
    let pushes = 0;
    let moves = '';
    let attemptWork = 0;
    let ok = true;
    for (let ri = 0; ri < rooms.length; ri++) {
      const room = rooms[ri]!;
      const subWall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) subWall[b] = 1;
      }
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall: subWall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      const bfs = solvePush(sub, roomBfsCap, 'bfs');
      attemptWork += bfs.work;
      let solved = bfs.res;
      if (solved === null) {
        const astar = solvePush(sub, roomAstarCap, 'astar');
        attemptWork += astar.work;
        solved = astar.res;
      }
      if (solved === null) {
        ok = false;
        break;
      }
      pushes += solved.pushes;
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      room.done = true;
      if (ri < rooms.length - 1) {
        const occupied = new Uint8Array(level.wall.length);
        for (const other of rooms) {
          for (const b of other.done ? other.goals : other.boxes) occupied[b] = 1;
        }
        if (!canExit(level, occupied, player)) {
          ok = false;
          break;
        }
      }
    }
    totalWork += attemptWork;
    if (ok) return { res: { pushes, moves }, work: totalWork };
  }
  return { res: null, work: totalWork };
}

/**
 * Solve one level's XSB text. Phases, each with its own fresh budget slice
 * of `workCap`: room-by-room BFS (+ A* per room), then whole-level BFS, then
 * whole-level weighted A*. Returns an empty move string when nothing
 * succeeds (still a valid, scored-0 answer). `work` is the total counted
 * work spent, so the caller can budget the rest of the set honestly.
 */
export function solveLevel(
  text: string,
  workCap = 6_000_000,
): SolveResult & { work: number } {
  const level = parseLevel(text);
  if (level.boxes.every((b) => level.goals.includes(b))) return { pushes: 0, moves: '', work: 0 };
  // Phase caps are sized from measured work: one hard room's BFS needs up
  // to ~4M counted work (the generator guarantees at most 250k states, and
  // a state costs ~60 counted work in floods), so the BFS slices are large
  // enough to let the optimal search finish, with A* as the fallback.
  const roomBfsCap = Math.max(4_000_000, workCap * 2);
  const roomAstarCap = Math.max(2_000_000, workCap);
  const wholeBfsCap = Math.max(4_000_000, workCap * 2);
  const wholeAstarCap = Math.max(2_000_000, workCap);
  let work = 0;
  const byRooms = solveByRooms(level, roomBfsCap, roomAstarCap);
  work += byRooms.work;
  if (byRooms.res !== null) return { ...byRooms.res, work };
  const wholeBfs = solvePush(level, wholeBfsCap, 'bfs');
  work += wholeBfs.work;
  if (wholeBfs.res !== null) return { ...wholeBfs.res, work };
  const wholeAstar = solvePush(level, wholeAstarCap, 'astar');
  work += wholeAstar.work;
  if (wholeAstar.res !== null) return { ...wholeAstar.res, work };
  return { pushes: -1, moves: '', work };
}

/** Independent replay checker: legality, solved, push count. */
export function replayCheck(
  text: string,
  moves: string,
): { legal: boolean; solved: boolean; pushes: number; detail: string } {
  const level = parseLevel(text);
  const { w, wall } = level;
  const goal = new Uint8Array(wall.length);
  for (const g of level.goals) goal[g] = 1;
  const box = new Uint8Array(wall.length);
  for (const b of level.boxes) box[b] = 1;
  let player = level.player;
  let pushes = 0;
  for (let k = 0; k < moves.length; k++) {
    const ch = moves[k]!.toLowerCase();
    const dir = DIRS.find((d) => d.ch === ch);
    if (!dir) return { legal: false, solved: false, pushes, detail: `char ${JSON.stringify(moves[k])} at ${k} is no move` };
    const d = dir.dy * w + dir.dx;
    const next = player + d;
    if (next < 0 || next >= wall.length || wall[next]) {
      return { legal: false, solved: false, pushes, detail: `move ${k + 1} walks into a wall` };
    }
    if (box[next]) {
      const beyond = next + d;
      if (beyond < 0 || beyond >= wall.length || wall[beyond] || box[beyond]) {
        return { legal: false, solved: false, pushes, detail: `move ${k + 1} pushes a box into a wall or box` };
      }
      box[next] = 0;
      box[beyond] = 1;
      pushes++;
    }
    player = next;
  }
  for (let i = 0; i < box.length; i++) {
    if (box[i] && !goal[i]) {
      return { legal: true, solved: false, pushes, detail: `ends with a box off its goal after ${pushes} pushes` };
    }
  }
  return { legal: true, solved: true, pushes, detail: `solved in ${pushes} pushes` };
}

function main(): void {
  const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  let ids: string[];
  try {
    ids = readdirSync(I).sort();
  } catch {
    return;
  }
  // Shared work budget across the set: each level gets at most 6M units but
  // never eats the 1M reserve kept per level still to come; actual work
  // spent (not the cap) is subtracted, so an easy level leaves more for
  // the rest.
  let remaining = 30_000_000;
  for (let li = 0; li < ids.length; li++) {
    const id = ids[li]!;
    const levelsLeft = ids.length - li;
    const cap = Math.max(500_000, Math.min(6_000_000, remaining - 1_000_000 * (levelsLeft - 1)));
    let moves = '';
    try {
      const res = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'), cap);
      moves = res.moves;
      remaining -= res.work;
    } catch {
      moves = '';
    }
    try {
      mkdirSync(join(O, id), { recursive: true });
      writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
    } catch {
      // One unwritable output directory must not abort the later levels.
    }
  }
}

const isMain =
  process.argv[1] !== undefined && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) main();
