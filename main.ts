// Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// Strategy (written from the competition rules + the generator's structure):
// every level is a 1-wide hallway with 1-4 rooms off it, one door each, and a
// box pushed through a door can never come back, so each room is an independent
// push-optimal problem and a level's optimum P* is the sum of its rooms' optima.
// Each room is solved in push space (a state is the sorted box cells plus the
// player's reachable-region representative), breadth-first so the first solved
// state is push-optimal, pruned by live squares (reverse-push reachability to
// a goal) and by frozen 2x2 blocks. A weighted A* with a min-assignment
// heuristic over reverse push distances is the fallback when a breadth-first
// search exhausts its work slice, so a level still gets a valid (if not
// optimal) answer. All budgeting is by counted work (expansions + flood-fill
// cells), never by the wall clock, so two runs produce byte-identical output.
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
  const rows = text.split('\n').filter((r) => r.length > 0);
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

/** A state's key: region representative followed by the sorted box cells. */
export function keyOf(boxes: number[], low: number, size: number): number {
  let k = low;
  for (const b of boxes) k = k * size + b;
  return k;
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

/**
 * Min-cost one-to-one assignment of boxes to goals over reverse distances
 * (brute-force permutations; at most 4 boxes). Admissible: each box needs at
 * least its reverse distance to its assigned goal, ignoring the other boxes.
 * Infinity when no finite assignment exists (dead position: prune).
 */
export function assignmentHeuristic(boxes: number[], dist: number[][]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  const perm = goalsIndex(n);
  let best = Infinity;
  for (const p of perm) {
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

function goalsIndex(n: number): number[][] {
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

export type WorkBudget = { work: number; cap: number };
export type SolveResult = { pushes: number; moves: string };

type Succ = { boxes: number[]; player: number; dir: number; low: number; key: number };

/**
 * Push-space search over one (sub-)level. mode 'bfs' expands in push order
 * (first solved state is push-optimal); mode 'astar' orders by g + 2*h with
 * the assignment heuristic and keeps the best solved state as incumbent,
 * pruning anything with g + h >= incumbent. Returns null when the work
 * budget runs out or no solution is found within it.
 */
export function solvePush(
  level: Level,
  budget: WorkBudget,
  mode: 'bfs' | 'astar',
  maxStates = 250_000,
): SolveResult | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b])) return null;
  if (solved(level.boxes)) return { pushes: 0, moves: '' };

  const dist = mode === 'astar' ? reverseDistances(level) : null;
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const statesG: number[] = [0];
  const parent: number[] = [-1];
  const pushDir: number[] = [-1];
  const index = new Map<number, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const startLow = floodRegion(level, occupied, level.player, seen, ++stamp, stack).low;
  for (const b of level.boxes) occupied[b] = 0;
  index.set(keyOf(level.boxes, startLow, size), 0);

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
        const low = floodRegion(level, occupied, b, mark, ++stamp, stack).low;
        budget.work += 1;
        const key = keyOf(next, low, size);
        occupied[t] = 0;
        occupied[b] = 1;
        out.push({ boxes: next, player: b, dir: di, low, key });
      }
    }
    for (const b of boxes) occupied[b] = 0;
    return out;
  };

  // Reconstruct the move string for state `found` by walking its push path.
  const reconstruct = (found: number): SolveResult => {
    const path: number[] = [];
    for (let s = found; s > 0; s = parent[s]!) path.push(s);
    path.reverse();
    let moves = '';
    let player = level.player;
    let boxes = level.boxes;
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
    return { pushes: path.length, moves };
  };

  if (mode === 'bfs') {
    const queue: number[] = [0];
    for (let head = 0; head < queue.length; head++) {
      if (budget.work > budget.cap) return null;
      const s = queue[head]!;
      budget.work += 1;
      for (const succ of successors(s)) {
        if (index.has(succ.key)) continue;
        if (statesBoxes.length >= maxStates) return null;
        const t = statesBoxes.length;
        index.set(succ.key, t);
        statesBoxes.push(succ.boxes);
        statesPlayer.push(succ.player);
        statesG.push(statesG[s]! + 1);
        parent.push(s);
        pushDir.push(succ.dir);
        if (solved(succ.boxes)) return reconstruct(t);
        queue.push(t);
      }
    }
    return null;
  }

  // Weighted A*: heap ordered by (g + 2h, h, insertion). First solved pop is
  // optimal for the weighted objective; the incumbent path is kept so a
  // budget cut still returns a valid solution when one was found.
  const hOf = (boxes: number[]): number => assignmentHeuristic(boxes, dist!);
  const heap: { f: number; h: number; n: number; s: number }[] = [];
  let counter = 0;
  let incumbent: number | null = null;
  let incumbentG = Infinity;
  const bestG = new Map<number, number>([[keyOf(level.boxes, startLow, size), 0]]);
  const heapPush = (e: { f: number; h: number; n: number; s: number }): void => {
    heap.push(e);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      const a = heap[i]!;
      const b = heap[p]!;
      if (a.f > b.f || (a.f === b.f && (a.h > b.h || (a.h === b.h && a.n > b.n)))) break;
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
        const left = heap[c]!;
        const right = heap[c + 1];
        if (right && (right.f < left.f || (right.f === left.f && (right.h < left.h || (right.h === left.h && right.n < left.n))))) c++;
        const child = heap[c]!;
        const cur = heap[i]!;
        if (child.f > cur.f || (child.f === cur.f && (child.h > cur.h || (child.h === cur.h && child.n > cur.n)))) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  heapPush({ f: 2 * hOf(level.boxes), h: hOf(level.boxes), n: counter++, s: 0 });
  while (heap.length > 0) {
    if (budget.work > budget.cap) return incumbent === null ? null : reconstruct(incumbent);
    const top = heap[0]!;
    if (incumbent !== null && top.f >= 2 * incumbentG) break;
    const { s } = heapPop();
    const g = statesG[s]!;
    if (bestG.get(keyOf(statesBoxes[s]!, floodRegion(level, (() => {
      for (const b of statesBoxes[s]!) occupied[b] = 1;
      return occupied;
    })(), statesPlayer[s]!, seen, ++stamp, stack).low, size)) !== g) {
      for (const b of statesBoxes[s]!) occupied[b] = 0;
      continue;
    }
    for (const b of statesBoxes[s]!) occupied[b] = 0;
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
      if (statesBoxes.length >= maxStates) return incumbent === null ? null : reconstruct(incumbent);
      const t = statesBoxes.length;
      bestG.set(succ.key, ng);
      index.set(succ.key, t);
      statesBoxes.push(succ.boxes);
      statesPlayer.push(succ.player);
      statesG.push(ng);
      parent.push(s);
      pushDir.push(succ.dir);
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
  return incumbent === null ? null : reconstruct(incumbent);
}

export type Room = { squares: number[]; boxes: number[]; goals: number[]; done: boolean };

/**
 * Split a level into rooms: cut the hallway row's dead squares (floor the
 * player walks but from which no box reaches a goal) out of the floor; each
 * remaining component holding boxes is a room. Null when the level is not a
 * set of rooms (a box in the hallway, or a component with boxes != goals).
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
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve a level room by room; null when any room's search gives up. */
export function solveByRooms(level: Level, budget: WorkBudget): SolveResult | null {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
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
    const solved = solvePush(sub, budget, 'bfs') ?? solvePush(sub, budget, 'astar');
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

/**
 * Solve one level's XSB text. Tries room decomposition first (push-optimal
 * per room), then whole-level BFS, then whole-level weighted A*; returns an
 * empty move string when nothing succeeds (still a valid, scored-0 answer).
 */
export function solveLevel(text: string, workCap = 6_000_000): SolveResult {
  const level = parseLevel(text);
  if (level.boxes.every((b) => level.goals.includes(b))) return { pushes: 0, moves: '' };
  const budget: WorkBudget = { work: 0, cap: workCap };
  const byRooms = solveByRooms(level, budget);
  if (byRooms !== null) return byRooms;
  const whole = solvePush(level, budget, 'bfs') ?? solvePush(level, budget, 'astar');
  if (whole !== null) return whole;
  return { pushes: -1, moves: '' };
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
  const ids = readdirSync(I).sort();
  // Shared work budget across the set: each level gets at most 6M units but
  // never eats the 1M reserve kept per level still to come.
  let remaining = 30_000_000;
  for (let li = 0; li < ids.length; li++) {
    const id = ids[li]!;
    const levelsLeft = ids.length - li;
    const cap = Math.max(500_000, Math.min(6_000_000, remaining - 1_000_000 * (levelsLeft - 1)));
    let moves = '';
    try {
      const res = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'), cap);
      moves = res.moves;
      remaining -= Math.min(cap, 6_000_000);
    } catch {
      moves = '';
    }
    mkdirSync(join(O, id), { recursive: true });
    writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
  }
}

const isMain =
  process.argv[1] !== undefined && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) main();
