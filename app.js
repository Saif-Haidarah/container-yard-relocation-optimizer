// ============================================================
// Stage 1: Container Yard Core Model, Placement Rules & Moves
// ============================================================

const WEIGHT_RANK = {
  light: 1,
  medium: 2,
  heavy: 3
};

/**
 * Builds normalized rules for the yard.
 * @param {Object} [overrides]
 * @returns {{ H: number, enforceWeightRule: boolean }}
 */
function createRules(overrides = {}) {
  return {
    H: Number.isInteger(overrides.H) && overrides.H > 0 ? overrides.H : 3,
    enforceWeightRule: Boolean(overrides.enforceWeightRule)
  };
}

/**
 * Creates a container object.
 * @param {Object} spec
 */
function createContainer(spec) {
  const weightClass = WEIGHT_RANK[spec.weightClass] ? spec.weightClass : "medium";
  return {
    id: String(spec.id),
    weightClass,
    containerType: spec.containerType || "dry",
    pickupOrder: Number(spec.pickupOrder)
  };
}

/**
 * Single source of truth for whether a container can be placed on top of a stack.
 * When enforceWeightRule is true, a heavier container cannot sit on a lighter one.
 * @param {Array<Object>} stack
 * @param {Object} container
 * @param {{ H: number, enforceWeightRule: boolean }} rules
 * @returns {boolean}
 */
function canPlace(stack, container, rules) {
  if (!Array.isArray(stack) || !container || !rules) return false;
  if (stack.length >= rules.H) return false;
  if (stack.length === 0) return true;
  if (!rules.enforceWeightRule) return true;

  const topContainer = stack[stack.length - 1];
  return WEIGHT_RANK[container.weightClass] <= WEIGHT_RANK[topContainer.weightClass];
}

/**
 * Creates initial yard state with nextNeeded starting at 1.
 * @param {Array<Array<Object>>} stacks
 * @returns {{ stacks: Array<Array<Object>>, nextNeeded: number }}
 */
function initialState(stacks) {
  return {
    stacks: stacks.map((stack) => stack.slice()),
    nextNeeded: 1
  };
}

/**
 * Deep-clones the mutable state structure.
 * @param {{ stacks: Array<Array<Object>>, nextNeeded: number }} state
 */
function cloneState(state) {
  return {
    stacks: state.stacks.map((stack) => stack.slice()),
    nextNeeded: state.nextNeeded
  };
}

/**
 * Removes any top containers matching state.nextNeeded in sequence.
 * Retrieval does NOT count as a relocation.
 * @param {{ stacks: Array<Array<Object>>, nextNeeded: number }} state
 * @returns {{ state: { stacks: Array<Array<Object>>, nextNeeded: number }, retrievedIds: Array<string> }}
 */
function autoRetrieve(state) {
  const next = cloneState(state);
  const retrievedIds = [];
  let progressed = true;

  while (progressed) {
    progressed = false;
    for (let i = 0; i < next.stacks.length; i++) {
      const stack = next.stacks[i];
      if (stack.length === 0) continue;

      const top = stack[stack.length - 1];
      if (top.pickupOrder === next.nextNeeded) {
        stack.pop();
        retrievedIds.push(top.id);
        next.nextNeeded += 1;
        progressed = true;
      }
    }
  }

  return { state: next, retrievedIds };
}

/**
 * Executes one legal relocation from the top of `fromIdx` to `toIdx`.
 * Returns null if the move violates stack bounds, move rules, or placement rules.
 * @param {{ stacks: Array<Array<Object>>, nextNeeded: number }} state
 * @param {number} fromIdx
 * @param {number} toIdx
 * @param {{ H: number, enforceWeightRule: boolean }} rules
 * @returns {{ nextState: { stacks: Array<Array<Object>>, nextNeeded: number }, movedContainer: Object } | null}
 */
function applyRelocation(state, fromIdx, toIdx, rules) {
  const stackCount = state.stacks.length;
  if (fromIdx === toIdx) return null;
  if (fromIdx < 0 || fromIdx >= stackCount || toIdx < 0 || toIdx >= stackCount) {
    return null;
  }

  const sourceStack = state.stacks[fromIdx];
  const targetStack = state.stacks[toIdx];
  if (sourceStack.length === 0) return null;

  const movingContainer = sourceStack[sourceStack.length - 1];
  if (!canPlace(targetStack, movingContainer, rules)) {
    return null;
  }

  const next = cloneState(state);
  next.stacks[fromIdx].pop();
  next.stacks[toIdx].push(movingContainer);

  return {
    nextState: next,
    movedContainer: movingContainer
  };
}

/**
 * Returns true when all stacks are empty.
 * @param {{ stacks: Array<Array<Object>>, nextNeeded: number }} state
 * @returns {boolean}
 */
function isSolved(state) {
  return state.stacks.every((stack) => stack.length === 0);
}
// ============================================================
// Stage 2: Seeded Generator, Solvers (Naive, Heuristic, Optimal) & Self-Check
// ============================================================

const CONTAINER_TYPES = ["dry", "reefer", "hazmat"];
const WEIGHT_CLASSES = ["light", "medium", "heavy"];

/**
 * Deterministic 32-bit PRNG (Mulberry32) so any seed reproduces identical scenarios.
 * @param {number} seed
 * @returns {() => number}
 */
function createRng(seed) {
  let state = (Number(seed) >>> 0) || 1;
  return function nextFloat() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomInt(rng, min, max) {
  return min + Math.floor(rng() * (max - min + 1));
}

function randomPick(rng, list) {
  return list[Math.floor(rng() * list.length)];
}

/**
 * Generates shuffled pickup orders 1..count.
 */
function buildPickupOrders(rng, count) {
  const orders = Array.from({ length: count }, (_, idx) => idx + 1);
  for (let i = orders.length - 1; i > 0; i--) {
    const j = randomInt(rng, 0, i);
    const tmp = orders[i];
    orders[i] = orders[j];
    orders[j] = tmp;
  }
  return orders;
}

/**
 * Bounded feasibility search to verify whether a solvable move sequence exists.
 * Returns "feasible", "infeasible", or "limit reached".
 */
function checkFeasibility(scenario, options = {}) {
  const maxStates = options.maxStates || 4000;
  const start = autoRetrieve(initialState(scenario.stacks)).state;
  if (isSolved(start)) return "feasible";

  const queue = [start];
  const visited = new Set([canonicalStateKey(start)]);
  let explored = 0;

  while (queue.length > 0) {
    const current = queue.shift();
    explored++;
    if (explored > maxStates) return "limit reached";

    for (let from = 0; from < current.stacks.length; from++) {
      if (current.stacks[from].length === 0) continue;
      for (let to = 0; to < current.stacks.length; to++) {
        if (from === to) continue;
        const moved = applyRelocation(current, from, to, scenario.rules);
        if (!moved) continue;

        const afterRetrieve = autoRetrieve(moved.nextState).state;
        if (isSolved(afterRetrieve)) return "feasible";

        const key = canonicalStateKey(afterRetrieve);
        if (!visited.has(key)) {
          visited.add(key);
          queue.push(afterRetrieve);
        }
      }
    }
  }

  return "infeasible";
}

/**
 * Generates a reproducible, feasible yard scenario.
 * @param {number} seed
 * @param {{ N?: number, H?: number, enforceWeightRule?: boolean, containerCount?: number, maxAttempts?: number }} [config]
 */
function generateScenario(seed, config = {}) {
  const N = config.N ?? 6;
  const H = config.H ?? 4;
  const enforceWeightRule = Boolean(config.enforceWeightRule);
  const maxAllowedItems = N * (H - 1);
  const requestedCount = config.containerCount ?? Math.min(maxAllowedItems, Math.max(4, Math.floor(N * (H - 1) * 0.75)));
  const containerCount = Math.min(requestedCount, maxAllowedItems);
  const maxAttempts = config.maxAttempts ?? 40;
  const rng = createRng(seed);
  const rules = createRules({ H, enforceWeightRule });

  let noFeasibleRejections = 0;
  let stateCapRejections = 0;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const stacks = Array.from({ length: N }, () => []);
    const pickupOrders = Array.from({ length: containerCount }, (_, i) => i + 1);
    for (let i = pickupOrders.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [pickupOrders[i], pickupOrders[j]] = [pickupOrders[j], pickupOrders[i]];
    }

    const containers = pickupOrders.map((order, idx) => {
      const wRoll = rng();
      const weightClass = wRoll < 0.34 ? "heavy" : wRoll < 0.67 ? "medium" : "light";
      const tRoll = rng();
      const containerType = tRoll < 0.55 ? "dry" : tRoll < 0.85 ? "reefer" : "hazmat";
      return createContainer({
        id: `C${idx + 1}`,
        weightClass,
        containerType,
        pickupOrder: order
      });
    });

    if (enforceWeightRule) {
      const weightRank = { heavy: 3, medium: 2, light: 1 };
      containers.sort((a, b) => weightRank[b.weightClass] - weightRank[a.weightClass]);
    }

    let placedAll = true;
    for (const c of containers) {
      const validStacks = [];
      for (let s = 0; s < N; s++) {
        if (canPlace(stacks[s], c, rules)) validStacks.push(s);
      }
      if (validStacks.length === 0) {
        placedAll = false;
        break;
      }
      const chosenStack = validStacks[Math.floor(rng() * validStacks.length)];
      stacks[chosenStack].push(c);
    }

    if (!placedAll) {
      noFeasibleRejections++;
      continue;
    }

    const candidate = { seed, N, rules, stacks };
    const check = checkFeasibility(candidate, { maxStates: 4000 });
    if (check === "feasible") {
      return {
        scenario: candidate,
        attempts: attempt + 1,
        noFeasibleRejections,
        stateCapRejections
      };
    }
    if (check === "limit reached") {
      stateCapRejections++;
    } else {
      noFeasibleRejections++;
    }
  }

  return {
    scenario: null,
    attempts: maxAttempts,
    noFeasibleRejections,
    stateCapRejections
  };
}

/**
 * Finds the stack index holding `nextNeeded` and its top blocker if buried.
 */
function locateNextNeeded(state) {
  for (let s = 0; s < state.stacks.length; s++) {
    const stack = state.stacks[s];
    for (let d = stack.length - 1; d >= 0; d--) {
      if (stack[d].pickupOrder === state.nextNeeded) {
        const isTop = d === stack.length - 1;
        return {
          stackIdx: s,
          isTop,
          blocker: isTop ? null : stack[stack.length - 1]
        };
      }
    }
  }
  return null;
}

/**
 * a. Naive Solver: moves each blocker to the nearest legal stack by index distance.
 */
function solveNaive(scenario) {
  let state = autoRetrieve(initialState(scenario.stacks)).state;
  const moves = [];
  const moveCap = 5000;

  while (!isSolved(state)) {
    if (moves.length >= moveCap) {
      return { status: "limit reached", moves, relocations: moves.length };
    }

    const target = locateNextNeeded(state);
    if (!target) {
      return { status: "infeasible", moves, relocations: moves.length };
    }

    if (target.isTop) {
      state = autoRetrieve(state).state;
      continue;
    }

    const fromIdx = target.stackIdx;
    const blocker = target.blocker;

    let bestDest = -1;
    let bestDist = Infinity;
    for (let toIdx = 0; toIdx < state.stacks.length; toIdx++) {
      if (toIdx === fromIdx) continue;
      if (!canPlace(state.stacks[toIdx], blocker, scenario.rules)) continue;
      const dist = Math.abs(toIdx - fromIdx);
      if (dist < bestDist || (dist === bestDist && toIdx < bestDest)) {
        bestDist = dist;
        bestDest = toIdx;
      }
    }

    if (bestDest === -1) {
      return { status: "infeasible", moves, relocations: moves.length };
    }

    const applied = applyRelocation(state, fromIdx, bestDest, scenario.rules);
    if (!applied) {
      return { status: "infeasible", moves, relocations: moves.length };
    }

    moves.push({ from: fromIdx, to: bestDest, containerId: blocker.id });
    state = autoRetrieve(applied.nextState).state;
  }

  return { status: "solved", moves, relocations: moves.length };
}

/**
 * b. Heuristic Solver (min-max rule):
 * Picks the legal non-empty destination stack whose top container has the latest
 * pickupOrder (least likely to block soon). Uses an empty stack only when no
 * legal non-empty stack exists. Ties are broken by lowest stack index.
 */
function solveHeuristic(scenario, options = {}) {
  const maxSteps = options.maxSteps ?? 500;
  const rules = scenario.rules;
  let state = autoRetrieve(initialState(scenario.stacks)).state;
  const moves = [];
  let steps = 0;

  while (!isSolved(state) && steps < maxSteps) {
    steps++;
    const targetInfo = findNeededLocation(state);
    if (!targetInfo) {
      return { status: "infeasible", relocations: moves.length, moves, finalState: state };
    }

    const { stackIdx: fromStack } = targetInfo;
    const blocker = state.stacks[fromStack][state.stacks[fromStack].length - 1];
    if (!blocker) {
      return { status: "infeasible", relocations: moves.length, moves, finalState: state };
    }

    const candidates = [];
    for (let toStack = 0; toStack < state.stacks.length; toStack++) {
      if (toStack === fromStack) continue;
      if (!canPlace(state.stacks[toStack], blocker, rules)) continue;

      const stackMin = state.stacks[toStack].length === 0
        ? Infinity
        : Math.min(...state.stacks[toStack].map((item) => item.pickupOrder));

      candidates.push({
        toStack,
        stackMin,
        isSafe: stackMin > blocker.pickupOrder
      });
    }

    if (candidates.length === 0) {
      return { status: "infeasible", relocations: moves.length, moves, finalState: state };
    }

    const safeCandidates = candidates.filter((c) => c.isSafe);
    let chosen;
    if (safeCandidates.length > 0) {
      safeCandidates.sort((a, b) => (a.stackMin - b.stackMin) || (a.toStack - b.toStack));
      chosen = safeCandidates[0];
    } else {
      candidates.sort((a, b) => (b.stackMin - a.stackMin) || (a.toStack - b.toStack));
      chosen = candidates[0];
    }

    const next = applyRelocation(state, fromStack, chosen.toStack, rules);
    if (!next) {
      return { status: "infeasible", relocations: moves.length, moves, finalState: state };
    }

    moves.push({
      from: fromStack,
      to: chosen.toStack,
      containerId: blocker.id,
      pickupOrder: blocker.pickupOrder
    });
    state = next;
  }

  if (!isSolved(state)) {
    return { status: "limit reached", relocations: moves.length, moves, finalState: state };
  }

  return { status: "solved", relocations: moves.length, moves, finalState: state };
}
    const applied = applyRelocation(state, fromIdx, chosenDest, scenario.rules);
    if (!applied) {
      return { status: "infeasible", moves, relocations: moves.length };
    }

    moves.push({ from: fromIdx, to: chosenDest, containerId: blocker.id });
    state = autoRetrieve(applied.nextState).state;
  }

  return { status: "solved", moves, relocations: moves.length };
}

function countContainers(scenario) {
  return scenario.stacks.reduce((sum, stack) => sum + stack.length, 0);
}

function isWithinOptimalLimits(scenario) {
  const totalContainers = countContainers(scenario);
  return totalContainers <= 8 && scenario.N <= 4 && scenario.rules.H <= 3;
}

function canonicalStateKey(state) {
  const stackSignatures = state.stacks.map((stack) =>
    stack.map((c) => `${c.pickupOrder}:${c.weightClass[0]}`).join(",")
  );
  stackSignatures.sort();
  return `${stackSignatures.join("|")}#${state.nextNeeded}`;
}

/**
 * c. Optimal Solver (BFS with canonical memoization).
 * Hard bounds: <= 8 containers, N <= 4, H <= 3, <= 200,000 states, <= 3000ms.
 */
function solveOptimal(scenario, options = {}) {
  if (!isWithinOptimalLimits(scenario)) {
    return {
      status: "limit reached",
      reason: "too large for optimal solver",
      moves: [],
      relocations: 0
    };
  }

  const maxStates = options.maxStates || 200000;
  const timeCapMs = options.timeCapMs || 3000;
  const startTime = Date.now();

  const initial = autoRetrieve(initialState(scenario.stacks)).state;
  if (isSolved(initial)) {
    return { status: "solved", moves: [], relocations: 0 };
  }

  let frontier = [{ state: initial, moves: [] }];
  const visited = new Set([canonicalStateKey(initial)]);
  let exploredStates = 0;

  while (frontier.length > 0) {
    const nextFrontier = [];

    for (let idx = 0; idx < frontier.length; idx++) {
      const node = frontier[idx];
      exploredStates++;

      if (exploredStates > maxStates || Date.now() - startTime > timeCapMs) {
        return {
          status: "limit reached",
          reason: exploredStates > maxStates ? "state cap reached" : "time cap reached",
          moves: [],
          relocations: 0
        };
      }

      const target = locateNextNeeded(node.state);
      if (!target || target.isTop) continue;

      const fromIdx = target.stackIdx;
      const blocker = target.blocker;

      for (let toIdx = 0; toIdx < node.state.stacks.length; toIdx++) {
        if (toIdx === fromIdx) continue;
        const moved = applyRelocation(node.state, fromIdx, toIdx, scenario.rules);
        if (!moved) continue;

        const afterRetrieve = autoRetrieve(moved.nextState).state;
        const updatedMoves = node.moves.concat({
          from: fromIdx,
          to: toIdx,
          containerId: blocker.id
        });

        if (isSolved(afterRetrieve)) {
          return {
            status: "solved",
            moves: updatedMoves,
            relocations: updatedMoves.length
          };
        }

        const key = canonicalStateKey(afterRetrieve);
        if (!visited.has(key)) {
          visited.add(key);
          nextFrontier.push({ state: afterRetrieve, moves: updatedMoves });
        }
      }
    }

    frontier = nextFrontier;
  }

  return { status: "infeasible", moves: [], relocations: 0 };
}

/**
 * Replays a move sequence step-by-step against the model to verify legality and completion.
 */
function verifyMoveSequence(scenario, moves) {
  let state = autoRetrieve(initialState(scenario.stacks)).state;
  if (isSolved(state)) {
    return { valid: moves.length === 0, reason: moves.length === 0 ? "ok" : "extra moves after solved" };
  }

  for (let i = 0; i < moves.length; i++) {
    const move = moves[i];
    const target = locateNextNeeded(state);
    if (!target || target.isTop) {
      return { valid: false, reason: `Move #${i + 1}: no buried blocker needed relocation` };
    }
    if (move.from !== target.stackIdx) {
      return { valid: false, reason: `Move #${i + 1}: expected from stack ${target.stackIdx}, got ${move.from}` };
    }

    const applied = applyRelocation(state, move.from, move.to, scenario.rules);
    if (!applied) {
      return { valid: false, reason: `Move #${i + 1}: illegal placement ${move.from} -> ${move.to}` };
    }

    state = autoRetrieve(applied.nextState).state;
    if (isSolved(state) && i < moves.length - 1) {
      return { valid: false, reason: `Move #${i + 1}: yard already solved before last move` };
    }
  }

  if (!isSolved(state)) {
    return { valid: false, reason: "Yard still has remaining containers after all moves" };
  }

  return { valid: true, reason: "ok" };
}

/**
 * Self-check callable from browser console.
 * 1) Verifies no solver produces illegal moves.
 * 2) Asserts Optimal <= Heuristic and Optimal <= Naive whenever Optimal is "solved".
 * 3) Reports (without asserting) how often Heuristic > Naive.
 */
function runSelfCheck(options = {}) {
  const sampleCount = options.count || 120;
  const enforceWeightRule = Boolean(options.enforceWeightRule);

  let testedScenarios = 0;
  let illegalSequences = 0;
  let optimalSolvedCount = 0;
  let optimalViolations = 0;
  let heuristicWorseThanNaive = 0;
  const issues = [];

  for (let i = 0; i < sampleCount; i++) {
    const seed = 1000 + i;
    const { scenario } = generateScenario(seed, {
      N: 4,
      H: 3,
      enforceWeightRule,
      containerCount: 7
    });
    if (!scenario) continue;

    testedScenarios++;
    const naiveRes = solveNaive(scenario);
    const heuristicRes = solveHeuristic(scenario);
    const optimalRes = solveOptimal(scenario);

    const solversToCheck = [
      { name: "Naive", res: naiveRes },
      { name: "Heuristic", res: heuristicRes },
      { name: "Optimal", res: optimalRes }
    ];

    for (const entry of solversToCheck) {
      if (entry.res.status === "solved") {
        const check = verifyMoveSequence(scenario, entry.res.moves);
        if (!check.valid) {
          illegalSequences++;
          issues.push(`Seed ${seed} [${entry.name}]: ${check.reason}`);
        }
      }
    }

    if (optimalRes.status === "solved") {
      optimalSolvedCount++;
      if (heuristicRes.status === "solved" && optimalRes.relocations > heuristicRes.relocations) {
        optimalViolations++;
        issues.push(`Seed ${seed}: Optimal (${optimalRes.relocations}) > Heuristic (${heuristicRes.relocations})`);
      }
      if (naiveRes.status === "solved" && optimalRes.relocations > naiveRes.relocations) {
        optimalViolations++;
        issues.push(`Seed ${seed}: Optimal (${optimalRes.relocations}) > Naive (${naiveRes.relocations})`);
      }
    }

    if (
      heuristicRes.status === "solved" &&
      naiveRes.status === "solved" &&
      heuristicRes.relocations > naiveRes.relocations
    ) {
      heuristicWorseThanNaive++;
    }
  }

  const passed = illegalSequences === 0 && optimalViolations === 0;

  console.group("Container Yard Optimizer — runSelfCheck()");
  console.log("Tested scenarios:", testedScenarios);
  console.log("Illegal move sequences:", illegalSequences);
  console.log("Optimal solved count:", optimalSolvedCount);
  console.log("Optimal optimality violations:", optimalViolations);
  console.log("Report — Heuristic > Naive occurrences:", heuristicWorseThanNaive);
  if (issues.length > 0) {
    console.warn("Issues:", issues);
  }
  console.log("Overall status:", passed ? "PASS" : "FAIL");
  console.groupEnd();

  return {
    passed,
    testedScenarios,
    illegalSequences,
    optimalSolvedCount,
    optimalViolations,
    heuristicWorseThanNaive,
    issues
  };
}
function debugSeed(seed = 42) {
  const sizeKey = document.getElementById("yardSizeSelect")?.value || "small";
  const enforceWeightRule = Boolean(document.getElementById("weightRuleToggle")?.checked);
  const dims = sizeKey === "main" ? { N: 6, H: 4 } : { N: 4, H: 3 };
  const gen = generateScenario(Number(seed), { ...dims, enforceWeightRule });

  if (!gen.scenario) {
    console.warn(`Seed ${seed}: no feasible scenario generated.`);
    return null;
  }

  const formatMoves = (moves) =>
    moves.map((m) => `${m.from} -> ${m.to} | #${m.pickupOrder}`).join("\n  ") || "(none)";

  const naive = solveNaive(gen.scenario);
  const heuristic = solveHeuristic(gen.scenario);
  const optimal = solveOptimal(gen.scenario);

  console.group(`debugSeed(${seed}) [${dims.N}x${dims.H}, weightRule=${enforceWeightRule}]`);
  console.log("Initial stacks:", gen.scenario.stacks.map((st, idx) =>
    `S${idx + 1}: [${st.map((c) => `#${c.pickupOrder}(${c.weightClass[0].toUpperCase()})`).join(", ")}]`
  ).join(" | "));
  console.log(`Naive (${naive.status}, ${naive.relocations} moves):\n  ${formatMoves(naive.moves)}`);
  console.log(`Heuristic (${heuristic.status}, ${heuristic.relocations} moves):\n  ${formatMoves(heuristic.moves)}`);
  console.log(`Optimal (${optimal.status}, ${optimal.relocations} moves):\n  ${formatMoves(optimal.moves)}`);
  console.groupEnd();

  return { scenario: gen.scenario, naive, heuristic, optimal };
}
// ============================================================
// Stage 3: i18n, Theme Toggle, Lightweight 2.5D SVG Renderer & Manual Mode
// ============================================================

const I18N = {
  en: {
    pageTitle: "Container Yard Relocation Optimizer",
    heroSub: "Containers are stacked vertically; when the next container needed is buried underneath, every extra crane move wastes time and fuel.",
    howItWorks: "How it works: 1) Pick a scenario  2) Compare methods  3) See moves and cost",
    heroLabel: "Headline Metric",
    heroBeforeRun: "Run the benchmark to see results",
    heroAfterRun: "{pct}% relocation reduction vs Naive ({solved} solved scenarios)",
    yardTitle: "Interactive Yard & Manual Mode",
    yardDesc: "Click or press Enter on a stack to pick up its top container, then select another stack to place it. Retrieval happens automatically.",
    seedLabel: "Seed",
    yardSizeLabel: "Yard Size",
    sizeSmall: "Small (4 × 3 — Optimal Ready)",
    sizeMain: "Standard (6 × 4)",
    weightRuleLabel: "Weight Rule (No Heavy on Light)",
    btnLoadSeed: "Load Seed",
    btnNewScenario: "New Random",
    btnReset: "Reset",
    btnSolveNaive: "Solve: Naive",
    btnSolveHeuristic: "Solve: Heuristic",
    btnSolveOptimal: "Solve: Optimal",
    relocationsLabel: "Relocations",
    nextNeededLabel: "Next Pickup",
    legendDry: "Dry",
    legendReefer: "Reefer",
    legendHazmat: "Hazmat",
    legendNext: "Next Needed",
    statusPick: "Next needed: #{n}. Select a stack to pick up its top container.",
    statusSelected: "Selected stack S{s} (#{n}). Choose destination stack.",
    statusSolved: "Solved! All containers retrieved in {moves} relocations.",
    statusIllegal: "Illegal move: violates height limit or weight stacking rule.",
    replayTitle: "Side-by-Side Replay",
    replayDesc: "Step through identical initial yard states across Naive, Heuristic, and Optimal solvers.",
    replayPlay: "Play",
    replayPause: "Pause",
    replayStep: "Step",
    replayReset: "Reset",
    replaySpeed: "Speed:",
    replayStepLabel: "Step:",
    replayMoves: "Moves:",
    replayRunning: "In Progress",
    replayDoneBadge: "All containers retrieved",
    replayLimitBadge: "Limit reached",
    benchTitle: "Benchmarks & Crane Cost Estimator",
    benchDesc: "Run non-blocking batch experiments to measure average relocations, optimality gap, crane time, and hourly cost impact.",
    runMain: "Run Main Benchmark (1,000 × 6×4)",
    runSmall: "Run Small Benchmark (200 × 4×3 + Optimal)",
    naiveAvg: "Naive Avg",
    heuristicAvg: "Heuristic Avg",
    reductionLabel: "Reduction vs Naive",
    worseThanNaive: "worse than Naive",
    optimalAvg: "Optimal Avg",
    optimalityGap: "Gap to Optimal",
    solvedCount: "Solved Scenarios",
    rejectedNoFeasible: "Rejected (No Feasible)",
    rejectedStateCap: "Rejected (State Cap)",
    worseSummary: "Scenarios where Heuristic lost to Naive ({count})",
    chartTitle: "Average Relocations per Scenario",
    costTitle: "Crane Time & Cost Savings",
    costMinLabel: "Minutes per Relocation",
    costHourlyLabel: "Crane Hourly Cost ($)",
    costPerScenario: "Per scenario: {moves} relocations saved ≈ {mins} min ≈ ${cost}",
    costTotal: "Across {n} scenarios: {moves} relocations saved ≈ {mins} min ≈ ${cost}",
    exportCsv: "Export CSV",
    printReport: "Print Report",
    caseTitle: "Case Study & Model Scope",
    caseSub: "Empirical findings and explicit modeling assumptions updated automatically after each benchmark run."
  },
  ar: {
    pageTitle: "محسّن مناولة الحاويات في الموانئ",
    heroSub: "تُرصّ الحاويات فوق بعضها؛ وعندما تكون الحاوية المطلوبة مدفونة بالأسفل، تستهلك كل حركة رافعة إضافية وقتاً ووقوداً.",
    howItWorks: "طريقة العمل: 1) اختر سيناريو  2) قارن الطرق  3) شاهد الحركات والتكلفة",
    heroLabel: "المؤشر الرئيسي",
    heroBeforeRun: "شغّل القياس لعرض النتائج",
    heroAfterRun: "تخفيض المناورات بنسبة {pct}% مقارنة بالحل Naive ({solved} سيناريو محلول)",
    yardTitle: "الساحة التفاعلية والوضع اليدوي",
    yardDesc: "انقر على عمود لالتقاط الحاوية العلوية، ثم اختر عموداً آخر لوضعها. يتم سحب الحاوية المطلوبة تلقائياً عند وصولها للقمة.",
    seedLabel: "البذرة (Seed)",
    yardSizeLabel: "حجم الساحة",
    sizeSmall: "صغير (4 × 3 — يدعم الحل Optimal)",
    sizeMain: "قياسي (6 × 4)",
    weightRuleLabel: "قاعدة الوزن (منع الثقيل فوق الخفيف)",
    btnLoadSeed: "توليد بالبذرة",
    btnNewScenario: "سيناريو عشوائي",
    btnReset: "إعادة ضبط",
    btnSolveNaive: "حل: Naive",
    btnSolveHeuristic: "حل: Heuristic",
    btnSolveOptimal: "حل: Optimal",
    relocationsLabel: "المناورات",
    nextNeededLabel: "السحب التالي",
    legendDry: "عادية (Dry)",
    legendReefer: "مبردة (Reefer)",
    legendHazmat: "خطرة (Hazmat)",
    legendNext: "المطلوبة تالياً",
    statusPick: "المطلوب تالياً: #{n}. اختر عموداً لالتقاط حاويته العلوية.",
    statusSelected: "تم تحديد العمود S{s} (#{n}). اختر العمود الوجهة.",
    statusSolved: "اكتمل الحل! تم سحب جميع الحاويات خلال {moves} مناورات.",
    statusIllegal: "حركة غير مسموحة: تخالف حد الارتفاع أو قاعدة تكديس الأوزان.",
    replayTitle: "العرض المتزامن للمحلّات",
    replayDesc: "تتبّع نفس حالة الساحة الابتدائية خطوة بخطوة عبر Naive و Heuristic و Optimal.",
    replayPlay: "تشغيل",
    replayPause: "إيقاف مؤقت",
    replayStep: "خطوة",
    replayReset: "إعادة",
    replaySpeed: "السرعة:",
    replayStepLabel: "الخطوة:",
    replayMoves: "الحركات:",
    replayRunning: "قيد التنفيذ",
    replayDoneBadge: "تم سحب جميع الحاويات",
    replayLimitBadge: "بلغ حد البحث",
    benchTitle: "اختبارات الأداء وحاسبة تكلفة الرافعة",
    benchDesc: "تشغيل تجارب دفعية غير معطّلة لقياس متوسط المناورات، الفجوة عن الحل الأمثل، زمن الرافعة، والأثر المالي.",
    runMain: "تشغيل القياس الرئيسي (1,000 × 6×4)",
    runSmall: "تشغيل القياس المصغّر (200 × 4×3 + Optimal)",
    naiveAvg: "متوسط Naive",
    heuristicAvg: "متوسط Heuristic",
    reductionLabel: "نسبة التخفيض",
    worseThanNaive: "أسوأ من Naive",
    optimalAvg: "متوسط Optimal",
    optimalityGap: "الفجوة عن Optimal",
    solvedCount: "السيناريوهات المحلولة",
    rejectedNoFeasible: "مرفوض (لا يوجد حل مجدٍ)",
    rejectedStateCap: "مرفوض (بلغ حد الحالات)",
    worseSummary: "السيناريوهات التي تفوّق فيها Naive على Heuristic ({count})",
    chartTitle: "متوسط المناورات لكل سيناريو",
    costTitle: "توفير زمن وتكلفة الرافعة",
    costMinLabel: "دقائق لكل مناورة",
    costHourlyLabel: "تكلفة ساعة الرافعة ($)",
    costPerScenario: "لكل سيناريو: توفير {moves} مناورة ≈ {mins} دقيقة ≈ ${cost}",
    costTotal: "عبر {n} سيناريو: توفير {moves} مناورة ≈ {mins} دقيقة ≈ ${cost}",
    exportCsv: "تصدير CSV",
    printReport: "طباعة التقرير",
    caseTitle: "دراسة الحالة ونطاق النموذج",
    caseSub: "النتائج التجريبية وافتراضات النمذجة الصريحة تتحدث تلقائياً بعد كل قياس."
  }
};

function t(key, vars = {}) {
  const lang = document.documentElement.lang === "ar" ? "ar" : "en";
  let str = I18N[lang][key] || I18N.en[key] || key;
  for (const [k, v] of Object.entries(vars)) {
    str = str.replace(`{${k}}`, String(v));
  }
  return str;
}

function applyLanguage(lang) {
  const activeLang = lang === "ar" ? "ar" : "en";
  document.documentElement.lang = activeLang;
  document.documentElement.dir = activeLang === "ar" ? "rtl" : "ltr";

  const langBtn = document.getElementById("langToggle");
  if (langBtn) langBtn.textContent = activeLang === "ar" ? "English" : "العربية";

  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (key) el.textContent = t(key);
  });

  renderAllDynamicSections();
}
}

function applyTheme(theme) {
  const validTheme = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", validTheme);
  const icon = document.getElementById("themeIcon");
  if (icon) icon.textContent = validTheme === "dark" ? "🌙" : "☀️";

  try {
    localStorage.setItem("yard_theme", validTheme);
  } catch (_) {}
}

// ============================================================
// Lightweight 2.5D SVG Yard Visualization
// ============================================================

const CONTAINER_PALETTE = {
  dry: { front: "#2563eb", top: "#3b82f6", side: "#1d4ed8" },
  reefer: { front: "#0d9488", top: "#14b8a6", side: "#0f766e" },
  hazmat: { front: "#d97706", top: "#f59e0b", side: "#b45309" }
};

function weightShortLabel(weightClass) {
  if (weightClass === "light") return t("weightLight");
  if (weightClass === "heavy") return t("weightHeavy");
  return t("weightMedium");
}

/**
 * Renders the yard stacks using lightweight SVG with subtle 2.5D bevels.
 */
function renderYardSvg(containerEl, state, options = {}) {
  if (!containerEl || !state) return;

  const N = state.stacks.length;
  const H = options.H || 4;
  const selectedStack = options.selectedStack ?? null;
  const interactive = Boolean(options.interactive);

  const boxW = 68;
  const boxH = 34;
  const depthX = 7;
  const depthY = 6;
  const gapX = 18;
  const gapY = 6;
  const padX = 24;
  const padTop = 26;
  const padBottom = 30;

  const totalW = padX * 2 + N * boxW + (N - 1) * gapX + depthX;
  const totalH = padTop + H * (boxH + gapY) + padBottom;

  let nextPos = null;
  for (let s = 0; s < N; s++) {
    for (let d = 0; d < state.stacks[s].length; d++) {
      if (state.stacks[s][d].pickupOrder === state.nextNeeded) {
        nextPos = { s, d };
      }
    }
  }

  let svg = `<svg viewBox="0 0 ${totalW} ${totalH}" xmlns="http://www.w3.org/2000/svg" class="yard-svg" role="img" aria-label="Container Yard">`;

  for (let s = 0; s < N; s++) {
    const stack = state.stacks[s];
    const x = padX + s * (boxW + gapX);
    const groundY = padTop + H * (boxH + gapY);
    const isSelected = selectedStack === s;

    const ariaText = t("stackAria", { index: s + 1, count: stack.length, max: H });
    const interactiveAttrs = interactive
      ? `tabindex="0" role="button" aria-label="${ariaText}" style="cursor:pointer;"`
      : "";

    svg += `<g data-stack-idx="${s}" ${interactiveAttrs}>`;

    // Hit area & subtle stack slot guide
    svg += `<rect x="${x - 3}" y="${padTop - 8}" width="${boxW + depthX + 6}" height="${groundY - padTop + 28}" rx="8" fill="${isSelected ? "rgba(59,130,246,0.14)" : "transparent"}" stroke="${isSelected ? "#3b82f6" : "transparent"}" stroke-width="1.5" />`;

    // Empty slot outlines
    for (let slot = 0; slot < H; slot++) {
      const slotY = groundY - (slot + 1) * (boxH + gapY) + gapY;
      svg += `<rect x="${x}" y="${slotY}" width="${boxW}" height="${boxH}" rx="4" fill="none" stroke="var(--border)" stroke-dasharray="3 3" stroke-width="1" opacity="0.55" />`;
    }

    // Ground rail
    svg += `<line x1="${x - 2}" y1="${groundY + 2}" x2="${x + boxW + depthX}" y2="${groundY + 2}" stroke="var(--border)" stroke-width="2.5" />`;
    svg += `<text x="${x + boxW / 2}" y="${groundY + 20}" text-anchor="middle" fill="var(--text-muted)" font-size="11" font-weight="600">S${s + 1}</text>`;

    // Containers in stack
    for (let d = 0; d < stack.length; d++) {
      const c = stack[d];
      const y = groundY - (d + 1) * (boxH + gapY) + gapY;
      const isTop = d === stack.length - 1;
      const liftY = isSelected && isTop ? y - 6 : y;

      const palette = CONTAINER_PALETTE[c.containerType] || CONTAINER_PALETTE.dry;
      const isNext = nextPos && nextPos.s === s && nextPos.d === d;
      const strokeColor = isNext ? "#fbbf24" : "rgba(15,23,42,0.45)";
      const strokeWidth = isNext ? 2.5 : 1;

      // Top 2.5D face
      svg += `<polygon points="${x},${liftY} ${x + depthX},${liftY - depthY} ${x + boxW + depthX},${liftY - depthY} ${x + boxW},${liftY}" fill="${palette.top}" stroke="${strokeColor}" stroke-width="0.8" />`;
      // Right 2.5D side face
      svg += `<polygon points="${x + boxW},${liftY} ${x + boxW + depthX},${liftY - depthY} ${x + boxW + depthX},${liftY + boxH - depthY} ${x + boxW},${liftY + boxH}" fill="${palette.side}" stroke="${strokeColor}" stroke-width="0.8" />`;
      // Front face
      svg += `<rect x="${x}" y="${liftY}" width="${boxW}" height="${boxH}" rx="3" fill="${palette.front}" stroke="${strokeColor}" stroke-width="${strokeWidth}" />`;

      // Pickup order & weight badge
      svg += `<text x="${x + 22}" y="${liftY + boxH / 2 + 4}" text-anchor="middle" fill="#ffffff" font-size="13" font-weight="700">#${c.pickupOrder}</text>`;
      svg += `<text x="${x + boxW - 14}" y="${liftY + boxH / 2 + 4}" text-anchor="middle" fill="rgba(255,255,255,0.9)" font-size="10" font-weight="600">${weightShortLabel(c.weightClass)}</text>`;
    }

    svg += `</g>`;
  }

  svg += `</svg>`;
  containerEl.innerHTML = svg;
}

function renderYardLegend(legendEl) {
  if (!legendEl) return;
  const items = [
    { label: t("legendDry"), color: CONTAINER_PALETTE.dry.front, outline: false },
    { label: t("legendReefer"), color: CONTAINER_PALETTE.reefer.front, outline: false },
    { label: t("legendHazmat"), color: CONTAINER_PALETTE.hazmat.front, outline: false },
    { label: t("legendNext"), color: "#fbbf24", outline: true }
  ];

  legendEl.innerHTML = `<div class="yard-legend">` + items.map((item) => {
    const swatchStyle = item.outline
      ? `border: 2px solid ${item.color}; background: transparent;`
      : `background: ${item.color};`;
    return `<span class="legend-item"><span class="legend-swatch" style="${swatchStyle}"></span>${item.label}</span>`;
  }).join("") + `</div>`;
}

// ============================================================
// Manual Mode Controller
// ============================================================

const manualDemo = {
  scenario: null,
  state: null,
  selectedStack: null,
  relocations: 0,
  statusMessage: null,
  statusType: "neutral"
};

function getDemoConfigFromInputs() {
  const seedEl = document.getElementById("seedInput");
  const sizeEl = document.getElementById("yardSizeSelect");
  const weightEl = document.getElementById("weightRuleToggle");

  const seed = seedEl ? parseInt(seedEl.value, 10) || 42 : 42;
  const isSmall = sizeEl && sizeEl.value === "small";
  const enforceWeightRule = weightEl ? Boolean(weightEl.checked) : false;

  return {
    seed,
    N: isSmall ? 4 : 6,
    H: isSmall ? 3 : 4,
    enforceWeightRule
  };
}

function loadDemoScenario(customSeed) {
  const cfg = getDemoConfigFromInputs();
  const seedToUse = customSeed !== undefined ? customSeed : cfg.seed;

  const seedEl = document.getElementById("seedInput");
  if (seedEl) seedEl.value = seedToUse;

  const { scenario } = generateScenario(seedToUse, {
    N: cfg.N,
    H: cfg.H,
    enforceWeightRule: cfg.enforceWeightRule
  });

  if (!scenario) {
    manualDemo.statusMessage = t("statusGenFailed");
    manualDemo.statusType = "error";
    renderManualDemo();
    return;
  }

  manualDemo.scenario = scenario;
  manualDemo.state = autoRetrieve(initialState(scenario.stacks)).state;
  manualDemo.selectedStack = null;
  manualDemo.relocations = 0;
  manualDemo.statusMessage = null;
  manualDemo.statusType = "neutral";

  renderManualDemo();
}

function handleStackInteraction(stackIdx) {
  if (!manualDemo.scenario || !manualDemo.state) return;
  if (isSolved(manualDemo.state)) return;

  const stacks = manualDemo.state.stacks;
  if (stackIdx < 0 || stackIdx >= stacks.length) return;

  // Step 1: Pick up top container from clicked stack
  if (manualDemo.selectedStack === null) {
    if (stacks[stackIdx].length === 0) {
      manualDemo.statusMessage = t("statusEmptyStack", { stack: stackIdx + 1 });
      manualDemo.statusType = "error";
      renderManualDemo();
      return;
    }

    manualDemo.selectedStack = stackIdx;
    manualDemo.statusMessage = null;
    manualDemo.statusType = "neutral";
    renderManualDemo();
    return;
  }

  // Clicking the same stack cancels selection
  if (manualDemo.selectedStack === stackIdx) {
    manualDemo.selectedStack = null;
    manualDemo.statusMessage = null;
    manualDemo.statusType = "neutral";
    renderManualDemo();
    return;
  }

  // Step 2: Attempt to place onto destination stack
  const fromIdx = manualDemo.selectedStack;
  const movingContainer = stacks[fromIdx][stacks[fromIdx].length - 1];
  const result = applyRelocation(manualDemo.state, fromIdx, stackIdx, manualDemo.scenario.rules);

  if (!result) {
    manualDemo.statusMessage = t("statusIllegalMove", {
      order: movingContainer.pickupOrder,
      stack: stackIdx + 1
    });
    manualDemo.statusType = "error";
    manualDemo.selectedStack = null;
    renderManualDemo();
    return;
  }

  manualDemo.state = autoRetrieve(result.nextState).state;
  manualDemo.relocations += 1;
  manualDemo.selectedStack = null;
  manualDemo.statusMessage = null;
  manualDemo.statusType = "neutral";
  renderManualDemo();
}

function resetManualDemo() {
  if (!manualDemo.scenario) return;
  manualDemo.state = autoRetrieve(initialState(manualDemo.scenario.stacks)).state;
  manualDemo.selectedStack = null;
  manualDemo.relocations = 0;
  manualDemo.statusMessage = null;
  manualDemo.statusType = "neutral";
  renderManualDemo();
}

function previewSolverOnDemo(solverName) {
  if (!manualDemo.scenario) return;

  let res;
  if (solverName === "naive") res = solveNaive(manualDemo.scenario);
  else if (solverName === "heuristic") res = solveHeuristic(manualDemo.scenario);
  else res = solveOptimal(manualDemo.scenario);

  if (res.status !== "solved") {
    manualDemo.statusMessage = t("statusSolverLimit", {
      solver: solverName.toUpperCase(),
      reason: res.reason || res.status
    });
    manualDemo.statusType = "error";
    renderManualDemo();
    return;
  }

  let state = autoRetrieve(initialState(manualDemo.scenario.stacks)).state;
  for (const move of res.moves) {
    const step = applyRelocation(state, move.from, move.to, manualDemo.scenario.rules);
    if (!step) break;
    state = autoRetrieve(step.nextState).state;
  }

  manualDemo.state = state;
  manualDemo.relocations = res.relocations;
  manualDemo.selectedStack = null;
  manualDemo.statusMessage = t("statusAutoSolved", {
    solver: solverName.toUpperCase(),
    count: res.relocations
  });
  manualDemo.statusType = "success";
  renderManualDemo();
}

function renderManualDemo() {
  const yardEl = document.getElementById("yardContainer");
  const legendEl = document.getElementById("yardLegend");
  const statusEl = document.getElementById("yardStatus");
  const relocEl = document.getElementById("relocationCount");
  const nextEl = document.getElementById("nextNeededCount");

  if (yardEl && manualDemo.state && manualDemo.scenario) {
    renderYardSvg(yardEl, manualDemo.state, {
      H: manualDemo.scenario.rules.H,
      selectedStack: manualDemo.selectedStack,
      interactive: true
    });
  }

  renderYardLegend(legendEl);

  if (relocEl) relocEl.textContent = String(manualDemo.relocations);
  if (nextEl && manualDemo.state) {
    nextEl.textContent = isSolved(manualDemo.state) ? "—" : `#${manualDemo.state.nextNeeded}`;
  }

  if (!statusEl || !manualDemo.state) return;

  if (manualDemo.statusMessage) {
    statusEl.textContent = manualDemo.statusMessage;
    statusEl.className = `yard-status ${manualDemo.statusType}`;
    return;
  }

  if (isSolved(manualDemo.state)) {
    statusEl.textContent = t("statusSolved", { count: manualDemo.relocations });
    statusEl.className = "yard-status success";
    return;
  }

  if (manualDemo.selectedStack !== null) {
    const stack = manualDemo.state.stacks[manualDemo.selectedStack];
    const top = stack[stack.length - 1];
    statusEl.textContent = t("statusSelected", {
      stack: manualDemo.selectedStack + 1,
      order: top ? top.pickupOrder : "?",
      weight: top ? weightShortLabel(top.weightClass) : ""
    });
    statusEl.className = "yard-status";
  } else {
    statusEl.textContent = t("statusReady", { next: manualDemo.state.nextNeeded });
    statusEl.className = "yard-status";
  }
}

function bindStage3Events() {
  const yardEl = document.getElementById("yardContainer");
  if (yardEl && !yardEl.dataset.bound) {
    yardEl.dataset.bound = "1";
    yardEl.addEventListener("click", (e) => {
      const group = e.target.closest("[data-stack-idx]");
      if (!group) return;
      handleStackInteraction(parseInt(group.getAttribute("data-stack-idx"), 10));
    });
    yardEl.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      const group = e.target.closest("[data-stack-idx]");
      if (!group) return;
      e.preventDefault();
      handleStackInteraction(parseInt(group.getAttribute("data-stack-idx"), 10));
    });
  }

  const onClick = (id, fn) => {
    const el = document.getElementById(id);
    if (el && !el.dataset.bound) {
      el.dataset.bound = "1";
      el.addEventListener("click", fn);
    }
  };

  onClick("btnGenerateSeed", () => loadDemoScenario());
  onClick("btnNewScenario", () => loadDemoScenario(Math.floor(Math.random() * 100000)));
  onClick("btnReset", resetManualDemo);
  onClick("btnSolveNaive", () => previewSolverOnDemo("naive"));
  onClick("btnSolveHeuristic", () => previewSolverOnDemo("heuristic"));
  onClick("btnSolveOptimal", () => previewSolverOnDemo("optimal"));

  const sizeEl = document.getElementById("yardSizeSelect");
  const weightEl = document.getElementById("weightRuleToggle");
  if (sizeEl && !sizeEl.dataset.bound) {
    sizeEl.dataset.bound = "1";
    sizeEl.addEventListener("change", () => loadDemoScenario());
  }
  if (weightEl && !weightEl.dataset.bound) {
    weightEl.dataset.bound = "1";
    weightEl.addEventListener("change", () => loadDemoScenario());
  }
}
// ============================================================
// Stage 4: Side-by-Side Replay, Chunked Benchmarks, SVG Chart & Cost Calculator
// ============================================================

// ------------------------------------------------------------
// 4A. Side-by-Side Replay Engine
// ------------------------------------------------------------

const replayController = {
  scenario: null,
  results: {
    naive: null,
    heuristic: null,
    optimal: null
  },
  step: 0,
  maxSteps: 0,
  playing: false,
  speed: 1,
  timerId: null
};

function setupReplayScenario(customScenario) {
  stopReplayLoop();

  let scenario = customScenario;
  if (!scenario) {
    const cfg = typeof getDemoConfigFromInputs === "function"
      ? getDemoConfigFromInputs()
      : { seed: 42, N: 4, H: 3, enforceWeightRule: false };

    const generated = generateScenario(cfg.seed, {
      N: cfg.N,
      H: cfg.H,
      enforceWeightRule: cfg.enforceWeightRule
    });
    scenario = generated.scenario;
  }

  if (!scenario) return;

  replayController.scenario = scenario;
  replayController.results.naive = solveNaive(scenario);
  replayController.results.heuristic = solveHeuristic(scenario);
  replayController.results.optimal = solveOptimal(scenario);

  const counts = ["naive", "heuristic", "optimal"].map((key) => {
    const r = replayController.results[key];
    return r && r.status === "solved" ? r.moves.length : 0;
  });

  replayController.maxSteps = Math.max(0, ...counts);
  replayController.step = 0;
  replayController.playing = false;

  updateReplayPlayButton();
  renderReplayFrame();
}

function stateAfterMoves(scenario, solverResult, stepLimit) {
  let state = autoRetrieve(initialState(scenario.stacks)).state;
  if (!solverResult || solverResult.status !== "solved") return state;

  const limit = Math.min(stepLimit, solverResult.moves.length);
  for (let i = 0; i < limit; i++) {
    const m = solverResult.moves[i];
    const moved = applyRelocation(state, m.from, m.to, scenario.rules);
    if (!moved) break;
    state = autoRetrieve(moved.nextState).state;
  }
  return state;
}

function renderReplayFrame() {
  const scenario = replayController.scenario;
  if (!scenario) return;

  const solvers = ["naive", "heuristic", "optimal"];
  for (const key of solvers) {
    const yardEl = document.getElementById(`${key}Yard`);
    const countEl = document.getElementById(`${key}Count`);
    const statusEl = document.getElementById(`${key}Status`);
    const panelEl = document.querySelector(`.replay-panel[data-solver="${key}"]`);
    const res = replayController.results[key];

    if (!yardEl || !res) continue;

    if (res.status !== "solved") {
      const msg = res.reason === "too large for optimal solver"
        ? (currentLang === "ar" ? "الحجم أكبر من حد الحل الأمثل" : "Too large for optimal solver")
        : (currentLang === "ar" ? "تم بلوغ الحد الأقصى" : "Limit reached");

      yardEl.innerHTML = `<div style="padding:24px;text-align:center;color:var(--text-muted);font-size:0.875rem;">${msg}</div>`;
      if (countEl) countEl.textContent = "—";
      if (statusEl) {
        statusEl.textContent = "N/A";
        statusEl.className = "replay-panel-status";
      }
      if (panelEl) panelEl.classList.remove("solved");
      continue;
    }

    const currentYardState = stateAfterMoves(scenario, res, replayController.step);
    renderYardSvg(yardEl, currentYardState, {
      H: scenario.rules.H,
      interactive: false
    });

    const appliedCount = Math.min(replayController.step, res.moves.length);
    const done = replayController.step >= res.moves.length;

    if (countEl) countEl.textContent = String(appliedCount);
    if (statusEl) {
      statusEl.textContent = done
        ? (currentLang === "ar" ? "مكتمل ✓" : "Done ✓")
        : (currentLang === "ar" ? "قيد التشغيل…" : "Running…");
      statusEl.className = "replay-panel-status" + (done ? " won" : "");
    }
    if (panelEl) {
      panelEl.classList.toggle("solved", done);
    }
  }

  const stepEl = document.getElementById("replayStepCount");
  const totalEl = document.getElementById("replayTotal");
  if (stepEl) stepEl.textContent = String(replayController.step);
  if (totalEl) totalEl.textContent = String(replayController.maxSteps);
}

function stepReplayOnce() {
  if (replayController.step >= replayController.maxSteps) {
    stopReplayLoop();
    return;
  }
  replayController.step += 1;
  renderReplayFrame();

  if (replayController.step >= replayController.maxSteps) {
    stopReplayLoop();
  }
}

function toggleReplayPlayback() {
  if (replayController.playing) {
    stopReplayLoop();
    return;
  }
  if (replayController.step >= replayController.maxSteps) {
    replayController.step = 0;
  }
  replayController.playing = true;
  updateReplayPlayButton();
  startReplayLoop();
}

function startReplayLoop() {
  if (replayController.timerId) clearInterval(replayController.timerId);
  const intervalMs = Math.max(180, Math.round(950 / replayController.speed));
  replayController.timerId = setInterval(() => {
    if (replayController.step >= replayController.maxSteps) {
      stopReplayLoop();
      return;
    }
    replayController.step += 1;
    renderReplayFrame();
  }, intervalMs);
}

function stopReplayLoop() {
  replayController.playing = false;
  if (replayController.timerId) {
    clearInterval(replayController.timerId);
    replayController.timerId = null;
  }
  updateReplayPlayButton();
}

function resetReplayPlayback() {
  stopReplayLoop();
  replayController.step = 0;
  renderReplayFrame();
}

function updateReplayPlayButton() {
  const playBtn = document.getElementById("replayPlay");
  if (!playBtn) return;
  if (replayController.playing) {
    playBtn.innerHTML = currentLang === "ar" ? "⏸ إيقاف" : "⏸ Pause";
  } else {
    playBtn.innerHTML = currentLang === "ar" ? "▶ تشغيل" : "▶ Play";
  }
}

// ------------------------------------------------------------
// 4B. Chunked Async Benchmarks (Main 1000 & Small 200)
// ------------------------------------------------------------

function computeStats(values) {
  if (!values || values.length === 0) {
    return { avg: 0, min: 0, max: 0, total: 0 };
  }
  let total = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    total += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return {
    avg: total / values.length,
    min,
    max,
    total
  };
}

function runMainBenchmarkAsync(options = {}) {
  const totalCount = options.count || 1000;
  const chunkSize = options.chunkSize || 40;
  const enforceWeightRule = Boolean(options.enforceWeightRule);
  const onProgress = options.onProgress || (() => {});

  return new Promise((resolve) => {
    const naiveCounts = [];
    const heuristicCounts = [];
    const heuristicWorseList = [];
    let rejectedScenarios = 0;
    let index = 0;

    function nextBatch() {
      const end = Math.min(index + chunkSize, totalCount);

      for (; index < end; index++) {
        const seed = index + 1;
        const { scenario, rejectedCount } = generateScenario(seed, {
          N: 6,
          H: 4,
          enforceWeightRule,
          maxAttempts: 15
        });

        rejectedScenarios += rejectedCount;
        if (!scenario) {
          rejectedScenarios += 1;
          continue;
        }

        const naiveRes = solveNaive(scenario);
        const heurRes = solveHeuristic(scenario);

        if (naiveRes.status !== "solved" || heurRes.status !== "solved") {
          rejectedScenarios += 1;
          continue;
        }

        naiveCounts.push(naiveRes.relocations);
        heuristicCounts.push(heurRes.relocations);

        if (heurRes.relocations > naiveRes.relocations) {
          heuristicWorseList.push({
            seed,
            naive: naiveRes.relocations,
            heuristic: heurRes.relocations
          });
        }
      }

      onProgress(index, totalCount);

      if (index < totalCount) {
        setTimeout(nextBatch, 0);
      } else {
        const naiveStats = computeStats(naiveCounts);
        const heurStats = computeStats(heuristicCounts);
        const reductionPct = naiveStats.avg > 0
          ? ((naiveStats.avg - heurStats.avg) / naiveStats.avg) * 100
          : 0;

        resolve({
          type: "main",
          total: totalCount,
          solved: naiveCounts.length,
          rejected: rejectedScenarios,
          enforceWeightRule,
          naive: naiveStats,
          heuristic: heurStats,
          reductionPct,
          heuristicWorse: heuristicWorseList
        });
      }
    }

    setTimeout(nextBatch, 0);
  });
}

function runSmallBenchmarkAsync(options = {}) {
  const totalCount = options.count || 200;
  const chunkSize = options.chunkSize || 8;
  const enforceWeightRule = Boolean(options.enforceWeightRule);
  const onProgress = options.onProgress || (() => {});

  return new Promise((resolve) => {
    const naiveCounts = [];
    const heuristicCounts = [];
    const optimalCounts = [];
    const pairedForGap = [];
    const heuristicWorseList = [];
    let rejectedScenarios = 0;
    let optimalLimitCount = 0;
    let index = 0;

    function nextBatch() {
      const end = Math.min(index + chunkSize, totalCount);

      for (; index < end; index++) {
        const seed = 5000 + index;
        const { scenario, rejectedCount } = generateScenario(seed, {
          N: 4,
          H: 3,
          enforceWeightRule,
          containerCount: 7,
          maxAttempts: 15
        });

        rejectedScenarios += rejectedCount;
        if (!scenario) {
          rejectedScenarios += 1;
          continue;
        }

        const naiveRes = solveNaive(scenario);
        const heurRes = solveHeuristic(scenario);

        if (naiveRes.status !== "solved" || heurRes.status !== "solved") {
          rejectedScenarios += 1;
          continue;
        }

        naiveCounts.push(naiveRes.relocations);
        heuristicCounts.push(heurRes.relocations);

        if (heurRes.relocations > naiveRes.relocations) {
          heuristicWorseList.push({
            seed,
            naive: naiveRes.relocations,
            heuristic: heurRes.relocations
          });
        }

        const optRes = solveOptimal(scenario);
        if (optRes.status === "solved") {
          optimalCounts.push(optRes.relocations);
          pairedForGap.push({
            heur: heurRes.relocations,
            opt: optRes.relocations
          });
        } else if (optRes.status === "limit reached") {
          optimalLimitCount += 1;
        } else {
          rejectedScenarios += 1;
        }
      }

      onProgress(index, totalCount);

      if (index < totalCount) {
        setTimeout(nextBatch, 0);
      } else {
        const naiveStats = computeStats(naiveCounts);
        const heurStats = computeStats(heuristicCounts);
        const optStats = computeStats(optimalCounts);

        const reductionPct = naiveStats.avg > 0
          ? ((naiveStats.avg - heurStats.avg) / naiveStats.avg) * 100
          : 0;

        let gapSum = 0;
        let gapSamples = 0;
        for (const pair of pairedForGap) {
          if (pair.opt > 0) {
            gapSum += (pair.heur - pair.opt) / pair.opt;
            gapSamples += 1;
          }
        }
        const gapToOptimalPct = gapSamples > 0 ? (gapSum / gapSamples) * 100 : 0;

        resolve({
          type: "small",
          total: totalCount,
          solved: naiveCounts.length,
          rejected: rejectedScenarios,
          enforceWeightRule,
          naive: naiveStats,
          heuristic: heurStats,
          optimal: optStats,
          reductionPct,
          gapToOptimalPct,
          limitReachedCount: optimalLimitCount,
          pairedCount: pairedForGap.length,
          heuristicWorse: heuristicWorseList
        });
      }
    }

    setTimeout(nextBatch, 0);
  });
}

// ------------------------------------------------------------
// 4C. Hand-Written SVG Bar Chart
// ------------------------------------------------------------

function renderBenchmarkBarChart(containerEl, bars, title) {
  if (!containerEl) return;

  const width = 640;
  const height = 270;
  const pad = { top: 42, right: 24, bottom: 48, left: 54 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const maxVal = Math.max(1, ...bars.map((b) => b.value)) * 1.15;
  const gap = 36;
  const barW = Math.min(110, (plotW - gap * (bars.length - 1)) / bars.length);
  const groupW = bars.length * barW + (bars.length - 1) * gap;
  const startX = pad.left + (plotW - groupW) / 2;

  let gridLines = "";
  const steps = 4;
  for (let i = 0; i <= steps; i++) {
    const y = pad.top + plotH - (plotH * i) / steps;
    const val = ((maxVal * i) / steps).toFixed(1);
    gridLines += `<line x1="${pad.left}" y1="${y}" x2="${pad.left + plotW}" y2="${y}" stroke="var(--border)" stroke-width="1" />`;
    gridLines += `<text x="${pad.left - 8}" y="${y + 4}" text-anchor="end" fill="var(--text-muted)" font-size="11">${val}</text>`;
  }

  let barsSvg = "";
  bars.forEach((b, idx) => {
    const h = (b.value / maxVal) * plotH;
    const x = startX + idx * (barW + gap);
    const y = pad.top + plotH - h;

    barsSvg += `<rect x="${x}" y="${y}" width="${barW}" height="${h}" rx="6" fill="${b.color}" />`;
    barsSvg += `<text x="${x + barW / 2}" y="${y - 8}" text-anchor="middle" fill="var(--text)" font-size="13" font-weight="700">${b.value.toFixed(2)}</text>`;
    barsSvg += `<text x="${x + barW / 2}" y="${pad.top + plotH + 22}" text-anchor="middle" fill="var(--text-muted)" font-size="12" font-weight="600">${b.label}</text>`;
  });

  containerEl.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${title}">
      <text x="${width / 2}" y="22" text-anchor="middle" fill="var(--text)" font-size="14" font-weight="700">${title}</text>
      ${gridLines}
      ${barsSvg}
    </svg>
  `;
}

// ------------------------------------------------------------
// 4D. Benchmark UI & Crane Cost Estimator
// ------------------------------------------------------------

let latestBenchmarkResult = null;
let benchmarkRunning = false;

function setBenchmarkButtonsDisabled(disabled) {
  const mainBtn = document.getElementById("runMainBenchmark");
  const smallBtn = document.getElementById("runSmallBenchmark");
  if (mainBtn) mainBtn.disabled = disabled;
  if (smallBtn) smallBtn.disabled = disabled;
}

async function executeBenchmarkUI(mode) {
  if (benchmarkRunning) return;
  benchmarkRunning = true;
  setBenchmarkButtonsDisabled(true);

  const progressBox = document.getElementById("benchmarkProgress");
  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");
  const resultsBox = document.getElementById("benchmarkResults");
  const summaryEl = document.getElementById("resultsSummary");
  const chartEl = document.getElementById("resultsChart");
  const costBox = document.getElementById("costSection");
  const exportBox = document.getElementById("benchmarkExport");

  const weightToggle = document.getElementById("weightRuleToggle");
  const enforceWeightRule = weightToggle ? Boolean(weightToggle.checked) : false;

  if (progressBox) progressBox.style.display = "flex";
  if (resultsBox) resultsBox.style.display = "none";
  if (progressFill) progressFill.style.width = "0%";

  const onProgress = (done, total) => {
    if (progressFill) progressFill.style.width = `${Math.round((done / total) * 100)}%`;
    if (progressText) progressText.textContent = `${done} / ${total}`;
  };

  const result = mode === "small"
    ? await runSmallBenchmarkAsync({ count: 200, chunkSize: 8, enforceWeightRule, onProgress })
    : await runMainBenchmarkAsync({ count: 1000, chunkSize: 40, enforceWeightRule, onProgress });

  latestBenchmarkResult = result;
  window._lastBenchmark = result;

  if (progressBox) progressBox.style.display = "none";
  if (resultsBox) resultsBox.style.display = "block";
  if (costBox) costBox.style.display = "block";
  if (exportBox) exportBox.style.display = "flex";

  // Update Hero headline metric
  const heroVal = document.getElementById("heroMetric");
  const heroSub = document.querySelector(".metric-sub");
  if (heroVal) heroVal.textContent = `${result.reductionPct.toFixed(1)}%`;
  if (heroSub) {
    heroSub.textContent = currentLang === "ar"
      ? `تخفيض في المناورات مقارنة بالحل الساذج (${result.solved} سيناريو)`
      : `relocation reduction vs Naive (${result.solved} solved scenarios)`;
  }

  // Build summary cards
  const cards = [
    { label: currentLang === "ar" ? "متوسط الساذج" : "Naive Avg", val: result.naive.avg.toFixed(2) },
    { label: currentLang === "ar" ? "متوسط الاستدلالي" : "Heuristic Avg", val: result.heuristic.avg.toFixed(2) },
    { label: currentLang === "ar" ? "نسبة التخفيض" : "Reduction vs Naive", val: `${result.reductionPct.toFixed(1)}%`, highlight: true }
  ];

  if (result.type === "small") {
    cards.push({ label: currentLang === "ar" ? "متوسط الأمثل" : "Optimal Avg", val: result.optimal.avg.toFixed(2), highlight: true });
    cards.push({ label: currentLang === "ar" ? "الفجوة عن الأمثل" : "Gap to Optimal", val: `${result.gapToOptimalPct.toFixed(1)}%` });
    cards.push({ label: currentLang === "ar" ? "حدود الأمثل" : "Optimal Limit Hit", val: String(result.limitReachedCount) });
  } else {
    cards.push({ label: currentLang === "ar" ? "سيناريوهات محلولة" : "Solved Scenarios", val: String(result.solved) });
  }

  cards.push({ label: currentLang === "ar" ? "مرفوضة / غير قابلة للحل" : "Rejected", val: String(result.rejected) });

  let worseListHtml = "";
  if (result.heuristicWorse.length > 0) {
    const sampleItems = result.heuristicWorse
      .slice(0, 12)
      .map((item) => `Seed ${item.seed} (Naive: ${item.naive}, Heuristic: ${item.heuristic})`)
      .join(" • ");
    const heading = currentLang === "ar"
      ? `حالات تفوّق فيها الساذج على الاستدلالي (${result.heuristicWorse.length}):`
      : `Scenarios where Heuristic > Naive (${result.heuristicWorse.length}):`;
    worseListHtml = `<div style="grid-column: 1 / -1; padding: 12px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; font-size: 0.85rem; color: var(--text-muted);"><strong>${heading}</strong> ${sampleItems}</div>`;
  }

  if (summaryEl) {
    summaryEl.innerHTML = cards.map((c) => `
      <div class="result-card ${c.highlight ? "result-success" : ""}">
        <div class="result-label">${c.label}</div>
        <div class="result-value">${c.val}</div>
      </div>
    `).join("") + worseListHtml;
  }

  const chartBars = [
    { label: currentLang === "ar" ? "ساذج" : "Naive", value: result.naive.avg, color: "#ef4444" },
    { label: currentLang === "ar" ? "استدلالي" : "Heuristic", value: result.heuristic.avg, color: "#f59e0b" }
  ];
  if (result.type === "small") {
    chartBars.push({ label: currentLang === "ar" ? "أمثل" : "Optimal", value: result.optimal.avg, color: "#22c55e" });
  }

  renderBenchmarkBarChart(
    chartEl,
    chartBars,
    currentLang === "ar" ? "متوسط عدد المناورات لكل سيناريو" : "Average Relocations per Scenario"
  );

  updateCraneCostEstimate();
  benchmarkRunning = false;
  setBenchmarkButtonsDisabled(false);
}

function updateCraneCostEstimate() {
  const resultEl = document.getElementById("costResult");
  if (!resultEl || !latestBenchmarkResult) return;

  const minsPerMove = Math.max(0, parseFloat(document.getElementById("costMinutes")?.value) || 0);
  const hourlyRate = Math.max(0, parseFloat(document.getElementById("costHourly")?.value) || 0);

  const r = latestBenchmarkResult;
  const savedMovesPerScenario = r.naive.avg - r.heuristic.avg;
  const savedMinsPerScenario = savedMovesPerScenario * minsPerMove;
  const savedCostPerScenario = (savedMinsPerScenario / 60) * hourlyRate;

  const totalSavedMoves = savedMovesPerScenario * r.solved;
  const totalSavedMins = totalSavedMoves * minsPerMove;
  const totalSavedCost = (totalSavedMins / 60) * hourlyRate;

  if (currentLang === "ar") {
    resultEl.innerHTML = `
      <div><strong>لكل سيناريو:</strong> توفير ${savedMovesPerScenario.toFixed(2)} مناورة ≈ ${savedMinsPerScenario.toFixed(1)} دقيقة ≈ <strong>$${savedCostPerScenario.toFixed(2)}</strong></div>
      <div><strong>عبر ${r.solved} سيناريو:</strong> توفير ${totalSavedMoves.toFixed(0)} مناورة ≈ ${totalSavedMins.toFixed(0)} دقيقة ≈ <strong>$${totalSavedCost.toFixed(0)}</strong></div>
    `;
  } else {
    resultEl.innerHTML = `
      <div><strong>Per scenario:</strong> ${savedMovesPerScenario.toFixed(2)} relocations saved ≈ ${savedMinsPerScenario.toFixed(1)} min ≈ <strong>$${savedCostPerScenario.toFixed(2)}</strong></div>
      <div><strong>Across ${r.solved} scenarios:</strong> ${totalSavedMoves.toFixed(0)} relocations saved ≈ ${totalSavedMins.toFixed(0)} min ≈ <strong>$${totalSavedCost.toFixed(0)}</strong></div>
    `;
  }
}

function bindStage4Events() {
  const onClick = (id, fn) => {
    const el = document.getElementById(id);
    if (el && !el.dataset.bound) {
      el.dataset.bound = "1";
      el.addEventListener("click", fn);
    }
  };

  onClick("replayPlay", toggleReplayPlayback);
  onClick("replayStep", stepReplayOnce);
  onClick("replayReset", resetReplayPlayback);

  const speedEl = document.getElementById("replaySpeed");
  const speedLabel = document.getElementById("replaySpeedLabel");
  if (speedEl && !speedEl.dataset.bound) {
    speedEl.dataset.bound = "1";
    speedEl.addEventListener("input", (e) => {
      replayController.speed = parseInt(e.target.value, 10) || 1;
      if (speedLabel) speedLabel.textContent = `${replayController.speed}x`;
      if (replayController.playing) startReplayLoop();
    });
  }

  onClick("runMainBenchmark", () => executeBenchmarkUI("main"));
  onClick("runSmallBenchmark", () => executeBenchmarkUI("small"));

  const minsInput = document.getElementById("costMinutes");
  const hourlyInput = document.getElementById("costHourly");
  if (minsInput && !minsInput.dataset.bound) {
    minsInput.dataset.bound = "1";
    minsInput.addEventListener("input", updateCraneCostEstimate);
  }
  if (hourlyInput && !hourlyInput.dataset.bound) {
    hourlyInput.dataset.bound = "1";
    hourlyInput.addEventListener("input", updateCraneCostEstimate);
  }
}
// ============================================================
// Stage 5: CSV Export, Case Study Sync & Final INIT Entrypoint
// ============================================================

function escapeCsvField(value) {
  const text = String(value ?? "");
  if (/[",\r\n]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function getCostSummary(result) {
  const minsPerMove = Math.max(0, parseFloat(document.getElementById("costMinutes")?.value) || 0);
  const hourlyRate = Math.max(0, parseFloat(document.getElementById("costHourly")?.value) || 0);

  const savedMovesPerScenario = result.naive.avg - result.heuristic.avg;
  const savedMinsPerScenario = savedMovesPerScenario * minsPerMove;
  const savedCostPerScenario = (savedMinsPerScenario / 60) * hourlyRate;

  const totalSavedMoves = savedMovesPerScenario * result.solved;
  const totalSavedMins = totalSavedMoves * minsPerMove;
  const totalSavedCost = (totalSavedMins / 60) * hourlyRate;

  return {
    minsPerMove,
    hourlyRate,
    savedMovesPerScenario,
    savedMinsPerScenario,
    savedCostPerScenario,
    totalSavedMoves,
    totalSavedMins,
    totalSavedCost
  };
}

function buildBenchmarkCsv(result) {
  const cost = getCostSummary(result);

  const rows = [
    ["Metric", "Value"],
    ["Benchmark Mode", result.type],
    ["Weight Rule Enforced", result.enforceWeightRule ? "yes" : "no"],
    ["Requested Scenarios", result.total],
    ["Solved Scenarios", result.solved],
    ["Rejected / Unsolved", result.rejected],
    ["Naive Avg Relocations", result.naive.avg.toFixed(4)],
    ["Naive Min Relocations", result.naive.min],
    ["Naive Max Relocations", result.naive.max],
    ["Heuristic Avg Relocations", result.heuristic.avg.toFixed(4)],
    ["Heuristic Min Relocations", result.heuristic.min],
    ["Heuristic Max Relocations", result.heuristic.max],
    ["Relocation Reduction vs Naive (%)", result.reductionPct.toFixed(2)]
  ];

  if (result.type === "small") {
    rows.push(["Optimal Avg Relocations", result.optimal.avg.toFixed(4)]);
    rows.push(["Paired Optimal Samples", result.pairedCount]);
    rows.push(["Heuristic Gap to Optimal (%)", result.gapToOptimalPct.toFixed(2)]);
    rows.push(["Optimal Limit Hit Count", result.limitReachedCount]);
  }

  rows.push(
    ["Heuristic Worse Than Naive Count", result.heuristicWorse.length],
    ["Minutes per Relocation", cost.minsPerMove],
    ["Crane Hourly Cost", cost.hourlyRate],
    ["Saved Moves per Scenario", cost.savedMovesPerScenario.toFixed(4)],
    ["Saved Minutes per Scenario", cost.savedMinsPerScenario.toFixed(2)],
    ["Saved Cost per Scenario", cost.savedCostPerScenario.toFixed(2)],
    ["Total Saved Moves", cost.totalSavedMoves.toFixed(2)],
    ["Total Saved Minutes", cost.totalSavedMins.toFixed(2)],
    ["Total Saved Cost", cost.totalSavedCost.toFixed(2)]
  );

  return rows.map((row) => row.map(escapeCsvField).join(",")).join("\n");
}

function exportBenchmarkCsv() {
  const benchmarkData = latestBenchmarkResult || window._lastBenchmark;
  if (!benchmarkData) {
    alert(
      currentLang === "ar"
        ? "يرجى تشغيل البنشمارك أولاً قبل تصدير ملف CSV."
        : "Please run a benchmark first before exporting CSV."
    );
    return;
  }

  const csvWithBom = "\uFEFF" + buildBenchmarkCsv(benchmarkData);
  const blob = new Blob([csvWithBom], { type: "text/csv;charset=utf-8;" });
  const objectUrl = URL.createObjectURL(blob);

  const link = document.createElement("a");
  const dateStamp = new Date().toISOString().slice(0, 10);
  link.href = objectUrl;
  link.download = `yard-relocation-${benchmarkData.type}-${dateStamp}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

function syncCaseStudySection(result) {
  const targetEl = document.getElementById("caseStudyDynamic");
  if (!targetEl || !result) return;

  const cost = getCostSummary(result);

  if (currentLang === "ar") {
    const scopeLabel = result.type === "small"
      ? "قياس مصغّر (200 سيناريو، ساحة 4×3)"
      : "قياس رئيسي (1,000 سيناريو، ساحة 6×4)";

    const optimalBullet = result.type === "small"
      ? `<li><strong>الفجوة عن الحل الأمثل (BFS):</strong> ${result.gapToOptimalPct.toFixed(1)}% (متوسط الأمثل ${result.optimal.avg.toFixed(2)} مقابل الاستدلالي ${result.heuristic.avg.toFixed(2)}).</li>`
      : "";

    targetEl.innerHTML = `
      <h3>ملخص النتائج — ${scopeLabel}</h3>
      <ul>
        <li><strong>متوسط المناورات:</strong> الحل الساذج ${result.naive.avg.toFixed(2)} مقابل الاستدلالي ${result.heuristic.avg.toFixed(2)}، بتخفيض قدره <strong>${result.reductionPct.toFixed(1)}%</strong>.</li>
        ${optimalBullet}
        <li><strong>الأثر الزمني والمالي:</strong> توفير ${cost.savedMovesPerScenario.toFixed(2)} مناورة لكل سيناريو ≈ ${cost.savedMinsPerScenario.toFixed(1)} دقيقة ≈ ${cost.savedCostPerScenario.toFixed(2)} وحدة تكلفة.</li>
        <li><strong>حالات تفوّق الحل الساذج:</strong> حدثت في ${result.heuristicWorse.length} سيناريو نتيجة اعتماد القاعدة الاستدلالية على قرار موضعي مباشر.</li>
      </ul>
      <h3>القيود الفنية والافتراضات</h3>
      <ul>
        <li>تعتمد السيناريوهات على توليد حتمي بالبذرة مع معرفة تسلسل السحب مسبقاً، دون نمذجة تأخر الشاحنات العشوائي أو تغيّر الأولويات أثناء التشغيل.</li>
        <li>تُفترض سعة قصوى موحدة لجميع الأعمدة ويُحتسب عدد المناورات مباشرة دون محاكاة ازدحام الرافعات أو مسارات الساحة.</li>
        <li>يقتصر الحل الأمثل (BFS) على الحالات الصغيرة (≤ 8 حاويات، N ≤ 4، H ≤ 3) بحدود استكشاف ووقت صارمة لتفادي انفجار الحالات.</li>
      </ul>
    `;
  } else {
    const scopeLabel = result.type === "small"
      ? "Small Benchmark (200 scenarios, 4×3 yard)"
      : "Main Benchmark (1,000 scenarios, 6×4 yard)";

    const optimalBullet = result.type === "small"
      ? `<li><strong>Gap to Optimal (BFS):</strong> ${result.gapToOptimalPct.toFixed(1)}% (Optimal mean ${result.optimal.avg.toFixed(2)} vs Heuristic ${result.heuristic.avg.toFixed(2)}).</li>`
      : "";

    targetEl.innerHTML = `
      <h3>Empirical Findings — ${scopeLabel}</h3>
      <ul>
        <li><strong>Mean Relocations:</strong> Naive ${result.naive.avg.toFixed(2)} vs Heuristic ${result.heuristic.avg.toFixed(2)}, a <strong>${result.reductionPct.toFixed(1)}%</strong> reduction.</li>
        ${optimalBullet}
        <li><strong>Time & Cost Impact:</strong> ${cost.savedMovesPerScenario.toFixed(2)} saved relocations per scenario ≈ ${cost.savedMinsPerScenario.toFixed(1)} min ≈ ${cost.savedCostPerScenario.toFixed(2)} cost units.</li>
        <li><strong>Naive Outperformance Cases:</strong> Occurred in ${result.heuristicWorse.length} scenarios due to short-horizon greedy choices.</li>
      </ul>
      <h3>Limitations & Scope</h3>
      <ul>
        <li>Uses deterministic seeded synthetic instances with known pickup orders rather than stochastic arrival delays or live TOS feeds.</li>
        <li>Assumes uniform stack height limits and counts relocations directly without modeling multi-crane traffic or yard routing.</li>
        <li>Exact BFS is bounded to small instances (≤ 8 containers, N ≤ 4, H ≤ 3) with state and time limits to avoid combinatorial blow-up.</li>
      </ul>
    `;
  }
}

const _baseExecuteBenchmarkUI = executeBenchmarkUI;
executeBenchmarkUI = async function (mode) {
  await _baseExecuteBenchmarkUI(mode);
  if (latestBenchmarkResult) {
    syncCaseStudySection(latestBenchmarkResult);
  }
};

const _baseUpdateCraneCostEstimate = updateCraneCostEstimate;
updateCraneCostEstimate = function () {
  _baseUpdateCraneCostEstimate();
  if (latestBenchmarkResult) {
    syncCaseStudySection(latestBenchmarkResult);
  }
};

function bindStage5Events() {
  const onClick = (id, handler) => {
    const el = document.getElementById(id);
    if (el && !el.dataset.bound) {
      el.dataset.bound = "1";
      el.addEventListener("click", handler);
    }
  };

  onClick("btnExportCsv", exportBenchmarkCsv);
  onClick("btnPrintReport", () => window.print());

  onClick("themeToggle", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    applyTheme(currentTheme === "dark" ? "light" : "dark");
  });

  onClick("langToggle", () => {
    applyLanguage(currentLang === "en" ? "ar" : "en");
    if (typeof renderReplayFrame === "function" && replayController.scenario) {
      renderReplayFrame();
    }
    if (latestBenchmarkResult) {
      syncCaseStudySection(latestBenchmarkResult);
    }
  });
}

function INIT() {
  let savedLang = "en";
  let savedTheme = "dark";

  try {
    savedLang = localStorage.getItem("yard_lang") || "en";
    savedTheme = localStorage.getItem("yard_theme") || "dark";
  } catch (_) {}

  applyTheme(savedTheme);
  applyLanguage(savedLang);

  bindStage3Events();
  bindStage4Events();
  bindStage5Events();

  loadDemoScenario(42);
  if (typeof setupReplayScenario === "function" && manualDemo.scenario) {
    setupReplayScenario(manualDemo.scenario);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", INIT);
} else {
  INIT();
}
