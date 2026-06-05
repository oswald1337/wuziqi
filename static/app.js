const agentSelect = document.querySelector("#agentSelect");
const sideSelect = document.querySelector("#sideSelect");
const newGameButton = document.querySelector("#newGameButton");
const resignButton = document.querySelector("#resignButton");
const boardEl = document.querySelector("#board");
const statusText = document.querySelector("#statusText");
const agentName = document.querySelector("#agentName");
const agentElo = document.querySelector("#agentElo");
const boardFact = document.querySelector("#boardFact");
const turnFact = document.querySelector("#turnFact");
const resultFact = document.querySelector("#resultFact");
const presetBadge = document.querySelector("#presetBadge");
const metricElo = document.querySelector("#metricElo");
const metricGames = document.querySelector("#metricGames");
const metricReplay = document.querySelector("#metricReplay");
const metricExpert = document.querySelector("#metricExpert");
const metricAnchor = document.querySelector("#metricAnchor");
const metricPuzzles = document.querySelector("#metricPuzzles");
const metricPlayouts = document.querySelector("#metricPlayouts");
const metricMode = document.querySelector("#metricMode");
const metricLoss = document.querySelector("#metricLoss");
const metricPolicyLoss = document.querySelector("#metricPolicyLoss");
const metricValueLoss = document.querySelector("#metricValueLoss");
const metricEntropy = document.querySelector("#metricEntropy");
const metricModel = document.querySelector("#metricModel");
const metricPromotion = document.querySelector("#metricPromotion");
const metricGate = document.querySelector("#metricGate");
const metricInit = document.querySelector("#metricInit");
const metricSeed = document.querySelector("#metricSeed");
const gateChecks = document.querySelector("#gateChecks");
const lossChart = document.querySelector("#lossChart");
const evalRows = document.querySelector("#evalRows");
const selfPlayList = document.querySelector("#selfPlayList");
const goalStatus = document.querySelector("#goalStatus");
const targetElo = document.querySelector("#targetElo");
const bestElo = document.querySelector("#bestElo");
const heuristicGate = document.querySelector("#heuristicGate");
const previousGate = document.querySelector("#previousGate");
const promotionGate = document.querySelector("#promotionGate");
const logPath = document.querySelector("#logPath");
const registryPath = document.querySelector("#registryPath");
const configPath = document.querySelector("#configPath");
const leaderboardRows = document.querySelector("#leaderboardRows");
const logTail = document.querySelector("#logTail");
const refreshStatusButton = document.querySelector("#refreshStatusButton");
const lastRefresh = document.querySelector("#lastRefresh");

let agents = [];
let gameState = null;
let trainingStatus = null;
let busy = false;
let statusBusy = false;

async function api(path, payload) {
  const response = await fetch(path, {
    method: payload ? "POST" : "GET",
    headers: payload ? { "Content-Type": "application/json" } : {},
    body: payload ? JSON.stringify(payload) : undefined,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function selectedAgent() {
  return agents.find((agent) => agent.id === agentSelect.value) ||
    (gameState ? gameState.agent : null);
}

function activeAgent() {
  if (gameState) return gameState.agent;
  return selectedAgent();
}

function formatNumber(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(digits);
}

function playerName(player) {
  if (!gameState) return "--";
  if (player === gameState.human_player) return "You";
  if (player === gameState.agent_player) return "Agent";
  return `Player ${player}`;
}

function resultText() {
  if (!gameState) return "--";
  if (gameState.status === "active") return "In progress";
  if (gameState.status === "resigned") return "Resigned";
  if (gameState.winner === -1) return "Draw";
  if (gameState.winner === gameState.human_player) return "You won";
  if (gameState.winner === gameState.agent_player) return "Agent won";
  return "Ended";
}

function updateFacts() {
  if (!gameState) {
    boardFact.textContent = "--";
    turnFact.textContent = "--";
    resultFact.textContent = "--";
    return;
  }

  boardFact.textContent = `${gameState.width}x${gameState.height}, ${gameState.n_in_row} in row`;
  turnFact.textContent = gameState.status === "active"
    ? playerName(gameState.current_player)
    : "--";
  resultFact.textContent = resultText();

  const selected = agents.find((agent) => agent.id === gameState.agent.id);
  agentName.textContent = selected ? selected.name : gameState.agent.name;
  agentElo.textContent = selected ? `${selected.elo}` : `${gameState.agent.elo || "--"}`;

  if (gameState.status !== "active") {
    statusText.textContent = resultText();
  } else if (gameState.current_player === gameState.human_player) {
    statusText.textContent = "Your turn";
  } else {
    statusText.textContent = "Agent turn";
  }
}

function renderLossChart(training) {
  lossChart.innerHTML = "";
  if (!training || training.length === 0) {
    lossChart.textContent = "No training loss yet";
    return;
  }

  const width = 260;
  const height = 90;
  const pad = 10;
  const losses = training.map((item) => Number(item.loss));
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const span = max - min || 1;
  const points = training.map((item, index) => {
    const x = pad + (training.length === 1 ? 0.5 : index / (training.length - 1)) * (width - pad * 2);
    const y = height - pad - ((Number(item.loss) - min) / span) * (height - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  lossChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Loss over training games">
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" />
      <polyline points="${points}" />
      ${training.map((item, index) => {
        const [x, y] = points.split(" ")[index].split(",");
        return `<circle cx="${x}" cy="${y}" r="2.8"><title>Game ${item.game}: loss ${formatNumber(item.loss, 3)}</title></circle>`;
      }).join("")}
    </svg>
    <div class="chart-range">
      <span>${formatNumber(max, 3)}</span>
      <span>${formatNumber(min, 3)}</span>
    </div>
  `;
}

function renderEvalRows(evalMetrics) {
  evalRows.innerHTML = "";
  if (!evalMetrics || Object.keys(evalMetrics).length === 0) {
    evalRows.innerHTML = `<div class="eval-empty">No eval yet</div>`;
    return;
  }

  for (const [name, record] of Object.entries(evalMetrics)) {
    const row = document.createElement("div");
    row.className = "eval-row";
    row.innerHTML = `
      <span>${name}</span>
      <span>${record.wins ?? "--"}</span>
      <span>${record.draws ?? "--"}</span>
      <span>${record.losses ?? "--"}</span>
      <span>${formatNumber((record.score ?? 0) * 100, 0)}%</span>
    `;
    evalRows.appendChild(row);
  }
}

function renderSelfPlay(selfPlay) {
  selfPlayList.innerHTML = "";
  const recent = (selfPlay || []).slice(-6).reverse();
  if (recent.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No self-play games recorded";
    selfPlayList.appendChild(item);
    return;
  }

  for (const game of recent) {
    const item = document.createElement("li");
    const loss = game.loss === undefined ? "" : ` · loss ${formatNumber(game.loss, 3)}`;
    const mode = game.self_play_mode ? ` · ${game.self_play_mode}` : "";
    const forced = game.forced_tactical_moves === undefined
      ? ""
      : ` · forced ${game.forced_tactical_moves}`;
    const twoPly = game.two_ply_threat_moves === undefined
      ? ""
      : ` · 2-ply ${game.two_ply_threat_moves}`;
    const beam = game.beam_moves === undefined
      ? ""
      : ` · beam ${game.beam_moves}`;
    const candidates = game.candidate_evaluations === undefined
      ? ""
      : ` · evals ${game.candidate_evaluations}`;
    item.textContent = `g${game.game}: winner ${game.winner}, ${game.moves} moves${mode}${forced}${twoPly}${beam}${candidates}${loss}`;
    selfPlayList.appendChild(item);
  }
}

function renderGateChecks(promotion) {
  gateChecks.innerHTML = "";
  let rendered = 0;
  const promotionChecks = promotion && promotion.promotion_checks;
  if (promotionChecks) {
    for (const [name, passed] of Object.entries(promotionChecks)) {
      const item = document.createElement("span");
      item.className = passed ? "gate-pass" : "gate-fail";
      item.textContent = `promotion ${name.replaceAll("_", " ")}: ${passed ? "pass" : "fail"}`;
      gateChecks.appendChild(item);
      rendered += 1;
    }
  }

  const gate = promotion && (promotion.completion_gate || promotion.average_human_gate);
  if (!gate || !gate.checks) {
    if (!rendered) gateChecks.textContent = "No gate evaluation yet";
    return;
  }

  for (const [name, passed] of Object.entries(gate.checks)) {
    const item = document.createElement("span");
    item.className = passed ? "gate-pass" : "gate-fail";
    item.textContent = `completion ${name.replaceAll("_", " ")}: ${passed ? "pass" : "fail"}`;
    gateChecks.appendChild(item);
  }
}

function renderTelemetry() {
  const agent = activeAgent();
  const metrics = agent ? (agent.metrics || {}) : {};
  const config = metrics.config || agent || {};
  const history = metrics.history || {};
  const lastTrain = metrics.last_train || {};
  const promotion = metrics.promotion || {};
  const initFrom = metrics.init_from || config.init_from;
  const trainingState = metrics.training_state || {};

  presetBadge.textContent = config.preset || agent?.type || "--";
  metricElo.textContent = agent?.elo ?? "--";
  metricGames.textContent = agent?.games_trained ?? config.self_play_games ?? "--";
  metricReplay.textContent = trainingState.replay_samples !== undefined
    ? `${trainingState.replay_samples} samples`
    : "--";
  metricExpert.textContent = config.expert_games !== undefined
    ? `${config.expert_games} games`
    : "--";
  metricAnchor.textContent = config.anchor_samples !== undefined
    ? `${config.anchor_samples} samples`
    : "--";
  metricPuzzles.textContent = config.tactical_puzzles !== undefined
    ? `${config.tactical_puzzles} positions`
    : "--";
  metricPlayouts.textContent = agent?.n_playout ?? config.eval_n_playout ?? "--";
  metricMode.textContent = config.self_play_mode || "--";
  metricLoss.textContent = formatNumber(lastTrain.loss, 3);
  metricPolicyLoss.textContent = formatNumber(lastTrain.policy_loss, 3);
  metricValueLoss.textContent = formatNumber(lastTrain.value_loss, 3);
  metricEntropy.textContent = formatNumber(lastTrain.entropy, 3);
  metricModel.textContent = config.num_res_blocks && config.num_filters
    ? `${config.architecture || agent?.architecture || "residual"} ${config.num_res_blocks}x${config.num_filters}`
    : "--";
  metricPromotion.textContent = promotion.promoted === undefined
    ? "--"
    : (promotion.promoted ? "promoted" : "rejected");
  metricGate.textContent = promotion.gate_passed === undefined
    ? "--"
    : (promotion.gate_passed ? "passed" : "not yet");
  metricInit.textContent = initFrom ? (initFrom.name || initFrom.id || "checkpoint") : "random";
  metricSeed.textContent = config.seed ?? "--";

  renderGateChecks(promotion);
  renderLossChart(history.training);
  renderEvalRows(metrics.eval);
  renderSelfPlay(history.self_play);
}

function renderLeaderboard(rows) {
  leaderboardRows.innerHTML = "";
  if (!rows || rows.length === 0) {
    leaderboardRows.innerHTML = `<div class="ladder-empty">No 16x16 checkpoints yet</div>`;
    return;
  }

  for (const checkpoint of rows.slice(0, 8)) {
    const promotion = checkpoint.metrics && checkpoint.metrics.promotion;
    const status = promotion
      ? (promotion.gate_passed ? "gate" : (promotion.promoted ? "promoted" : "rejected"))
      : "unevaluated";
    const row = document.createElement("button");
    row.type = "button";
    row.className = "ladder-row ladder-button";
    row.innerHTML = `
      <span>${checkpoint.name}</span>
      <span>${checkpoint.elo ?? "--"}</span>
      <span>${status}</span>
    `;
    row.addEventListener("click", () => {
      agentSelect.value = checkpoint.id;
      newGame();
    });
    leaderboardRows.appendChild(row);
  }
}

function renderTrainingStatus() {
  if (!trainingStatus) {
    goalStatus.textContent = "active";
    targetElo.textContent = "--";
    bestElo.textContent = "--";
    heuristicGate.textContent = "--";
    previousGate.textContent = "--";
    promotionGate.textContent = "--";
    logPath.textContent = "--";
    registryPath.textContent = "--";
    configPath.textContent = "--";
    leaderboardRows.innerHTML = "";
    logTail.textContent = "";
    return;
  }

  const gate = trainingStatus.goal?.completion_gate || {};
  const best = trainingStatus.best_agent || {};
  goalStatus.textContent = trainingStatus.goal?.status || "active";
  targetElo.textContent = gate.elo_target ?? "--";
  bestElo.textContent = best.elo ?? "--";
  heuristicGate.textContent = gate.heuristic_min_score !== undefined
    ? `${formatNumber(gate.heuristic_min_score * 100, 0)}% / ${gate.heuristic_min_games}g`
    : "--";
  previousGate.textContent = gate.previous_best_min_score !== undefined
    ? `${formatNumber(gate.previous_best_min_score * 100, 0)}% / ${gate.previous_best_min_games}g`
    : "--";
  const promotion = trainingStatus.goal?.promotion_gate || {};
  promotionGate.textContent = promotion.heuristic_min_score !== undefined
    ? `${formatNumber(promotion.heuristic_min_score * 100, 0)}% / ${promotion.heuristic_min_games}g · prev ${formatNumber(promotion.previous_best_min_score * 100, 0)}% / ${promotion.previous_best_min_games}g · Elo +${promotion.min_elo_delta ?? 0}`
    : "--";
  logPath.textContent = trainingStatus.event_log_path
    ? `${trainingStatus.log_path || "--"} + ${trainingStatus.event_log_path}`
    : (trainingStatus.log_path || "--");
  registryPath.textContent = trainingStatus.registry_path || "--";
  configPath.textContent = trainingStatus.config_path || "--";
  renderLeaderboard(trainingStatus.leaderboard);
  logTail.textContent = (trainingStatus.log_tail || []).slice(-24).join("\n");
}

function stoneAt(row, col) {
  if (!gameState) return null;
  return gameState.stones.find((stone) => stone.row === row && stone.col === col);
}

function isLast(row, col) {
  return gameState &&
    gameState.last_move &&
    gameState.last_move.row === row &&
    gameState.last_move.col === col;
}

function renderBoard() {
  boardEl.innerHTML = "";
  if (!gameState) return;

  boardEl.style.setProperty("--board-size", gameState.width);
  for (let row = 0; row < gameState.height; row += 1) {
    for (let col = 0; col < gameState.width; col += 1) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "cell";
      cell.dataset.row = row;
      cell.dataset.col = col;
      cell.setAttribute("aria-label", `Row ${row + 1}, column ${col + 1}`);

      const stone = stoneAt(row, col);
      if (stone) {
        cell.classList.add(stone.player === 1 ? "black" : "white");
      }
      if (isLast(row, col)) {
        cell.classList.add("last");
      }

      const humanTurn = gameState.status === "active" &&
        gameState.current_player === gameState.human_player;
      cell.disabled = busy || !humanTurn || Boolean(stone);
      cell.addEventListener("click", () => playMove(row, col));
      boardEl.appendChild(cell);
    }
  }
}

function render() {
  updateFacts();
  renderTelemetry();
  renderTrainingStatus();
  renderBoard();
  resignButton.disabled = busy || !gameState || gameState.status !== "active";
  newGameButton.disabled = busy;
}

async function loadAgents() {
  const data = await api("/api/checkpoints");
  const priorValue = agentSelect.value;
  agents = data.agents;
  agentSelect.innerHTML = "";
  for (const agent of agents) {
    const option = document.createElement("option");
    option.value = agent.id;
    option.textContent = `${agent.name} · Elo ${agent.elo}`;
    agentSelect.appendChild(option);
  }
  const stillAvailable = agents.some((agent) => agent.id === priorValue);
  agentSelect.value = stillAvailable ? priorValue : data.default_agent_id;
}

async function loadTrainingStatus() {
  trainingStatus = await api("/api/training-status");
  lastRefresh.textContent = new Date().toLocaleTimeString();
}

async function refreshStatus() {
  if (statusBusy) return;
  statusBusy = true;
  refreshStatusButton.disabled = true;
  try {
    await loadAgents();
    await loadTrainingStatus();
    if (gameState) {
      const active = agents.find((agent) => agent.id === gameState.agent.id);
      if (active) {
        gameState.agent = active;
      }
    }
  } catch (error) {
    logTail.textContent = `Could not refresh status: ${error.message}`;
  } finally {
    statusBusy = false;
    refreshStatusButton.disabled = false;
    render();
  }
}

async function newGame() {
  busy = true;
  render();
  try {
    gameState = await api("/api/new-game", {
      agent_id: agentSelect.value,
      human_player: Number(sideSelect.value),
    });
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    busy = false;
    render();
  }
}

async function playMove(row, col) {
  if (!gameState || busy) return;
  busy = true;
  render();
  try {
    gameState = await api("/api/move", {
      game_id: gameState.game_id,
      row,
      col,
    });
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    busy = false;
    render();
  }
}

async function resign() {
  if (!gameState || busy) return;
  busy = true;
  render();
  try {
    gameState = await api("/api/resign", { game_id: gameState.game_id });
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    busy = false;
    render();
  }
}

newGameButton.addEventListener("click", newGame);
resignButton.addEventListener("click", resign);
agentSelect.addEventListener("change", newGame);
sideSelect.addEventListener("change", newGame);
refreshStatusButton.addEventListener("click", refreshStatus);

(async function boot() {
  try {
    await loadAgents();
    await loadTrainingStatus();
    await newGame();
    window.setInterval(refreshStatus, 5000);
  } catch (error) {
    statusText.textContent = error.message;
  }
})();
