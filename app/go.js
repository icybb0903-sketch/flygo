import * as THREE from "/vendor/three.module.js";

const BOARD_SIZE = 9;
const GRID_STEP = 0.8;
const GRID_SPAN = GRID_STEP * (BOARD_SIZE - 1);
const BOARD_Y = 0.84;
const API_STATE = "/api/go/state";
const API_ACTION = "/api/go/action";
const API_NEURAL_STATUS = "/api/neural/status";
const API_NEURAL_FRAME = "/api/neural/frame";
const API_NEURAL_ANATOMY = "/api/neural/anatomy/soma_xyz.npy";
const API_NEURAL_POLICY_STATUS = "/api/go/neural-policy-status";
const API_EVALUATION_STATUS = "/api/go/evaluation-status";
const API_TOURNAMENT_STATUS = "/api/go/tournament-status";
const API_STAGE7_STATUS = "/api/go/stage7-status";
const API_STAGE8_STATUS = "/api/go/stage8-status";
const API_NEURAL_MOVE = "/api/go/neural-move";
const NEURAL_TICK_DISPLAY_MS = 18;

const els = Object.fromEntries([
  "roomViewport", "roomCanvas", "sceneError", "serviceDot", "serviceStatus",
  "turnLabel", "flyStage", "stageDot", "moveNumber", "lastMove", "boardHash",
  "decisionId", "neuralDecisionId", "neuralBoardHash", "encodingHash", "modelId", "frameId",
  "graphSize", "displaySize", "spikeSummary", "controllerLabel", "checkpointLabel",
  "policyStatusLabel", "policyActionLabel", "evaluationStatusLabel", "evaluationMetricLabel",
  "whiteTournamentLabel", "generalTournamentLabel", "signalZeroControlLabel", "signalShuffleControlLabel",
  "neuralStatusChip", "neuralViewport", "neuralCanvas", "neuralTick", "neuralTime",
  "neuralEmptyState", "neuralEmptyTitle", "neuralEmptyDetail", "neuralPlaybackState",
  "neuralSpikeState", "rerunNeuralButton", "brainAnatomyStatus",
  "interactionStatus", "rulesStatus", "undoButton", "resetButton", "passButton", "resignButton", "botMoveButton",
  "pauseButton", "stepButton", "speedControl", "speedOutput", "cameraReset",
  "fullscreenButton", "resultBanner", "resultTitle", "resultScore"
].map((id) => [id, document.getElementById(id)]));

let scene;
let camera;
let renderer;
let raycaster;
let boardHitPlane;
let hoverMarker;
let lastMoveMarker;
let stonesGroup;
let fly;
let carriedStone;
let animationFrame = 0;
let lastFrameTime = performance.now();
let currentState = null;
let currentAnimation = null;
let animationPaused = false;
let stepRequested = false;
let requestBusy = false;
let pointerDown = null;
let pointerMoved = false;
let neuralRuntimeAvailable = false;
let neuralPolicyAvailable = false;
let neuralPolicyStatus = null;
let neuralFrame = null;
let neuralFrameAbort = null;
let neuralRequestSerial = 0;
let neuralRequestBusy = false;
let neuralTargetStateKey = null;
let neuralAcceptedStateKey = null;
let neuralPlaybackStartedAt = 0;
let neuralLastDrawnTick = -1;
let neuralRetryTimer = 0;
let preserveNeuralFrameForStateKey = null;
let neuralBrainScene;
let neuralBrainCamera;
let neuralBrainRenderer;
let neuralBrainNetwork;
let neuralBrainPoints;
let neuralBrainLines;
let neuralBrainActivity;
let neuralBrainFiring;
let neuralBrainPositions = null;
let neuralBrainModelId = null;
let neuralBrainReady = false;
let neuralBrainYaw = -0.28;
let neuralBrainPitch = -0.08;
let neuralBrainDrag = null;
let neuralBrainLastRender = 0;

const cameraOrbit = {
  yaw: 0.72,
  pitch: 0.83,
  distance: 15,
  target: new THREE.Vector3(0, 0.75, 0),
};

const FLY_REST = new THREE.Vector3(4.6, 1.52, 2.1);
const FLY_BOWL = new THREE.Vector3(4.6, 1.38, -2.45);
const BOWL_PICKUP = new THREE.Vector3(4.35, 1.28, -2.85);
const FLY_OUTSIDE_EDGE = 4.6;

const ACTION_STAGES = [
  { key: "observe", label: "观察棋盘", duration: 650 },
  { key: "think", label: "摆动触角，做出决策", duration: 950 },
  { key: "approach", label: "走向棋罐", duration: 760 },
  { key: "grasp", label: "双前足夹住棋子", duration: 520 },
  { key: "carry", label: "移动到目标交叉点", duration: 920 },
  { key: "release", label: "释放棋子", duration: 430 },
  { key: "retract", label: "收回前足", duration: 360 },
  { key: "return", label: "返回观察位置", duration: 760 },
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function easeInOut(value) {
  const t = clamp(value, 0, 1);
  return t * t * (3 - 2 * t);
}

function displayValue(value, fallback = "不可获取") {
  if (value === undefined || value === null || value === "") return fallback;
  return String(value);
}

function shortValue(value, length = 16) {
  const text = displayValue(value);
  return text.length > length ? `${text.slice(0, length)}…` : text;
}

function extractSnapshot(payload) {
  if (!payload || typeof payload !== "object") throw new Error("服务返回了无法识别的数据");
  if (payload.ok === false) {
    const error = payload.error;
    throw new Error(typeof error === "string" ? error : (error?.message || payload.message || "请求失败"));
  }
  return payload.session ?? payload.state ?? payload.go ?? payload.snapshot ?? payload;
}

function normalizeStone(value) {
  if (value === null || value === undefined || value === 0 || value === "." || value === "empty") return null;
  if (value === 1 || value === "1" || value === "black" || value === "b" || value === "B" || value === "●") return "black";
  if (value === 2 || value === -1 || value === "2" || value === "-1" || value === "white" || value === "w" || value === "W" || value === "○") return "white";
  return null;
}

function normalizeBoard(rawBoard) {
  if (!Array.isArray(rawBoard) || rawBoard.length !== BOARD_SIZE) return null;
  const board = rawBoard.map((row) => {
    if (typeof row === "string") return [...row].map(normalizeStone);
    return Array.isArray(row) ? row.map(normalizeStone) : [];
  });
  return board.every((row) => row.length === BOARD_SIZE) ? board : null;
}

function normalizeColor(value) {
  if (value === 1 || value === "1" || String(value).toLowerCase() === "black" || String(value).toLowerCase() === "b") return "black";
  if (value === 2 || value === -1 || value === "2" || value === "-1" || String(value).toLowerCase() === "white" || String(value).toLowerCase() === "w") return "white";
  return value == null ? null : String(value);
}

function normalizeMove(rawMove, fallbackColor = null) {
  if (!rawMove) return null;
  if (Array.isArray(rawMove) && rawMove.length >= 2) {
    return { row: Number(rawMove[0]), col: Number(rawMove[1]), color: fallbackColor };
  }
  if (typeof rawMove !== "object") return null;
  const recordColor = normalizeColor(rawMove.color ?? rawMove.player ?? rawMove.side ?? fallbackColor);
  if (rawMove.move !== undefined) {
    const nested = normalizeMove(rawMove.move, recordColor);
    if (nested) return { ...nested, color: nested.color ?? recordColor };
  }
  if (rawMove.pass === true || rawMove.is_pass === true || rawMove.action === "pass") return { pass: true, color: recordColor };
  const row = Number(rawMove.row ?? rawMove.r ?? rawMove.y);
  const col = Number(rawMove.col ?? rawMove.column ?? rawMove.c ?? rawMove.x);
  if (!Number.isInteger(row) || !Number.isInteger(col)) return null;
  return { row, col, color: normalizeColor(rawMove.color ?? fallbackColor) };
}

function normalizeState(snapshot) {
  if (!snapshot || typeof snapshot !== "object") throw new Error("棋局快照不存在");
  const game = snapshot.game && typeof snapshot.game === "object" ? snapshot.game : snapshot;
  const board = normalizeBoard(game.board ?? snapshot.board ?? game.position?.board);
  if (!board) throw new Error("棋局快照缺少有效的 9×9 棋盘");
  const toMove = normalizeColor(game.to_move ?? game.turn ?? snapshot.to_move);
  const moveNumber = game.move_number ?? game.move_count ?? snapshot.move_number;
  return {
    raw: snapshot,
    board,
    toMove,
    humanColor: normalizeColor(game.human_color ?? snapshot.human_color ?? "black"),
    moveNumber,
    lastMove: normalizeMove(game.last_move ?? snapshot.last_move),
    canUndo: Boolean(game.can_undo ?? snapshot.can_undo ?? (Number(moveNumber) > 0)),
    decisionId: game.decision_id ?? snapshot.decision_id ?? snapshot.decision?.id,
    boardHash: game.board_hash ?? snapshot.board_hash,
    controller: snapshot.controller ?? game.controller ?? snapshot.policy?.name,
    checkpoint: snapshot.checkpoint ?? game.checkpoint ?? snapshot.policy?.checkpoint,
    lastDecision: game.strategy?.last_decision ?? snapshot.strategy?.last_decision ?? null,
    gameOver: Boolean(game.game_over ?? snapshot.game_over ?? game.status === "finished"),
    winner: normalizeColor(game.result?.winner ?? snapshot.result?.winner ?? game.winner ?? snapshot.winner),
    endReason: game.end_reason ?? snapshot.end_reason ?? game.result?.reason ?? snapshot.result?.reason ?? null,
    result: game.result ?? snapshot.result ?? null,
    score: game.score ?? snapshot.score ?? null,
    status: game.status ?? snapshot.status,
  };
}

function boardPoint(row, col, y = BOARD_Y + 0.12) {
  return new THREE.Vector3(
    -GRID_SPAN / 2 + col * GRID_STEP,
    y,
    -GRID_SPAN / 2 + row * GRID_STEP,
  );
}

function outsideRootForTarget(target) {
  const xStrength = Math.abs(target.x);
  const zStrength = Math.abs(target.z);
  if (xStrength < 0.01 && zStrength < 0.01) {
    return new THREE.Vector3(FLY_OUTSIDE_EDGE, 1.48, 0);
  }
  if (xStrength >= zStrength) {
    return new THREE.Vector3(
      target.x < 0 ? -FLY_OUTSIDE_EDGE : FLY_OUTSIDE_EDGE,
      1.48,
      clamp(target.z, -GRID_SPAN / 2, GRID_SPAN / 2),
    );
  }
  return new THREE.Vector3(
    clamp(target.x, -GRID_SPAN / 2, GRID_SPAN / 2),
    1.48,
    target.z < 0 ? -FLY_OUTSIDE_EDGE : FLY_OUTSIDE_EDGE,
  );
}

function perimeterWaypoints(targetRoot) {
  const edge = FLY_OUTSIDE_EDGE;
  const start = FLY_BOWL.clone();
  if (Math.abs(targetRoot.x - edge) < 0.01) return [start, targetRoot.clone()];
  if (Math.abs(targetRoot.z + edge) < 0.01) {
    return [start, new THREE.Vector3(edge, 1.43, -edge), targetRoot.clone()];
  }
  if (Math.abs(targetRoot.z - edge) < 0.01) {
    return [start, new THREE.Vector3(edge, 1.43, edge), targetRoot.clone()];
  }
  const useTop = targetRoot.z <= 0;
  const zEdge = useTop ? -edge : edge;
  return [
    start,
    new THREE.Vector3(edge, 1.43, zEdge),
    new THREE.Vector3(-edge, 1.43, zEdge),
    targetRoot.clone(),
  ];
}

function pointAlongPath(points, progress) {
  const lengths = [];
  let total = 0;
  for (let index = 1; index < points.length; index += 1) {
    const length = points[index - 1].distanceTo(points[index]);
    lengths.push(length);
    total += length;
  }
  let remaining = clamp(progress, 0, 1) * total;
  for (let index = 0; index < lengths.length; index += 1) {
    if (remaining <= lengths[index] || index === lengths.length - 1) {
      return new THREE.Vector3().lerpVectors(points[index], points[index + 1], lengths[index] ? remaining / lengths[index] : 1);
    }
    remaining -= lengths[index];
  }
  return points.at(-1).clone();
}

function colorName(color) {
  if (color === "black") return "黑方";
  if (color === "white") return "白方";
  return displayValue(color);
}

function coordinateName(move) {
  if (!move) return "—";
  if (move.pass) return "停一手";
  if (!Number.isInteger(move.row) || !Number.isInteger(move.col)) return "不可获取";
  const letters = "ABCDEFGHJ";
  return `${letters[move.col] ?? "?"}${BOARD_SIZE - move.row}`;
}

function setServiceStatus(kind, message) {
  els.serviceDot.classList.toggle("is-loading", kind === "loading");
  els.serviceDot.classList.toggle("is-error", kind === "error");
  els.serviceStatus.textContent = message;
}

function setInteraction(message, error = false) {
  els.interactionStatus.textContent = message;
  els.interactionStatus.style.color = error ? "#eaa08e" : "";
}

function showSceneError(message) {
  els.sceneError.textContent = message;
  els.sceneError.hidden = !message;
}

function setFlyStage(label, active = Boolean(currentAnimation)) {
  els.flyStage.textContent = label;
  els.stageDot.classList.toggle("is-active", active);
}

function stateKey(state) {
  return `${displayValue(state?.decisionId, "no-decision")}|${displayValue(state?.boardHash, "no-board")}`;
}

function setTextWithTitle(element, value, fallback = "不可获取", shortLength = 22) {
  const full = displayValue(value, fallback);
  element.textContent = full.length > shortLength ? `${full.slice(0, shortLength)}…` : full;
  element.title = full;
}

function setNeuralChip(kind, text) {
  els.neuralStatusChip.className = `neural-status-chip is-${kind}`;
  els.neuralStatusChip.textContent = text;
}

function clearNeuralFrame(kind, title, detail) {
  neuralFrame = null;
  neuralAcceptedStateKey = null;
  neuralPlaybackStartedAt = 0;
  neuralLastDrawnTick = -1;
  els.neuralViewport.classList.add("is-empty");
  els.neuralViewport.classList.toggle("is-loading", kind === "loading");
  els.neuralViewport.classList.toggle("is-error", kind === "error" || kind === "unavailable");
  els.neuralEmptyTitle.textContent = title;
  els.neuralEmptyDetail.textContent = detail;
  els.neuralTick.textContent = "—";
  els.neuralTime.textContent = "—";
  els.neuralPlaybackState.textContent = title;
  els.neuralSpikeState.textContent = detail;
  [els.encodingHash, els.frameId, els.displaySize, els.spikeSummary]
    .forEach((element) => setTextWithTitle(element, null));
  setTextWithTitle(els.policyActionLabel, null);
  els.rerunNeuralButton.disabled = true;
  clearNeuralCanvas();
}

function clearNeuralCanvas() {
  if (neuralBrainActivity) {
    neuralBrainActivity.array.fill(0);
    neuralBrainActivity.needsUpdate = true;
  }
  if (neuralBrainFiring) {
    neuralBrainFiring.array.fill(0);
    neuralBrainFiring.needsUpdate = true;
  }
  if (neuralBrainLines) {
    neuralBrainLines.geometry.dispose();
    neuralBrainLines.material.dispose();
    neuralBrainLines.removeFromParent();
    neuralBrainLines = null;
  }
  renderNeuralBrain(performance.now());
}

function resizeNeuralCanvas() {
  const rect = els.neuralViewport.getBoundingClientRect();
  if (!neuralBrainRenderer || !neuralBrainCamera || !rect.width || !rect.height) return;
  neuralBrainRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  neuralBrainRenderer.setSize(rect.width, rect.height, false);
  neuralBrainCamera.aspect = rect.width / Math.max(1, rect.height);
  neuralBrainCamera.updateProjectionMatrix();
  renderNeuralBrain(performance.now());
}

function createNeuralBrainScene() {
  neuralBrainRenderer = new THREE.WebGLRenderer({
    canvas: els.neuralCanvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
  neuralBrainRenderer.setClearColor(0x09100e, 0);
  neuralBrainScene = new THREE.Scene();
  neuralBrainCamera = new THREE.PerspectiveCamera(36, 1, 0.01, 20);
  neuralBrainCamera.position.set(0, 0.05, 3.2);
  neuralBrainCamera.lookAt(0, 0, 0);
  neuralBrainNetwork = new THREE.Group();
  neuralBrainScene.add(neuralBrainNetwork);

  els.neuralCanvas.addEventListener("pointerdown", (event) => {
    neuralBrainDrag = { x: event.clientX, y: event.clientY, yaw: neuralBrainYaw, pitch: neuralBrainPitch };
    els.neuralCanvas.setPointerCapture(event.pointerId);
  });
  els.neuralCanvas.addEventListener("pointermove", (event) => {
    if (!neuralBrainDrag) return;
    neuralBrainYaw = neuralBrainDrag.yaw + (event.clientX - neuralBrainDrag.x) * 0.008;
    neuralBrainPitch = clamp(neuralBrainDrag.pitch + (event.clientY - neuralBrainDrag.y) * 0.006, -0.72, 0.72);
  });
  const finishDrag = (event) => {
    if (neuralBrainDrag && els.neuralCanvas.hasPointerCapture(event.pointerId)) {
      els.neuralCanvas.releasePointerCapture(event.pointerId);
    }
    neuralBrainDrag = null;
  };
  els.neuralCanvas.addEventListener("pointerup", finishDrag);
  els.neuralCanvas.addEventListener("pointercancel", finishDrag);
}

function parseNpyFloat32(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 12 || String.fromCharCode(...bytes.slice(1, 6)) !== "NUMPY" || bytes[0] !== 0x93) {
    throw new Error("soma 坐标不是有效的 NPY 文件");
  }
  const view = new DataView(buffer);
  const major = bytes[6];
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerOffset = major === 1 ? 10 : 12;
  const dataOffset = headerOffset + headerLength;
  const header = new TextDecoder("latin1").decode(bytes.slice(headerOffset, dataOffset));
  if (!header.includes("'<f4'") && !header.includes('"<f4"')) throw new Error("soma 坐标必须是 little-endian float32");
  if (!/fortran_order['"]?\s*:\s*False/.test(header)) throw new Error("不支持 Fortran 顺序的 soma 坐标");
  const shapeMatch = header.match(/shape['"]?\s*:\s*\(\s*(\d+)\s*,\s*3\s*\)/);
  if (!shapeMatch) throw new Error("soma 坐标必须是 N×3");
  const nodeCount = Number(shapeMatch[1]);
  if (dataOffset % 4 !== 0 || buffer.byteLength !== dataOffset + nodeCount * 3 * 4) throw new Error("soma 坐标长度不匹配");
  return { nodeCount, values: new Float32Array(buffer, dataOffset, nodeCount * 3).slice() };
}

function normalizeSomaPositions(raw) {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < raw.length; index += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      minimum[axis] = Math.min(minimum[axis], raw[index + axis]);
      maximum[axis] = Math.max(maximum[axis], raw[index + axis]);
    }
  }
  const center = minimum.map((value, axis) => (value + maximum[axis]) / 2);
  const longestSpan = Math.max(...minimum.map((value, axis) => maximum[axis] - value), 1);
  const scale = 2.15 / longestSpan;
  const normalized = new Float32Array(raw.length);
  for (let index = 0; index < raw.length; index += 3) {
    normalized[index] = (raw[index] - center[0]) * scale;
    normalized[index + 1] = -(raw[index + 2] - center[2]) * scale;
    normalized[index + 2] = (raw[index + 1] - center[1]) * scale;
  }
  return normalized;
}

function buildNeuralBrainPoints(positions) {
  neuralBrainPoints?.geometry.dispose();
  neuralBrainPoints?.material.dispose();
  neuralBrainPoints?.removeFromParent();
  const nodeCount = positions.length / 3;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  neuralBrainActivity = new THREE.BufferAttribute(new Float32Array(nodeCount), 1).setUsage(THREE.DynamicDrawUsage);
  neuralBrainFiring = new THREE.BufferAttribute(new Float32Array(nodeCount), 1).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("activity", neuralBrainActivity);
  geometry.setAttribute("firing", neuralBrainFiring);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      pixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
      activeColor: { value: new THREE.Color("#69e4c1") },
    },
    vertexShader: `
      attribute float activity;
      attribute float firing;
      uniform float pixelRatio;
      varying float vActivity;
      varying float vFiring;
      void main() {
        vActivity = activity;
        vFiring = firing;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = pixelRatio * (1.0 + activity * 2.5 + firing * 4.0) * clamp(1.7 / -mv.z, 0.65, 2.2);
      }
    `,
    fragmentShader: `
      uniform vec3 activeColor;
      varying float vActivity;
      varying float vFiring;
      void main() {
        float radius = length(gl_PointCoord - vec2(0.5));
        if (radius > 0.5) discard;
        vec3 base = vec3(0.28, 0.42, 0.39);
        vec3 color = mix(base, activeColor, min(1.0, vActivity));
        color = mix(color, vec3(1.0, 0.84, 0.48), min(1.0, vFiring));
        float alpha = (0.10 + vActivity * 0.64 + vFiring * 0.80) * (1.0 - smoothstep(0.18, 0.5, radius));
        gl_FragColor = vec4(color, min(1.0, alpha));
        #include <colorspace_fragment>
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    blending: THREE.NormalBlending,
    toneMapped: false,
  });
  neuralBrainPoints = new THREE.Points(geometry, material);
  neuralBrainNetwork.add(neuralBrainPoints);
}

async function loadNeuralAnatomy(status) {
  if (neuralBrainReady && neuralBrainModelId === status.model_id) return;
  els.brainAnatomyStatus.textContent = "加载全部 soma…";
  const response = await fetch(API_NEURAL_ANATOMY, { headers: { Accept: "application/octet-stream" }, cache: "no-store" });
  if (!response.ok) throw new Error(`soma 坐标加载失败（HTTP ${response.status}）`);
  const responseModelId = response.headers.get("X-MaleCNS-Model-ID");
  if (responseModelId !== status.model_id) throw new Error("soma 坐标与运行模型不匹配");
  const parsed = parseNpyFloat32(await response.arrayBuffer());
  if (parsed.nodeCount !== Number(status.controller_node_count)) throw new Error("soma 节点数与运行图不匹配");
  neuralBrainPositions = normalizeSomaPositions(parsed.values);
  neuralBrainModelId = status.model_id;
  buildNeuralBrainPoints(neuralBrainPositions);
  neuralBrainReady = true;
  els.brainAnatomyStatus.textContent = `${parsed.nodeCount.toLocaleString()} soma`;
  els.brainAnatomyStatus.title = `${parsed.nodeCount.toLocaleString()} 个经模型清单校验的 MaleCNS soma 坐标`;
  if (neuralFrame) applyFrameToNeuralBrain(neuralFrame);
  resizeNeuralCanvas();
}

function renderNeuralBrain(timestamp) {
  if (!neuralBrainRenderer || !neuralBrainScene || !neuralBrainCamera || !neuralBrainNetwork) return;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const delta = neuralBrainLastRender ? Math.min(50, Math.max(0, timestamp - neuralBrainLastRender)) : 0;
  neuralBrainLastRender = timestamp;
  if (!neuralBrainDrag && !reduceMotion) neuralBrainYaw += 0.000045 * delta;
  neuralBrainNetwork.rotation.set(neuralBrainPitch, neuralBrainYaw, 0);
  neuralBrainRenderer.render(neuralBrainScene, neuralBrainCamera);
}

function unwrapNeuralStatus(payload) {
  if (!payload || typeof payload !== "object") throw new Error("神经状态响应无效");
  if (payload.ok === false) throw new Error(payload.error?.message ?? payload.error ?? payload.message ?? "神经运行时不可用");
  return payload.status && typeof payload.status === "object" ? payload.status : payload;
}

function statusExplicitlyUnavailable(status) {
  const flags = [status.available, status.ready, status.loaded, status.model_loaded, status.graph_loaded];
  if (flags.some((flag) => flag === false)) return true;
  const label = String(status.state ?? status.runtime_state ?? status.status ?? "").toLowerCase();
  return ["unavailable", "error", "failed", "missing", "not_loaded"].includes(label);
}

async function fetchNeuralStatus() {
  setNeuralChip("loading", "正在检查");
  try {
    const response = await fetch(API_NEURAL_STATUS, { headers: { Accept: "application/json" }, cache: "no-store" });
    const status = unwrapNeuralStatus(await readJson(response));
    if (statusExplicitlyUnavailable(status)) {
      neuralRuntimeAvailable = false;
      const detail = status.load_error ?? status.error?.message ?? status.error ?? status.message ?? "MaleCNS 图资产或运行时尚未就绪";
      setNeuralChip("unavailable", "不可用");
      clearNeuralFrame("unavailable", "神经运行时不可用", String(detail));
      setTextWithTitle(els.modelId, status.model_id);
      els.graphSize.textContent = `${displayValue(status.controller_node_count, "—")} 节点 / ${displayValue(status.controller_edge_count, "—")} 边`;
      return false;
    }
    neuralRuntimeAvailable = true;
    setNeuralChip("ready", "运行时就绪");
    setTextWithTitle(els.modelId, status.model_id);
    els.graphSize.textContent = `${displayValue(status.controller_node_count, "—")} 节点 / ${displayValue(status.controller_edge_count, "—")} 边`;
    try {
      await loadNeuralAnatomy(status);
    } catch (error) {
      neuralBrainReady = false;
      els.brainAnatomyStatus.textContent = "soma 不可用";
      els.brainAnatomyStatus.title = error instanceof Error ? error.message : String(error);
      clearNeuralFrame("error", "果蝇脑解剖坐标不可用", els.brainAnatomyStatus.title);
    }
    els.rerunNeuralButton.disabled = !currentState || (neuralPolicyAvailable && !neuralFrame);
    return true;
  } catch (error) {
    neuralRuntimeAvailable = false;
    const message = error instanceof Error ? error.message : String(error);
    setNeuralChip("unavailable", "不可用");
    clearNeuralFrame("unavailable", "无法连接神经运行时", message);
    els.rerunNeuralButton.disabled = true;
    return false;
  }
}

async function fetchNeuralPolicyStatus() {
  try {
    const response = await fetch(API_NEURAL_POLICY_STATUS, { headers: { Accept: "application/json" }, cache: "no-store" });
    const status = unwrapNeuralStatus(await readJson(response));
    neuralPolicyStatus = status;
    neuralPolicyAvailable = !statusExplicitlyUnavailable(status)
      && status.stage4_controls_go_moves === true
      && status.training_status === "trained"
      && status.teacher_accessed_at_runtime === false
      && status.baseline_controller_called === false
      && status.fallback_used === false;
    const hybridRulePass = status.training_info?.pass_control === "rule-after-opponent-pass";
    const candidatePreview = /^stage8_.*pilot/.test(status.checkpoint_directory_name ?? "");
    const policyLabel = !neuralPolicyAvailable
      ? "不可用 · 不会回退"
      : hybridRulePass
        ? `${candidatePreview ? "开发候选 · " : ""}MaleCNS 落子 / 规则 PASS`
        : "已训练 · 控制白棋";
    setTextWithTitle(els.policyStatusLabel, policyLabel);
    if (status.checkpoint_hash) setTextWithTitle(els.checkpointLabel, status.checkpoint_hash);
    if (currentState) applyState(currentState);
    updateControls();
    return neuralPolicyAvailable;
  } catch (error) {
    neuralPolicyStatus = null;
    neuralPolicyAvailable = false;
    setTextWithTitle(els.policyStatusLabel, "无法读取 · 不会回退");
    updateControls();
    return false;
  }
}

async function fetchEvaluationStatus() {
  try {
    const response = await fetch(API_EVALUATION_STATUS, { headers: { Accept: "application/json" }, cache: "no-store" });
    const payload = await readJson(response);
    if (!payload.available) {
      setTextWithTitle(els.evaluationStatusLabel, "尚未运行");
      setTextWithTitle(els.evaluationMetricLabel, null);
      return;
    }
    const evaluation = payload.evaluation;
    const metrics = evaluation?.metrics;
    if (!evaluation || !metrics) throw new Error("评测报告缺少指标");
    const current = payload.matches_loaded_checkpoint === true;
    setTextWithTitle(
      els.evaluationStatusLabel,
      current ? "开发指标 · 当前检查点" : "过期 · 非当前检查点",
      `${evaluation.report_sha256}；历史报告存在少量 state 重叠，只作为开发模仿指标，不作为最终独立验收`,
    );
    const agreement = Number(metrics.teacher_agreement_rate);
    const randomRate = Number(metrics.random_legal_expected_rate);
    const count = Number(evaluation.evaluation?.sample_count);
    els.evaluationMetricLabel.textContent = Number.isFinite(agreement) && Number.isFinite(randomRate)
      ? `${(agreement * 100).toFixed(2)}% / ${(randomRate * 100).toFixed(2)}% · n=${displayValue(count, "—")}`
      : "报告指标不可获取";
    els.evaluationMetricLabel.title = `范围：${evaluation.scope || "不可获取"}；p=${displayValue(metrics.poisson_binomial_p_value)}`;
  } catch (error) {
    setTextWithTitle(els.evaluationStatusLabel, "不可获取", error instanceof Error ? error.message : String(error));
    setTextWithTitle(els.evaluationMetricLabel, null);
  }
}

async function fetchTournamentStatus() {
  try {
    const response = await fetch(API_TOURNAMENT_STATUS, { headers: { Accept: "application/json" }, cache: "no-store" });
    const payload = await readJson(response);
    if (!payload.available) {
      setTextWithTitle(els.generalTournamentLabel, "尚未运行");
      return;
    }
    const report = payload.tournament;
    const metrics = report?.metrics;
    if (!metrics) throw new Error("对局报告缺少角色指标");
    setTextWithTitle(
      els.generalTournamentLabel,
      `${metrics.wins}/${report.configuration.game_count} · ${report.status === "passed" ? "通过" : "未通过"}`,
      "黑白通用门槛与网页白方角色门槛分开报告",
    );
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    setTextWithTitle(els.generalTournamentLabel, "不可获取", detail);
  }
}

async function fetchStage7Status() {
  try {
    const response = await fetch(API_STAGE7_STATUS, { headers: { Accept: "application/json" }, cache: "no-store" });
    const payload = await readJson(response);
    if (!payload.available) {
      setTextWithTitle(els.whiteTournamentLabel, "尚未运行");
      return;
    }
    const report = payload.development;
    const metrics = report?.metrics;
    if (!metrics) throw new Error("Stage 7 开发报告缺少指标");
    const current = payload.matches_loaded_checkpoint === true;
    const wins = Number(metrics.wins);
    const games = wins + Number(metrics.losses) + Number(metrics.draws);
    const interval = Array.isArray(metrics.wilson_95_interval)
      ? metrics.wilson_95_interval.map((value) => `${(Number(value) * 100).toFixed(1)}%`).join("–")
      : "不可获取";
    const label = `${wins}/${games} · 开发复测未通过`;
    setTextWithTitle(
      els.whiteTournamentLabel,
      current ? label : `${label} · 非当前检查点`,
      `胜率 ${(Number(metrics.win_rate) * 100).toFixed(1)}%；95% Wilson ${interval}；自然终局 ${metrics.natural_finish_count}/${games}；只属于开发证据，不是最终验收`,
    );
  } catch (error) {
    setTextWithTitle(els.whiteTournamentLabel, "不可获取", error instanceof Error ? error.message : String(error));
  }
}

async function fetchStage8Status() {
  try {
    const response = await fetch(API_STAGE8_STATUS, { headers: { Accept: "application/json" }, cache: "no-store" });
    const payload = await readJson(response);
    if (!payload.available || payload.matches_loaded_checkpoint !== true) {
      setTextWithTitle(els.signalZeroControlLabel, "不可获取", "Stage 8 消融报告不是当前运行检查点的结果");
      setTextWithTitle(els.signalShuffleControlLabel, "不可获取", "Stage 8 消融报告不是当前运行检查点的结果");
      return;
    }
    const unseen = payload.unseen;
    const both = payload.both?.metrics;
    const white = payload.white?.metrics;
    if (!unseen || !both || !white) throw new Error("Stage 8 报告指标缺失");
    setTextWithTitle(
      els.evaluationStatusLabel,
      "Stage 8 候选 · 独立局面复测",
      `已排除 ${unseen.excluded_training_state_count} 个与训练完全相同的局面；开发证据，非最终验收`,
    );
    setTextWithTitle(
      els.evaluationMetricLabel,
      `${(Number(unseen.teacher_agreement_rate) * 100).toFixed(2)}% / ${(Number(unseen.random_legal_expected_rate) * 100).toFixed(2)}% · n=${unseen.sample_count}`,
      `教师一致 ${unseen.teacher_matches}/${unseen.sample_count}；报告 ${unseen.report_sha256}`,
    );
    const ablation = payload.signal_ablation;
    if (ablation?.available === true) {
      setTextWithTitle(
        els.signalZeroControlLabel,
        `${(Number(ablation.intact_agreement_rate) * 100).toFixed(1)}% → ${(Number(ablation.zero_agreement_rate) * 100).toFixed(1)}% · n=${ablation.sample_count}`,
        `同一读出器：左为真实状态匹配 MaleCNS 输出，右为 128 维信号置零。两者都保留合法落点掩码；规则 PASS 不计入。不能证明生物线路优于人工网络。报告 ${ablation.report_sha256}`,
      );
      setTextWithTitle(
        els.signalShuffleControlLabel,
        `${(Number(ablation.shuffled_mean_agreement_rate) * 100).toFixed(1)}% · 打乱配对均值`,
        `128 次随机打乱验证状态与神经输出的配对；完整输出 ${(Number(ablation.intact_agreement_rate) * 100).toFixed(1)}%；经验尾概率 ${Number(ablation.empirical_tail_probability_shuffled_ge_intact).toFixed(4)}。${ablation.limitation}`,
      );
    } else {
      setTextWithTitle(els.signalZeroControlLabel, "未运行", "没有经过哈希校验的置零对照报告");
      setTextWithTitle(els.signalShuffleControlLabel, "未运行", "没有经过哈希校验的打乱对照报告");
    }
    setTextWithTitle(
      els.generalTournamentLabel,
      `${both.wins}/6 · 黑白候选初筛`,
      `自然终局 ${both.natural_finish_count}/6；神经读出 ${both.neural_readout_decision_count} 手；规则 PASS ${both.go_rule_pass_decision_count} 手；非最终棋力结论`,
    );
    const zeroGames = payload.zero_signal_games;
    setTextWithTitle(
      els.whiteTournamentLabel,
      zeroGames?.available === true
        ? `${white.wins}/8 · 置零对照 ${zeroGames.zero_white_wins}/8`
        : `${white.wins}/8 · 白方候选初筛`,
      zeroGames?.available === true
        ? `同种子同开局：正常神经版 ${white.wins}/8 胜、自然终局 ${white.natural_finish_count}/8；神经特征置零版 ${zeroGames.zero_white_wins}/8 胜、自然终局 ${zeroGames.zero_natural_finishes}/8；随机白方 ${white.random_white_baseline.wins}/8。8 局太少，不是最终棋力证据。报告 ${zeroGames.report_sha256}`
        : `相同种子和开局的随机白方 ${white.random_white_baseline.wins}/8；自然终局 ${white.natural_finish_count}/8；非最终验收`,
    );
  } catch (error) {
    setTextWithTitle(els.evaluationStatusLabel, "Stage 8 报告不可获取", error instanceof Error ? error.message : String(error));
    setTextWithTitle(els.signalZeroControlLabel, "不可获取", error instanceof Error ? error.message : String(error));
    setTextWithTitle(els.signalShuffleControlLabel, "不可获取", error instanceof Error ? error.message : String(error));
  }
}

async function readJson(response) {
  const text = await response.text();
  let payload;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`服务返回非 JSON 内容（HTTP ${response.status}）`);
  }
  if (!response.ok) {
    const detail = payload?.error?.message ?? payload?.message ?? `HTTP ${response.status}`;
    const error = new Error(detail);
    error.code = payload?.error?.code;
    error.status = response.status;
    throw error;
  }
  return payload;
}

function extractNeuralFrame(payload) {
  if (!payload || typeof payload !== "object") throw new Error("神经帧响应无效");
  if (payload.ok === false) throw new Error(payload.error?.message ?? payload.error ?? payload.message ?? "神经帧计算失败");
  const frame = payload.frame ?? payload.neural_frame ?? payload.result ?? payload;
  if (!frame || typeof frame !== "object") throw new Error("响应中缺少神经帧");
  return {
    frame,
    decisionId: payload.decision_id ?? frame.decision_id,
    boardHash: payload.board_hash ?? frame.board_hash,
    encoding: payload.encoding,
  };
}

function encodingMatchesState(encoding, state) {
  const vector = encoding?.vector;
  if (!Array.isArray(vector) || vector.length !== 415) return false;
  const ownColor = state.toMove;
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      const index = row * BOARD_SIZE + col;
      const stone = state.board[row][col];
      if (Number(vector[index]) !== (stone === ownColor ? 1 : 0)) return false;
      if (Number(vector[81 + index]) !== (stone && stone !== ownColor ? 1 : 0)) return false;
      if (Number(vector[162 + index]) !== (stone ? 0 : 1)) return false;
    }
  }
  const expectedLastMove = state.lastMove
    ? (state.lastMove.pass ? 81 : state.lastMove.row * BOARD_SIZE + state.lastMove.col)
    : null;
  if (encoding.last_move_action !== expectedLastMove) return false;
  if (Number(vector[405]) !== (state.toMove === "black" ? 1 : 0)) return false;
  if (Number(vector[406]) !== (state.toMove === "white" ? 1 : 0)) return false;
  if (Number(vector[412]) !== (state.gameOver ? 1 : 0)) return false;
  return Number(vector[413]) === Number(state.moveNumber);
}

function edgeEndpoint(edge, side) {
  const isSource = side === "source";
  const candidates = isSource
    ? ["source_model_index", "pre_model_index", "source_index", "pre_index", "source_body_id", "pre_body_id", "source", "pre", "from"]
    : ["target_model_index", "post_model_index", "target_index", "post_index", "target_body_id", "post_body_id", "target", "post", "to"];
  if (Array.isArray(edge)) return edge[isSource ? 0 : 1];
  for (const key of candidates) {
    if (edge?.[key] !== undefined && edge[key] !== null) return edge[key];
  }
  return null;
}

function resolveNodeReference(reference, byModelIndex, byBodyId) {
  if (reference && typeof reference === "object") {
    const modelIndex = reference.model_index ?? reference.index;
    const bodyId = reference.body_id ?? reference.id;
    return byModelIndex.get(String(modelIndex)) ?? byBodyId.get(String(bodyId)) ?? null;
  }
  return byModelIndex.get(String(reference)) ?? byBodyId.get(String(reference)) ?? null;
}

function prepareNeuralFrame(frame) {
  for (const key of ["frame_id", "model_id", "board_hash", "encoding_hash"]) {
    if (frame[key] === undefined || frame[key] === null || frame[key] === "") throw new Error(`神经帧缺少 ${key}`);
  }
  if (!Array.isArray(frame.display_nodes)) throw new Error("神经帧缺少 display_nodes");
  if (!Array.isArray(frame.display_edges)) throw new Error("神经帧缺少 display_edges");
  const nodes = frame.display_nodes.map((node) => {
    if (!Array.isArray(node.soma_xyz) || node.soma_xyz.length < 3 || !node.soma_xyz.slice(0, 3).every(Number.isFinite)) {
      throw new Error("display_nodes 含无效 soma_xyz");
    }
    return {
      ...node,
      model_index: Number(node.model_index),
      body_id: node.body_id,
      soma_xyz: node.soma_xyz.slice(0, 3).map(Number),
      max_activity: Math.max(0, Number(node.max_activity) || 0),
      spike_count: Math.max(0, Number(node.spike_count) || 0),
    };
  });
  if (nodes.length === 0) throw new Error("神经帧的显示节点为空");
  const byModelIndex = new Map(nodes.map((node) => [String(node.model_index), node]));
  const byBodyId = new Map(nodes.map((node) => [String(node.body_id), node]));
  const edges = frame.display_edges.map((edge) => ({
    source: resolveNodeReference(edgeEndpoint(edge, "source"), byModelIndex, byBodyId),
    target: resolveNodeReference(edgeEndpoint(edge, "target"), byModelIndex, byBodyId),
  })).filter((edge) => edge.source && edge.target);
  const recordedSpikes = Array.isArray(frame.recorded_spikes) ? frame.recorded_spikes : [];
  const spikeTicksByBody = new Map();
  const spikeTicksByModel = new Map();
  const spikeModelIndicesByTick = new Map();
  for (const event of recordedSpikes) {
    const tick = Number(event.tick);
    if (!Number.isFinite(tick)) continue;
    const bodyKey = String(event.body_id);
    const modelKey = String(event.model_index);
    if (!spikeTicksByBody.has(bodyKey)) spikeTicksByBody.set(bodyKey, []);
    if (!spikeTicksByModel.has(modelKey)) spikeTicksByModel.set(modelKey, []);
    spikeTicksByBody.get(bodyKey).push(tick);
    spikeTicksByModel.get(modelKey).push(tick);
    if (!spikeModelIndicesByTick.has(tick)) spikeModelIndicesByTick.set(tick, []);
    spikeModelIndicesByTick.get(tick).push(Number(event.model_index));
  }
  const tickCount = Math.max(1, Number(frame.parameters?.tick_count) || frame.tick_spike_counts?.length || 1);
  return {
    raw: frame,
    nodes,
    edges,
    recordedSpikes,
    spikeTicksByBody,
    spikeTicksByModel,
    spikeModelIndicesByTick,
    tickCount,
    dtMs: Number(frame.parameters?.dt_ms) || 0,
    maxActivity: Math.max(0, ...nodes.map((node) => node.max_activity)),
    maxSpikeCount: Math.max(0, ...nodes.map((node) => node.spike_count)),
  };
}

function rebuildNeuralBrainEdges(prepared) {
  if (!neuralBrainReady || !neuralBrainPositions) return;
  if (neuralBrainLines) {
    neuralBrainLines.geometry.dispose();
    neuralBrainLines.material.dispose();
    neuralBrainLines.removeFromParent();
  }
  const positions = new Float32Array(prepared.edges.length * 6);
  let offset = 0;
  for (const edge of prepared.edges) {
    const source = Number(edge.source.model_index);
    const target = Number(edge.target.model_index);
    if (!Number.isInteger(source) || !Number.isInteger(target)) continue;
    positions.set(neuralBrainPositions.subarray(source * 3, source * 3 + 3), offset);
    positions.set(neuralBrainPositions.subarray(target * 3, target * 3 + 3), offset + 3);
    offset += 6;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions.subarray(0, offset), 3));
  const material = new THREE.LineBasicMaterial({
    color: 0x70bfa9,
    transparent: true,
    opacity: 0.085,
    depthWrite: false,
    depthTest: false,
    blending: THREE.NormalBlending,
  });
  neuralBrainLines = new THREE.LineSegments(geometry, material);
  neuralBrainLines.renderOrder = -1;
  neuralBrainNetwork.add(neuralBrainLines);
}

function applyFrameToNeuralBrain(prepared) {
  if (!neuralBrainReady || !neuralBrainActivity || !neuralBrainFiring) return;
  const activity = neuralBrainActivity.array;
  const firing = neuralBrainFiring.array;
  activity.fill(0);
  firing.fill(0);
  for (const node of prepared.nodes) {
    const index = Number(node.model_index);
    if (!Number.isInteger(index) || index < 0 || index >= activity.length) continue;
    const activityLevel = prepared.maxActivity > 0 ? node.max_activity / prepared.maxActivity : 0;
    const spikeLevel = prepared.maxSpikeCount > 0 ? node.spike_count / prepared.maxSpikeCount : 0;
    activity[index] = Math.max(activityLevel, Math.sqrt(spikeLevel));
  }
  neuralBrainActivity.needsUpdate = true;
  neuralBrainFiring.needsUpdate = true;
  rebuildNeuralBrainEdges(prepared);
}

function setCurrentNeuralFrame(prepared, requestContext, responseDecisionId) {
  neuralFrame = prepared;
  neuralAcceptedStateKey = requestContext.key;
  neuralPlaybackStartedAt = performance.now();
  neuralLastDrawnTick = -1;
  els.neuralViewport.classList.remove("is-empty", "is-loading", "is-error");
  setNeuralChip("ready", "帧已同步");
  setTextWithTitle(els.neuralDecisionId, responseDecisionId);
  setTextWithTitle(els.neuralBoardHash, prepared.raw.board_hash);
  setTextWithTitle(els.encodingHash, prepared.raw.encoding_hash);
  setTextWithTitle(els.modelId, prepared.raw.model_id);
  setTextWithTitle(els.frameId, prepared.raw.frame_id);
  els.graphSize.textContent = `${displayValue(prepared.raw.controller_node_count, "—")} 节点 / ${displayValue(prepared.raw.controller_edge_count, "—")} 边`;
  els.displaySize.textContent = `${prepared.nodes.length} 节点 / ${prepared.edges.length} 边`;
  const totalSpikes = Number(prepared.raw.total_spike_count) || 0;
  const synapticEvents = Number(prepared.raw.synaptic_event_count) || 0;
  els.spikeSummary.textContent = `${totalSpikes} spike / ${synapticEvents} 突触事件`;
  els.spikeSummary.title = `${prepared.recordedSpikes.length} 条已记录 spike${prepared.raw.recorded_spikes_truncated ? "（记录已截断）" : ""}`;
  if (totalSpikes === 0 || prepared.raw.silenced === true) {
    els.neuralPlaybackState.textContent = "真实帧：静默";
    els.neuralSpikeState.textContent = "0 spike；不添加装饰性放电";
  } else {
    els.neuralPlaybackState.textContent = "按真实 tick 回放";
    els.neuralSpikeState.textContent = `${prepared.recordedSpikes.length} 条已记录 spike${prepared.raw.recorded_spikes_truncated ? "（截断）" : ""}`;
  }
  els.rerunNeuralButton.disabled = false;
  applyFrameToNeuralBrain(prepared);
  resizeNeuralCanvas();
  drawNeuralFrame(0);
}

function validateNeuralMovePayload(payload, preMoveState) {
  if (!payload || typeof payload !== "object" || payload.ok !== true) throw new Error("神经落子响应无效");
  if (payload.controller !== "malecns-neural-policy") throw new Error("响应控制器不是 MaleCNS 神经策略");
  if (payload.teacher_accessed_at_runtime !== false || payload.baseline_controller_called !== false || payload.fallback_used !== false) {
    throw new Error("神经落子违反禁止教师、baseline 与回退的约束");
  }
  const decision = payload.decision;
  const frame = payload.frame;
  if (!decision || !frame || decision.frame_id !== frame.frame_id) throw new Error("神经决策与显示帧不属于同一次计算");
  if (decision.board_hash !== frame.board_hash || decision.encoding_hash !== frame.encoding_hash) {
    throw new Error("神经决策与显示帧的来源不一致");
  }
  if (!encodingMatchesState(payload.pre_move_encoding, preMoveState)) {
    throw new Error("神经决策输入与落子前棋盘不一致");
  }
  if (decision.checkpoint_hash !== neuralPolicyStatus?.checkpoint_hash) {
    throw new Error("神经决策使用了与已验证状态不同的检查点");
  }
  if (decision.controller_source === "go-rule-pass-gate") {
    if (decision.pass_selected !== true || decision.neural_readout_used !== false) {
      throw new Error("规则 PASS 的来源声明与落子不一致");
    }
  } else if (decision.controller_source !== "malecns-linear-readout" || decision.neural_readout_used !== true) {
    throw new Error("落子缺少可验证的控制来源");
  }
  const state = normalizeState(extractSnapshot(payload));
  const selectedMove = decision.move === "pass"
    ? { pass: true }
    : { row: Number(decision.move?.[0]), col: Number(decision.move?.[1]) };
  if (coordinateName(selectedMove) !== coordinateName(state.lastMove)) {
    throw new Error("神经策略选择与实际提交的棋步不一致");
  }
  return { state, decision, frame };
}

async function postNeuralMove(preMoveState) {
  const response = await fetch(API_NEURAL_MOVE, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ seed: 0 }),
    cache: "no-store",
  });
  const payload = await readJson(response);
  const result = validateNeuralMovePayload(payload, preMoveState);
  const prepared = prepareNeuralFrame(result.frame);
  setCurrentNeuralFrame(prepared, {
    key: stateKey(preMoveState),
    decisionId: preMoveState.decisionId,
    boardHash: preMoveState.boardHash,
  }, result.decision.decision_hash);
  const rulePass = result.decision.controller_source === "go-rule-pass-gate";
  els.controllerLabel.textContent = rulePass ? "围棋规则 PASS" : "MaleCNS 神经读出";
  els.controllerLabel.title = rulePass
    ? "本次 PASS 由围棋规则门控；已显示的神经帧没有用于选择此步"
    : "本次棋步由当前显示神经帧的已训练读出选择";
  setTextWithTitle(els.policyActionLabel, `${result.decision.action_index} · ${coordinateName(result.state.lastMove)}`);
  preserveNeuralFrameForStateKey = stateKey(result.state);
  return result.state;
}

async function requestNeuralFrame({ force = false } = {}) {
  if (!currentState || !neuralRuntimeAvailable) return;
  const requestContext = {
    key: stateKey(currentState),
    decisionId: currentState.decisionId,
    boardHash: currentState.boardHash,
  };
  if (!force && (requestContext.key === neuralAcceptedStateKey || (neuralRequestBusy && requestContext.key === neuralTargetStateKey))) return;
  neuralTargetStateKey = requestContext.key;
  neuralRequestSerial += 1;
  window.clearTimeout(neuralRetryTimer);
  const serial = neuralRequestSerial;
  neuralFrameAbort?.abort();
  const controller = new AbortController();
  neuralFrameAbort = controller;
  neuralRequestBusy = true;
  els.rerunNeuralButton.disabled = true;
  setNeuralChip("loading", "正在计算");
  clearNeuralFrame("loading", "正在计算当前棋局的神经帧", "旧帧已撤下；完成前不会显示过期活动");
  setTextWithTitle(els.neuralDecisionId, null, "待生成");
  setTextWithTitle(els.neuralBoardHash, null, "待生成");
  try {
    const response = await fetch(API_NEURAL_FRAME, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({}),
      cache: "no-store",
      signal: controller.signal,
    });
    const result = extractNeuralFrame(await readJson(response));
    if (serial !== neuralRequestSerial || requestContext.key !== stateKey(currentState)) return;
    if (result.boardHash !== result.frame.board_hash || !encodingMatchesState(result.encoding, currentState)) {
      throw new Error("返回帧的棋盘编码与当前棋局不一致，已拒绝显示");
    }
    const prepared = prepareNeuralFrame(result.frame);
    setCurrentNeuralFrame(prepared, requestContext, result.decisionId);
  } catch (error) {
    if (error?.name === "AbortError" || serial !== neuralRequestSerial) return;
    const message = error instanceof Error ? error.message : String(error);
    const busy = error?.status === 409 || error?.code === "neural_busy";
    const unavailable = error?.status === 503 || error?.code === "neural_unavailable";
    if (unavailable) neuralRuntimeAvailable = false;
    setNeuralChip(unavailable ? "unavailable" : "error", unavailable ? "不可用" : (busy ? "运行时忙" : "帧错误"));
    clearNeuralFrame(
      unavailable ? "unavailable" : "error",
      unavailable ? "神经运行时不可用" : (busy ? "神经运行时正忙" : "神经帧未生成"),
      message,
    );
    if (busy) {
      neuralRetryTimer = window.setTimeout(() => {
        if (currentState && stateKey(currentState) === requestContext.key && neuralRuntimeAvailable && !neuralRequestBusy) {
          requestNeuralFrame({ force: true });
        }
      }, 900);
    }
  } finally {
    if (serial === neuralRequestSerial) {
      neuralRequestBusy = false;
      neuralFrameAbort = null;
      els.rerunNeuralButton.disabled = !neuralRuntimeAvailable || !currentState;
    }
  }
}

async function fetchState({ quiet = false } = {}) {
  if (!quiet) setServiceStatus("loading", "正在读取本地棋局…");
  const response = await fetch(API_STATE, { headers: { Accept: "application/json" }, cache: "no-store" });
  const snapshot = extractSnapshot(await readJson(response));
  const state = normalizeState(snapshot);
  if (!currentAnimation) applyState(state);
  setServiceStatus("ready", "本地围棋服务已连接");
  showSceneError("");
  return state;
}

async function postAction(body) {
  const response = await fetch(API_ACTION, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  return normalizeState(extractSnapshot(await readJson(response)));
}

function applyState(state) {
  currentState = state;
  renderStones(state.board, state.lastMove);
  els.turnLabel.textContent = state.gameOver ? "对局结束" : colorName(state.toMove);
  els.moveNumber.textContent = displayValue(state.moveNumber, "—");
  els.lastMove.textContent = coordinateName(state.lastMove);
  els.boardHash.textContent = shortValue(state.boardHash, 18);
  els.boardHash.title = displayValue(state.boardHash);
  els.decisionId.textContent = shortValue(state.decisionId, 18);
  els.decisionId.title = displayValue(state.decisionId);
  const controller = displayValue(state.controller, "未报告具体策略");
  const controllerDescription = neuralPolicyAvailable && !state.lastDecision
    ? "MaleCNS · 待决策"
    : state.lastDecision?.controller_source === "go-rule-pass-gate"
      ? "围棋规则 PASS"
      : state.lastDecision?.controller_source === "malecns-linear-readout"
        ? "MaleCNS 神经读出"
        : controller;
  els.controllerLabel.textContent = controllerDescription;
  els.controllerLabel.title = state.lastDecision?.controller_source === "go-rule-pass-gate"
    ? "本次 PASS 由围棋规则门控；神经帧没有用于选择此步"
    : state.lastDecision?.controller_source === "malecns-linear-readout"
      ? "本次棋步由显示神经帧的已训练读出选择"
      : `${controllerDescription}；未声明具体控制来源`;
  const checkpoint = neuralPolicyAvailable && neuralPolicyStatus?.checkpoint_hash
    ? neuralPolicyStatus.checkpoint_hash
    : state.checkpoint;
  els.checkpointLabel.textContent = shortValue(checkpoint, 22);
  els.checkpointLabel.title = displayValue(checkpoint);
  els.rulesStatus.textContent = state.gameOver
    ? `对局结束${state.endReason === "resignation" ? " · 黑方认输" : ""}${state.winner ? ` · ${colorName(state.winner)}胜` : ""}`
    : "合法性由本地 9×9 围棋引擎判定";
  renderResult(state);
  updateControls();
  handleNeuralStateChange(state);
}

function handleNeuralStateChange(state) {
  const key = stateKey(state);
  if (preserveNeuralFrameForStateKey === key && neuralFrame) {
    neuralTargetStateKey = key;
    setNeuralChip("ready", "决策帧同步");
    return;
  }
  preserveNeuralFrameForStateKey = null;
  if (key === neuralTargetStateKey) return;
  neuralTargetStateKey = key;
  neuralRequestSerial += 1;
  window.clearTimeout(neuralRetryTimer);
  neuralFrameAbort?.abort();
  neuralFrameAbort = null;
  neuralRequestBusy = false;
  setTextWithTitle(els.neuralDecisionId, null, "待生成");
  setTextWithTitle(els.neuralBoardHash, null, "待生成");
  if (neuralPolicyAvailable) {
    setNeuralChip("ready", "等待决策");
    if (state.toMove === "white" && !state.gameOver) {
      clearNeuralFrame("loading", "等待 MaleCNS 神经策略", "即将计算的这一帧将同时决定果蝇的下一步棋");
    } else {
      clearNeuralFrame("loading", "等待下一次真实神经决策", "请先落一枚黑棋；这里只显示真正用于白棋落子的决策帧");
    }
  } else if (neuralRuntimeAvailable) {
    requestNeuralFrame({ force: true });
  } else {
    const knownUnavailable = els.neuralStatusChip.classList.contains("is-unavailable");
    clearNeuralFrame(
      knownUnavailable ? "unavailable" : "loading",
      knownUnavailable ? "神经运行时不可用" : "等待神经运行时",
      "当前棋局尚无可显示的神经帧",
    );
  }
}

function renderResult(state) {
  els.resultBanner.classList.toggle("is-pending", !state.gameOver);
  els.resultBanner.classList.toggle("is-final", state.gameOver);
  if (!state.score) {
    els.resultTitle.textContent = state.gameOver ? "自动判胜 · 结果不可获取" : "自动判胜 · 对局进行中";
    els.resultScore.textContent = state.gameOver
      ? "服务未提供计分数据；请检查棋局记录，不能凭画面推断胜负。"
      : "等待真实盘面计分。双方连续停手或认输后才会产生正式胜负。";
    return;
  }
  const black = Number(state.score.black_total);
  const white = Number(state.score.white_total);
  const margin = Number(state.score.margin);
  const scoreText = `黑 ${Number.isFinite(black) ? black.toFixed(1) : "—"} · 白 ${Number.isFinite(white) ? white.toFixed(1) : "—"}（白方含贴目 ${displayValue(state.score.komi, "—")}）`;
  if (!state.gameOver) {
    els.resultTitle.textContent = "自动判胜 · 对局进行中";
    els.resultScore.textContent = `当前盘面参考：${scoreText}。尚未终局，不代表胜负；双方连续停手后自动数目。`;
    return;
  }
  const winner = state.winner || normalizeColor(state.result?.winner);
  if (state.endReason === "resignation") {
    els.resultTitle.textContent = winner === "white" ? "自动判胜 · 白方胜（黑方认输）" : "自动判胜 · 认输结果待核对";
    els.resultScore.textContent = `认输结束。当前盘面参考：${scoreText}；此数字不是正式目差。`;
    return;
  }
  if (state.endReason !== "two_consecutive_passes" || state.score.final !== true || !Number.isFinite(margin)) {
    els.resultTitle.textContent = "自动判胜 · 终局结果待核对";
    els.resultScore.textContent = "服务未提供可验证的正式计分，暂不宣布胜方。";
    return;
  }
  els.resultTitle.textContent = winner === "black"
    ? `自动判胜 · 黑方胜 ${margin.toFixed(1)} 目`
    : winner === "white"
      ? `自动判胜 · 白方胜 ${margin.toFixed(1)} 目`
      : winner === "draw"
        ? "自动判胜 · 和棋"
        : "自动判胜 · 胜方待核对";
  els.resultScore.textContent = `双方连续停手 · 中国面积计分：${scoreText}。未进行死子协商。`;
}

function updateControls() {
  const available = Boolean(currentState) && !requestBusy;
  const animating = Boolean(currentAnimation);
  els.undoButton.disabled = !available || animating || !currentState.canUndo;
  els.resetButton.disabled = !available || animating;
  els.passButton.disabled = !available || animating || currentState.gameOver || currentState.toMove !== currentState.humanColor;
  els.resignButton.disabled = !available || animating || currentState.gameOver;
  els.botMoveButton.disabled = !available || animating || !neuralPolicyAvailable || currentState.gameOver || currentState.toMove !== "white";
  els.pauseButton.disabled = !animating;
  els.stepButton.disabled = !animating;
  els.roomViewport.classList.toggle("is-busy", requestBusy);
}

async function runAction(body, pendingMessage) {
  if (requestBusy) return null;
  requestBusy = true;
  updateControls();
  setInteraction(pendingMessage);
  try {
    const state = await postAction(body);
    setServiceStatus("ready", "本地围棋服务已连接");
    return state;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setInteraction(message, true);
    setServiceStatus("error", "围棋服务请求失败");
    return null;
  } finally {
    requestBusy = false;
    updateControls();
  }
}

async function humanMove(row, col) {
  if (!currentState || currentAnimation || requestBusy || currentState.gameOver || currentState.toMove !== currentState.humanColor) return;
  const state = await runAction({ action: "human_move", row, col }, `正在提交 ${coordinateName({ row, col })}…`);
  if (!state) return;
  applyState(state);
  setInteraction(`你已落子 ${coordinateName({ row, col })}`);
  if (!state.gameOver) await requestBotMove(true);
}

async function requestBotMove(automatic = false) {
  if (!currentState || currentAnimation || requestBusy || currentState.gameOver) return;
  if (!neuralPolicyAvailable) {
    setInteraction("已训练神经策略不可用；为避免伪造，未调用 baseline，也没有落子", true);
    return;
  }
  const preMoveState = currentState;
  requestBusy = true;
  updateControls();
  setInteraction(automatic ? "MaleCNS 正在传播并读取下一步…" : "正在用 MaleCNS 神经策略决定落点…");
  try {
    let state = null;
    let lastError = null;
    for (let attempt = 0; attempt < 4 && !state; attempt += 1) {
      try {
        state = await postNeuralMove(preMoveState);
      } catch (error) {
        lastError = error;
        const busy = error?.status === 409 || error?.code === "neural_busy";
        if (!busy || attempt === 3) throw error;
        await new Promise((resolve) => window.setTimeout(resolve, 140 * (attempt + 1)));
      }
    }
    if (!state) throw lastError ?? new Error("神经策略未返回棋步");
    setServiceStatus("ready", "神经策略与围棋服务已连接");
    startFlyAnimation(state);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setInteraction(`神经策略失败，棋盘未回退到 baseline：${message}`, true);
    setServiceStatus("error", "神经落子失败（无回退）");
  } finally {
    requestBusy = false;
    updateControls();
  }
}

async function humanPass() {
  if (!currentState || currentAnimation || requestBusy || currentState.gameOver || currentState.toMove !== currentState.humanColor) return;
  const state = await runAction({ action: "human_pass" }, "你选择停一手，正在请求果蝇回应…");
  if (!state) return;
  applyState(state);
  if (!state.gameOver) await requestBotMove(true);
}

async function humanResign() {
  if (!currentState || currentAnimation || requestBusy || currentState.gameOver) return;
  const state = await runAction({ action: "human_resign" }, "正在记录黑方认输…");
  if (!state) return;
  applyState(state);
  resetFlyPose();
  setFlyStage("对局结束", false);
  setInteraction("你已认输；白方果蝇获胜。可以悔棋撤销认输，或重新开始。");
}

async function undoMove() {
  if (!currentState || currentAnimation || requestBusy) return;
  const state = await runAction({ action: "undo" }, "正在悔棋…");
  if (!state) return;
  applyState(state);
  resetFlyPose();
  setFlyStage("等待落子", false);
  setInteraction("已恢复到上一个真实棋局快照");
}

async function resetGame() {
  if (currentAnimation) cancelFlyAnimation();
  const state = await runAction({ action: "reset" }, "正在重新开始…");
  if (!state) return;
  applyState(state);
  resetFlyPose();
  setFlyStage("等待落子", false);
  setInteraction("新的 9×9 对局已开始");
}

function makeMaterial(color, roughness = 0.72, metalness = 0) {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness });
}

function addBox(parent, size, position, material, rotation = null) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
  mesh.position.set(...position);
  if (rotation) mesh.rotation.set(...rotation);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function cylinderBetween(start, end, radius, material, radialSegments = 8) {
  const direction = new THREE.Vector3().subVectors(end, start);
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, direction.length(), radialSegments), material);
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function createScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1d160f);
  scene.fog = new THREE.Fog(0x1d160f, 13, 27);

  camera = new THREE.PerspectiveCamera(42, 1, 0.1, 70);
  renderer = new THREE.WebGLRenderer({ canvas: els.roomCanvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  scene.add(new THREE.HemisphereLight(0xf3d8ad, 0x352719, 1.15));
  const keyLight = new THREE.DirectionalLight(0xffdfad, 2.2);
  keyLight.position.set(-4, 11, 5);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.left = -10;
  keyLight.shadow.camera.right = 10;
  keyLight.shadow.camera.top = 10;
  keyLight.shadow.camera.bottom = -10;
  scene.add(keyLight);
  const fill = new THREE.PointLight(0xe6a24e, 20, 14, 2);
  fill.position.set(5, 5, -5);
  scene.add(fill);

  createRoom();
  createGoBoard();
  fly = createFly();
  scene.add(fly.root);
  resetFlyPose();

  carriedStone = createStoneMesh("white");
  carriedStone.visible = false;
  carriedStone.scale.set(0.92, 0.32, 0.92);
  scene.add(carriedStone);

  raycaster = new THREE.Raycaster();
  updateCamera();
  resizeRenderer();
  installSceneInteractions();
}

function createRoom() {
  const group = new THREE.Group();
  const wood = makeMaterial(0x4d2e18, 0.86);
  const darkWood = makeMaterial(0x25150d, 0.82);
  const plaster = makeMaterial(0xbca982, 1);
  const tatamiA = makeMaterial(0x8d8357, 1);
  const tatamiB = makeMaterial(0x766f49, 1);
  const paper = new THREE.MeshStandardMaterial({ color: 0xe7dcc4, roughness: 1, transparent: true, opacity: 0.84 });

  addBox(group, [18, 0.22, 15], [0, -0.14, 0], darkWood);
  for (let x = -6; x <= 6; x += 4) {
    for (let z = -4.5; z <= 4.5; z += 3) {
      addBox(group, [3.75, 0.08, 2.75], [x, 0.02, z], ((x + z) / 3) % 2 === 0 ? tatamiA : tatamiB);
      addBox(group, [3.78, 0.035, 0.07], [x, 0.07, z - 1.38], darkWood);
    }
  }
  addBox(group, [18, 8, 0.22], [0, 4, -7.35], plaster);
  addBox(group, [0.22, 8, 15], [-9, 4, 0], plaster);
  for (let x = -7.4; x < 8; x += 2.45) {
    addBox(group, [0.09, 5.6, 0.14], [x, 3.15, -7.19], darkWood);
  }
  for (let y = 0.4; y < 6.2; y += 1.35) {
    addBox(group, [17.5, 0.075, 0.14], [0, y, -7.19], darkWood);
  }
  for (let i = 0; i < 6; i += 1) {
    addBox(group, [2.2, 5.2, 0.07], [-7.4 + i * 2.45, 3.2, -7.08], paper);
  }
  addBox(group, [0.12, 8, 15], [-8.82, 4, 0], wood);
  addBox(group, [18, 0.18, 0.2], [0, 6.4, -7.12], wood);

  const lantern = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.5, 1.05, 20), new THREE.MeshStandardMaterial({ color: 0xf4d79d, emissive: 0x7b431b, emissiveIntensity: 0.5, roughness: 0.85 }));
  lantern.position.set(5.8, 4.6, -6.55);
  group.add(lantern);
  group.add(cylinderBetween(new THREE.Vector3(5.8, 5.12, -6.55), new THREE.Vector3(5.8, 6.4, -6.55), 0.025, darkWood));
  scene.add(group);
}

function createGoBoard() {
  const group = new THREE.Group();
  const boardMaterial = makeMaterial(0xc58b42, 0.76);
  const edgeMaterial = makeMaterial(0x6b3c19, 0.82);
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x2b1b10, transparent: true, opacity: 0.9 });
  const stoneBowl = makeMaterial(0x2c1710, 0.48);

  addBox(group, [8.4, 0.52, 8.4], [0, 0.54, 0], edgeMaterial);
  addBox(group, [7.65, 0.22, 7.65], [0, 0.84, 0], boardMaterial);
  addBox(group, [7.2, 0.52, 0.48], [0, 0.2, -3.45], edgeMaterial);
  addBox(group, [7.2, 0.52, 0.48], [0, 0.2, 3.45], edgeMaterial);
  addBox(group, [0.48, 0.52, 6.4], [-3.45, 0.2, 0], edgeMaterial);
  addBox(group, [0.48, 0.52, 6.4], [3.45, 0.2, 0], edgeMaterial);

  const points = [];
  for (let i = 0; i < BOARD_SIZE; i += 1) {
    const v = -GRID_SPAN / 2 + i * GRID_STEP;
    points.push(-GRID_SPAN / 2, BOARD_Y + 0.12, v, GRID_SPAN / 2, BOARD_Y + 0.12, v);
    points.push(v, BOARD_Y + 0.12, -GRID_SPAN / 2, v, BOARD_Y + 0.12, GRID_SPAN / 2);
  }
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  group.add(new THREE.LineSegments(lineGeometry, lineMaterial));
  const starMaterial = makeMaterial(0x24150d, 0.8);
  for (const row of [2, 4, 6]) {
    for (const col of [2, 4, 6]) {
      const point = boardPoint(row, col, BOARD_Y + 0.126);
      const star = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.018, 14), starMaterial);
      star.position.copy(point);
      group.add(star);
    }
  }

  boardHitPlane = new THREE.Mesh(new THREE.PlaneGeometry(7.35, 7.35), new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide }));
  boardHitPlane.rotation.x = -Math.PI / 2;
  boardHitPlane.position.y = BOARD_Y + 0.17;
  group.add(boardHitPlane);

  hoverMarker = new THREE.Mesh(new THREE.TorusGeometry(0.25, 0.026, 10, 32), new THREE.MeshBasicMaterial({ color: 0xf7c36f, transparent: true, opacity: 0.9 }));
  hoverMarker.rotation.x = Math.PI / 2;
  hoverMarker.visible = false;
  scene.add(hoverMarker);
  lastMoveMarker = new THREE.Mesh(new THREE.TorusGeometry(0.14, 0.025, 8, 28), new THREE.MeshBasicMaterial({ color: 0xe36e4b }));
  lastMoveMarker.rotation.x = Math.PI / 2;
  lastMoveMarker.visible = false;
  scene.add(lastMoveMarker);

  stonesGroup = new THREE.Group();
  scene.add(stonesGroup);

  const bowlGeometry = new THREE.SphereGeometry(0.72, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.48);
  const bowl = new THREE.Mesh(bowlGeometry, stoneBowl);
  bowl.scale.y = 0.46;
  bowl.rotation.x = Math.PI;
  bowl.position.set(4.35, 0.86, -2.85);
  bowl.castShadow = true;
  scene.add(bowl);
  for (let i = 0; i < 5; i += 1) {
    const stone = createStoneMesh("white");
    stone.position.set(4.35 + Math.cos(i * 2.2) * 0.23, 1.02 + i * 0.018, -2.85 + Math.sin(i * 2.2) * 0.2);
    stone.scale.set(0.85, 0.28, 0.85);
    scene.add(stone);
  }
  group.userData.board = true;
  scene.add(group);
}

function createStoneMesh(color) {
  const material = color === "white"
    ? new THREE.MeshStandardMaterial({ color: 0xf1eee4, roughness: 0.3, metalness: 0.02 })
    : new THREE.MeshStandardMaterial({ color: 0x171514, roughness: 0.26, metalness: 0.12 });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.31, 24, 14), material);
  mesh.scale.y = 0.42;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function renderStones(board, lastMove) {
  if (!stonesGroup) return;
  stonesGroup.clear();
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      const color = board[row][col];
      if (!color) continue;
      const stone = createStoneMesh(color);
      stone.position.copy(boardPoint(row, col));
      stonesGroup.add(stone);
    }
  }
  if (lastMove && !lastMove.pass && Number.isInteger(lastMove.row) && Number.isInteger(lastMove.col)) {
    lastMoveMarker.position.copy(boardPoint(lastMove.row, lastMove.col, BOARD_Y + 0.265));
    lastMoveMarker.visible = true;
  } else {
    lastMoveMarker.visible = false;
  }
}

function createFly() {
  const root = new THREE.Group();
  const bodyMaterial = makeMaterial(0x3a2a20, 0.7);
  const abdomenMaterial = makeMaterial(0x73502e, 0.64);
  const headMaterial = makeMaterial(0x4d3324, 0.7);
  const eyeMaterial = new THREE.MeshStandardMaterial({ color: 0xa62b27, roughness: 0.28, metalness: 0.12, emissive: 0x2a0504, emissiveIntensity: 0.22 });
  const legMaterial = makeMaterial(0x241812, 0.8);
  const wingMaterial = new THREE.MeshPhysicalMaterial({ color: 0xcde0dc, roughness: 0.25, transparent: true, opacity: 0.43, transmission: 0.1, side: THREE.DoubleSide });

  const scale = 0.9;
  root.scale.setScalar(scale);

  const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.46, 24, 18), bodyMaterial);
  thorax.scale.set(0.92, 0.82, 1.13);
  thorax.castShadow = true;
  root.add(thorax);

  const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.48, 24, 18), abdomenMaterial);
  abdomen.scale.set(0.72, 0.68, 1.36);
  abdomen.position.set(0, 0.01, 0.72);
  abdomen.rotation.x = -0.08;
  abdomen.castShadow = true;
  root.add(abdomen);

  const headPivot = new THREE.Group();
  headPivot.position.set(0, 0.05, -0.5);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.38, 24, 18), headMaterial);
  head.scale.set(1.08, 0.94, 0.86);
  head.castShadow = true;
  headPivot.add(head);
  for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.25, 20, 14), eyeMaterial);
    eye.position.set(side * 0.29, 0.07, -0.13);
    eye.scale.set(0.58, 0.9, 0.72);
    eye.castShadow = true;
    headPivot.add(eye);
  }
  root.add(headPivot);

  const antennae = [];
  for (const side of [-1, 1]) {
    const pivot = new THREE.Group();
    pivot.position.set(side * 0.13, 0.27, -0.75);
    pivot.add(cylinderBetween(new THREE.Vector3(), new THREE.Vector3(side * 0.12, 0.18, -0.3), 0.018, legMaterial, 7));
    const tip = new THREE.Mesh(new THREE.SphereGeometry(0.035, 8, 6), legMaterial);
    tip.position.set(side * 0.12, 0.18, -0.3);
    pivot.add(tip);
    headPivot.add(pivot);
    antennae.push(pivot);
  }

  const wings = [];
  for (const side of [-1, 1]) {
    const wing = new THREE.Mesh(new THREE.SphereGeometry(0.62, 24, 10), wingMaterial);
    wing.scale.set(0.55, 0.08, 1.45);
    wing.position.set(side * 0.38, 0.31, 0.48);
    wing.rotation.y = side * 0.34;
    wing.rotation.z = side * -0.18;
    root.add(wing);
    wings.push(wing);
  }

  const legs = [];
  const frontLegs = [];
  const zOffsets = [-0.28, 0.12, 0.45];
  for (let pair = 0; pair < 3; pair += 1) {
    for (const side of [-1, 1]) {
      const pivot = new THREE.Group();
      pivot.position.set(side * 0.31, -0.22, zOffsets[pair]);
      const forward = pair === 0 ? -0.34 : pair === 1 ? 0.05 : 0.32;
      const knee = new THREE.Vector3(side * (0.45 + pair * 0.05), -0.38, forward);
      const foot = new THREE.Vector3(side * (0.78 + pair * 0.08), -0.7, forward + (pair - 1) * 0.34);
      pivot.add(cylinderBetween(new THREE.Vector3(), knee, 0.026, legMaterial, 7));
      pivot.add(cylinderBetween(knee, foot, 0.022, legMaterial, 7));
      const tarsus = cylinderBetween(foot, new THREE.Vector3(foot.x + side * 0.14, foot.y - 0.02, foot.z - 0.11), 0.014, legMaterial, 6);
      pivot.add(tarsus);
      root.add(pivot);
      legs.push(pivot);
      if (pair === 0) frontLegs.push(pivot);
    }
  }

  const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.85, 32), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.16, depthWrite: false }));
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = -0.78;
  root.add(shadow);

  return { root, headPivot, antennae, wings, legs, frontLegs };
}

function resetFlyPose() {
  if (!fly) return;
  fly.root.position.copy(FLY_REST);
  fly.root.rotation.set(0, -0.78, 0);
  fly.headPivot.rotation.set(0, 0, 0);
  fly.antennae.forEach((antenna) => antenna.rotation.set(0, 0, 0));
  fly.frontLegs.forEach((leg) => leg.rotation.set(0, 0, 0));
  fly.wings.forEach((wing, index) => { wing.rotation.x = 0; wing.rotation.z = (index === 0 ? 1 : -1) * 0.18; });
  if (carriedStone) carriedStone.visible = false;
}

function faceTowards(from, to) {
  const dx = to.x - from.x;
  const dz = to.z - from.z;
  return Math.atan2(-dx, -dz);
}

function startFlyAnimation(nextState) {
  const move = nextState.lastMove;
  if (!move || move.pass || !Number.isInteger(move.row) || !Number.isInteger(move.col)) {
    applyState(nextState);
    setFlyStage(move?.pass ? "果蝇选择停一手" : "决策已完成", false);
    setInteraction(move?.pass ? "果蝇选择停一手" : "果蝇决策已应用（落点不可获取）");
    return;
  }
  const target = boardPoint(move.row, move.col, BOARD_Y + 0.24);
  const targetRoot = outsideRootForTarget(target);
  currentAnimation = {
    nextState,
    move,
    target,
    targetRoot,
    travelPath: perimeterWaypoints(targetRoot),
    stageIndex: 0,
    elapsed: 0,
    applied: false,
    speed: Number(els.speedControl.value) || 1,
  };
  animationPaused = false;
  els.pauseButton.textContent = "暂停";
  els.pauseButton.setAttribute("aria-pressed", "false");
  enterAnimationStage();
  setInteraction(`果蝇正在执行 ${coordinateName(move)} 的落子动作`);
  updateControls();
}

function cancelFlyAnimation() {
  currentAnimation = null;
  animationPaused = false;
  stepRequested = false;
  resetFlyPose();
  updateControls();
}

function enterAnimationStage() {
  if (!currentAnimation) return;
  const stage = ACTION_STAGES[currentAnimation.stageIndex];
  currentAnimation.elapsed = 0;
  setFlyStage(stage.label, true);
}

function advanceAnimationStage() {
  if (!currentAnimation) return;
  currentAnimation.stageIndex += 1;
  if (currentAnimation.stageIndex >= ACTION_STAGES.length) {
    const finished = currentAnimation;
    if (!finished.applied) applyState(finished.nextState);
    currentAnimation = null;
    carriedStone.visible = false;
    resetFlyPose();
    setFlyStage(finished.nextState.gameOver ? "对局结束" : "观察对手落子", false);
    setInteraction(`果蝇已落子 ${coordinateName(finished.move)}`);
    updateControls();
    return;
  }
  enterAnimationStage();
}

function updateFlyAnimation(deltaMs, timestamp) {
  if (!currentAnimation) {
    if (fly) {
      const idle = timestamp * 0.001;
      fly.wings.forEach((wing, index) => { wing.rotation.x = Math.sin(idle * 3.2 + index) * 0.018; });
      fly.antennae.forEach((antenna, index) => { antenna.rotation.y = Math.sin(idle * 1.7 + index * 1.3) * 0.08; });
    }
    return;
  }
  const stage = ACTION_STAGES[currentAnimation.stageIndex];
  if (animationPaused && !stepRequested) return;
  if (stepRequested) {
    currentAnimation.elapsed = stage.duration;
    stepRequested = false;
  } else {
    currentAnimation.elapsed += deltaMs * currentAnimation.speed;
  }
  const linear = clamp(currentAnimation.elapsed / stage.duration, 0, 1);
  poseAnimation(stage.key, easeInOut(linear), timestamp);
  if (linear >= 1) advanceAnimationStage();
}

function poseAnimation(stage, t, timestamp) {
  const anim = currentAnimation;
  if (!anim) return;
  const tremor = Math.sin(timestamp * 0.022) * 0.045;
  if (stage === "observe") {
    fly.root.position.copy(FLY_REST);
    fly.headPivot.rotation.y = Math.sin(t * Math.PI * 2) * 0.2;
    fly.antennae[0].rotation.z = -0.12 - t * 0.16;
    fly.antennae[1].rotation.z = 0.12 + t * 0.16;
  } else if (stage === "think") {
    fly.headPivot.rotation.y = Math.sin(t * Math.PI * 4) * 0.16;
    fly.headPivot.rotation.x = -0.08 + Math.sin(t * Math.PI * 3) * 0.07;
    fly.antennae.forEach((antenna, index) => {
      antenna.rotation.y = Math.sin(t * Math.PI * 9 + index * Math.PI) * 0.24;
      antenna.rotation.z = (index === 0 ? -1 : 1) * (0.15 + Math.abs(tremor));
    });
  } else if (stage === "approach") {
    fly.root.position.lerpVectors(FLY_REST, FLY_BOWL, t);
    fly.root.rotation.y = faceTowards(fly.root.position, BOWL_PICKUP);
    applyWalkingPose(t);
  } else if (stage === "grasp") {
    fly.root.position.copy(FLY_BOWL);
    fly.root.rotation.y = faceTowards(fly.root.position, BOWL_PICKUP);
    poseFrontLegs(t);
    carriedStone.visible = t > 0.46;
    carriedStone.position.copy(BOWL_PICKUP);
  } else if (stage === "carry") {
    fly.root.position.copy(pointAlongPath(anim.travelPath, t));
    fly.root.rotation.y = faceTowards(fly.root.position, anim.target);
    poseFrontLegs(1);
    applyWalkingPose(t);
    carriedStone.visible = true;
    carriedStone.position.lerpVectors(BOWL_PICKUP, anim.target.clone().add(new THREE.Vector3(0, 0.66, 0)), t);
  } else if (stage === "release") {
    fly.root.position.copy(anim.targetRoot);
    fly.root.rotation.y = faceTowards(fly.root.position, anim.target);
    poseFrontLegs(1 - t * 0.35);
    carriedStone.visible = !anim.applied;
    carriedStone.position.copy(anim.target).add(new THREE.Vector3(0, (1 - t) * 0.66, 0));
    if (t >= 0.7 && !anim.applied) {
      anim.applied = true;
      carriedStone.visible = false;
      applyState(anim.nextState);
    }
  } else if (stage === "retract") {
    fly.root.position.copy(anim.targetRoot);
    poseFrontLegs(1 - t);
    fly.headPivot.rotation.x = -0.12 * (1 - t);
  } else if (stage === "return") {
    const returnPath = [...anim.travelPath].reverse().concat([FLY_REST]);
    fly.root.position.copy(pointAlongPath(returnPath, t));
    fly.root.rotation.y = faceTowards(fly.root.position, FLY_REST.clone().add(new THREE.Vector3(-1, 0, -1)));
    applyWalkingPose(t);
    if (t > 0.85) fly.root.rotation.y = -0.78;
  }
}

function poseFrontLegs(amount) {
  if (!fly) return;
  fly.frontLegs.forEach((leg, index) => {
    leg.rotation.x = -0.7 * amount;
    leg.rotation.y = (index === 0 ? -1 : 1) * 0.38 * amount;
    leg.rotation.z = (index === 0 ? 1 : -1) * 0.26 * amount;
  });
}

function applyWalkingPose(progress) {
  fly.legs.forEach((leg, index) => {
    if (fly.frontLegs.includes(leg) && carriedStone.visible) return;
    leg.rotation.x = Math.sin(progress * Math.PI * 7 + index * 1.5) * 0.18;
  });
  fly.root.position.y += Math.abs(Math.sin(progress * Math.PI * 6)) * 0.05;
}

function projectedNeuralNodes(nodes, width, height) {
  const angle = 0.62;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const points = nodes.map((node) => {
    const [x, y, z] = node.soma_xyz;
    const horizontal = x * cos - z * sin;
    const depth = x * sin + z * cos;
    return { node, rawX: horizontal, rawY: y - depth * 0.2, depth };
  });
  const xs = points.map((point) => point.rawX);
  const ys = points.map((point) => point.rawY);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const padding = Math.min(30, Math.max(14, Math.min(width, height) * 0.09));
  const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
  const usedWidth = spanX * scale;
  const usedHeight = spanY * scale;
  const offsetX = (width - usedWidth) / 2;
  const offsetY = (height - usedHeight) / 2;
  return points.map((point) => ({
    ...point,
    x: offsetX + (point.rawX - minX) * scale,
    y: height - offsetY - (point.rawY - minY) * scale,
  }));
}

function latestSpikeDelta(ticks, tick) {
  if (!ticks?.length) return Infinity;
  let low = 0;
  let high = ticks.length - 1;
  let found = -1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (ticks[middle] <= tick) {
      found = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return found < 0 ? Infinity : tick - ticks[found];
}

function nodePulse(node, tick) {
  const byBody = neuralFrame.spikeTicksByBody.get(String(node.body_id));
  const byModel = neuralFrame.spikeTicksByModel.get(String(node.model_index));
  const delta = latestSpikeDelta(byBody?.length ? byBody : byModel, tick);
  return delta >= 0 && delta < 3 ? 1 - delta / 3 : 0;
}

function drawNeuralFrame(tick) {
  if (!neuralFrame) return;
  const shownTick = clamp(Math.floor(tick), 0, neuralFrame.tickCount - 1);
  if (neuralBrainFiring) {
    const firing = neuralBrainFiring.array;
    firing.fill(0);
    for (let delta = 0; delta < 3; delta += 1) {
      const indices = neuralFrame.spikeModelIndicesByTick.get(shownTick - delta) ?? [];
      const strength = 1 - delta / 3;
      for (const index of indices) {
        if (Number.isInteger(index) && index >= 0 && index < firing.length) firing[index] = Math.max(firing[index], strength);
      }
    }
    neuralBrainFiring.needsUpdate = true;
  }
  neuralLastDrawnTick = shownTick;
  els.neuralTick.textContent = `${shownTick + 1}/${neuralFrame.tickCount}`;
  els.neuralTime.textContent = neuralFrame.dtMs > 0 ? `${(shownTick * neuralFrame.dtMs).toFixed(1)} ms` : "未报告";
}

function updateNeuralAnimation(timestamp) {
  if (!neuralFrame || neuralPlaybackStartedAt <= 0) {
    renderNeuralBrain(timestamp);
    return;
  }
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const duration = Math.max(1, neuralFrame.tickCount * NEURAL_TICK_DISPLAY_MS);
  const elapsed = reduceMotion ? duration : timestamp - neuralPlaybackStartedAt;
  const progress = clamp(elapsed / duration, 0, 1);
  const tick = progress * Math.max(0, neuralFrame.tickCount - 1);
  drawNeuralFrame(tick);
  const shownTick = Math.floor(tick);
  const realTickCount = Number(neuralFrame.raw.tick_spike_counts?.[shownTick]) || 0;
  if ((Number(neuralFrame.raw.total_spike_count) || 0) > 0) {
    els.neuralSpikeState.textContent = `tick ${shownTick}: ${realTickCount} spike${realTickCount === 1 ? "" : "s"} · recorded_spikes 驱动脉冲`;
  }
  if (progress >= 1) {
    neuralPlaybackStartedAt = 0;
    els.neuralPlaybackState.textContent = (Number(neuralFrame.raw.total_spike_count) || 0) === 0
      ? "真实帧：静默"
      : "真实帧回放完成";
  }
  renderNeuralBrain(timestamp);
}

function updateCamera() {
  const cosPitch = Math.cos(cameraOrbit.pitch);
  camera.position.set(
    cameraOrbit.target.x + cameraOrbit.distance * Math.sin(cameraOrbit.yaw) * cosPitch,
    cameraOrbit.target.y + cameraOrbit.distance * Math.sin(cameraOrbit.pitch),
    cameraOrbit.target.z + cameraOrbit.distance * Math.cos(cameraOrbit.yaw) * cosPitch,
  );
  camera.lookAt(cameraOrbit.target);
}

function resetCamera() {
  cameraOrbit.yaw = 0.72;
  cameraOrbit.pitch = 0.83;
  cameraOrbit.distance = 15;
  cameraOrbit.target.set(0, 0.75, 0);
  updateCamera();
}

function resizeRenderer() {
  if (!renderer) return;
  const width = Math.max(1, els.roomViewport.clientWidth);
  const height = Math.max(1, els.roomViewport.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function pointerToBoard(event) {
  if (!raycaster || !boardHitPlane) return null;
  const rect = els.roomCanvas.getBoundingClientRect();
  const pointer = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(boardHitPlane, false)[0];
  if (!hit) return null;
  const col = Math.round((hit.point.x + GRID_SPAN / 2) / GRID_STEP);
  const row = Math.round((hit.point.z + GRID_SPAN / 2) / GRID_STEP);
  if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return null;
  return { row, col };
}

function installSceneInteractions() {
  els.roomViewport.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    pointerDown = { x: event.clientX, y: event.clientY, yaw: cameraOrbit.yaw, pitch: cameraOrbit.pitch };
    pointerMoved = false;
    els.roomViewport.setPointerCapture(event.pointerId);
    els.roomViewport.classList.add("is-dragging");
  });
  els.roomViewport.addEventListener("pointermove", (event) => {
    if (pointerDown) {
      const dx = event.clientX - pointerDown.x;
      const dy = event.clientY - pointerDown.y;
      if (Math.hypot(dx, dy) > 4) pointerMoved = true;
      if (pointerMoved) {
        cameraOrbit.yaw = pointerDown.yaw - dx * 0.008;
        cameraOrbit.pitch = clamp(pointerDown.pitch + dy * 0.006, 0.28, 1.36);
        updateCamera();
      }
      return;
    }
    const point = pointerToBoard(event);
    if (!point || currentAnimation || requestBusy || !currentState || currentState.toMove !== currentState.humanColor || currentState.board[point.row][point.col]) {
      hoverMarker.visible = false;
      return;
    }
    hoverMarker.position.copy(boardPoint(point.row, point.col, BOARD_Y + 0.19));
    hoverMarker.visible = true;
  });
  const finishPointer = (event) => {
    if (!pointerDown) return;
    if (!pointerMoved) {
      const point = pointerToBoard(event);
      if (point) humanMove(point.row, point.col);
    }
    pointerDown = null;
    els.roomViewport.classList.remove("is-dragging");
  };
  els.roomViewport.addEventListener("pointerup", finishPointer);
  els.roomViewport.addEventListener("pointercancel", () => {
    pointerDown = null;
    els.roomViewport.classList.remove("is-dragging");
  });
  els.roomViewport.addEventListener("pointerleave", () => { if (!pointerDown) hoverMarker.visible = false; });
  els.roomViewport.addEventListener("wheel", (event) => {
    event.preventDefault();
    cameraOrbit.distance = clamp(cameraOrbit.distance + event.deltaY * 0.012, 7.5, 22);
    updateCamera();
  }, { passive: false });
  window.addEventListener("resize", resizeRenderer);
  new ResizeObserver(resizeRenderer).observe(els.roomViewport);
}

function installUiInteractions() {
  els.undoButton.addEventListener("click", undoMove);
  els.resetButton.addEventListener("click", resetGame);
  els.passButton.addEventListener("click", humanPass);
  els.resignButton.addEventListener("click", humanResign);
  els.botMoveButton.addEventListener("click", () => requestBotMove(false));
  els.rerunNeuralButton.addEventListener("click", () => {
    if (neuralPolicyAvailable) {
      if (!neuralFrame) return;
      neuralPlaybackStartedAt = performance.now();
      neuralLastDrawnTick = -1;
      els.neuralPlaybackState.textContent = "重新播放同一真实决策帧";
      drawNeuralFrame(0);
      return;
    }
    requestNeuralFrame({ force: true });
  });
  els.cameraReset.addEventListener("click", resetCamera);
  els.fullscreenButton.addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await els.roomViewport.requestFullscreen();
    } catch (error) {
      setInteraction(`无法切换全屏：${error.message}`, true);
    }
  });
  document.addEventListener("fullscreenchange", () => {
    els.fullscreenButton.textContent = document.fullscreenElement ? "退出全屏" : "全屏";
    setTimeout(resizeRenderer, 30);
  });
  els.speedControl.addEventListener("input", () => {
    const speed = Number(els.speedControl.value);
    els.speedOutput.textContent = `${speed}×`;
    if (currentAnimation) currentAnimation.speed = speed;
  });
  els.pauseButton.addEventListener("click", () => {
    if (!currentAnimation) return;
    animationPaused = !animationPaused;
    els.pauseButton.textContent = animationPaused ? "继续" : "暂停";
    els.pauseButton.setAttribute("aria-pressed", String(animationPaused));
    setInteraction(animationPaused ? "果蝇动作已暂停，可用“单步”逐阶段查看" : "果蝇动作继续播放");
  });
  els.stepButton.addEventListener("click", () => {
    if (!currentAnimation) return;
    animationPaused = true;
    stepRequested = true;
    els.pauseButton.textContent = "继续";
    els.pauseButton.setAttribute("aria-pressed", "true");
  });
}

function animate(timestamp) {
  const delta = Math.min(50, timestamp - lastFrameTime);
  lastFrameTime = timestamp;
  updateFlyAnimation(delta, timestamp);
  updateNeuralAnimation(timestamp);
  renderer.render(scene, camera);
  animationFrame = requestAnimationFrame(animate);
}

async function boot() {
  installUiInteractions();
  try {
    createNeuralBrainScene();
    resizeNeuralCanvas();
    new ResizeObserver(resizeNeuralCanvas).observe(els.neuralViewport);
    createScene();
    animationFrame = requestAnimationFrame(animate);
  } catch (error) {
    showSceneError(`无法启动 WebGL 三维场景：${error.message}`);
    setServiceStatus("error", "WebGL 场景启动失败");
    return;
  }
  try {
    const state = await fetchState();
    applyState(state);
    setFlyStage("观察棋盘", false);
    setInteraction("点击棋盘交叉点落子；果蝇将在服务决策后执行动作");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setServiceStatus("error", "无法连接本地围棋服务");
    setInteraction(message, true);
    showSceneError(`围棋 API 尚未就绪：${message}`);
  }
  await fetchNeuralPolicyStatus();
  await fetchEvaluationStatus();
  await fetchTournamentStatus();
  await fetchStage7Status();
  await fetchStage8Status();
  if (await fetchNeuralStatus() && !neuralPolicyAvailable) {
    await requestNeuralFrame({ force: true });
  } else if (neuralPolicyAvailable && currentState) {
    neuralTargetStateKey = null;
    handleNeuralStateChange(currentState);
  }
  window.setInterval(async () => {
    if (requestBusy || currentAnimation || document.hidden) return;
    try {
      await fetchState({ quiet: true });
    } catch {
      setServiceStatus("error", "本地围棋服务连接中断");
    }
  }, 5000);
}

window.addEventListener("beforeunload", () => {
  window.clearTimeout(neuralRetryTimer);
  neuralFrameAbort?.abort();
  cancelAnimationFrame(animationFrame);
});
boot();
