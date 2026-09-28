"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const elements = {
  controls: document.querySelector("#controls"),
  edgesEnabled: document.querySelector("#edgesEnabled"),
  edgeStateText: document.querySelector("#edgeStateText"),
  speed: document.querySelector("#speed"),
  speedValue: document.querySelector("#speedValue"),
  playButton: document.querySelector("#playButton"),
  playIcon: document.querySelector("#playIcon"),
  playLabel: document.querySelector("#playLabel"),
  resetButton: document.querySelector("#resetButton"),
  requestStatus: document.querySelector("#requestStatus"),
  verification: document.querySelector(".verification"),
  verificationText: document.querySelector("#verificationText"),
  graphShell: document.querySelector("#graphShell"),
  edgesLayer: document.querySelector("#edgesLayer"),
  pulsesLayer: document.querySelector("#pulsesLayer"),
  nodesLayer: document.querySelector("#nodesLayer"),
  timelineProgress: document.querySelector("#timelineProgress"),
  stepMarkers: document.querySelector("#stepMarkers"),
  stepMetric: document.querySelector("#stepMetric"),
  timeMetric: document.querySelector("#timeMetric"),
  stimulusMetric: document.querySelector("#stimulusMetric"),
  spikeMetric: document.querySelector("#spikeMetric"),
  spikeDetail: document.querySelector("#spikeDetail"),
  connectionMetric: document.querySelector("#connectionMetric"),
  stepNarrative: document.querySelector("#stepNarrative"),
  hashValue: document.querySelector("#hashValue"),
};

const state = {
  status: null,
  run: null,
  stepIndex: 0,
  playing: false,
  timer: null,
  requestSerial: 0,
  paths: new Map(),
  nodeElements: new Map(),
};

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function stopPlayback() {
  window.clearTimeout(state.timer);
  state.timer = null;
  state.playing = false;
  elements.playIcon.textContent = "▶";
  elements.playLabel.textContent = state.run && state.stepIndex === state.run.simulation.steps.length - 1 ? "重播" : "播放";
}

function setBusy(isBusy, message) {
  elements.requestStatus.textContent = message;
  elements.requestStatus.classList.remove("is-error");
  for (const control of elements.controls.elements) control.disabled = isBusy;
  elements.playButton.disabled = isBusy || !state.run;
  elements.resetButton.disabled = isBusy || !state.run;
  elements.graphShell.setAttribute("aria-busy", String(isBusy));
}

function selectedScenario() {
  return new FormData(elements.controls).get("scenario");
}

async function loadStatus() {
  const response = await fetch("/api/status", {headers: {"Accept": "application/json"}});
  const payload = await response.json();
  if (!response.ok || !payload.ok) throw new Error(payload.error?.message || "状态接口不可用");
  if (payload.topology.node_count !== 12 || payload.topology.edge_count !== 10) {
    throw new Error("服务端拓扑规模与 P1 预期不符");
  }
  state.status = payload;
  elements.hashValue.textContent = payload.p1_canonical_sha256;
  elements.hashValue.title = payload.p1_canonical_sha256;
  elements.verification.classList.add("is-ready");
  elements.verificationText.textContent = "P1 哈希已验证 · 本地服务在线";
  buildGraph(payload.topology);
}

async function requestSimulation() {
  stopPlayback();
  const serial = ++state.requestSerial;
  const scenario = selectedScenario();
  const edgesEnabled = elements.edgesEnabled.checked;
  setBusy(true, "本地计算中…");
  try {
    const response = await fetch("/api/simulate", {
      method: "POST",
      headers: {"Content-Type": "application/json", "Accept": "application/json"},
      body: JSON.stringify({scenario, edges_enabled: edgesEnabled}),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error?.message || "仿真请求失败");
    if (serial !== state.requestSerial) return;
    state.run = payload;
    state.stepIndex = 0;
    elements.stepMarkers.replaceChildren(...payload.simulation.steps.map((step) => {
      const marker = document.createElement("span");
      marker.textContent = `${step.time_ms} ms`;
      return marker;
    }));
    renderStep();
    setBusy(false, "结果来自本地 Python 模型");
  } catch (error) {
    if (serial !== state.requestSerial) return;
    setBusy(false, "运行失败");
    elements.requestStatus.classList.add("is-error");
    elements.stepNarrative.textContent = error instanceof Error ? error.message : "未知错误";
  }
}

function positionNodes(topology) {
  const edgesBySource = new Map();
  for (const edge of topology.edges) {
    if (!edgesBySource.has(edge.pre)) edgesBySource.set(edge.pre, []);
    edgesBySource.get(edge.pre).push(edge.post);
  }
  const positions = new Map();
  const sourceConfig = [
    {id: 12781, sourceY: 180, childYs: [62, 122, 182, 242, 302]},
    {id: 556329, sourceY: 510, childYs: [390, 450, 510, 570, 630]},
  ];
  for (const config of sourceConfig) {
    positions.set(config.id, {x: 155, y: config.sourceY});
    const children = edgesBySource.get(config.id) || [];
    children.forEach((id, index) => positions.set(id, {x: 775, y: config.childYs[index]}));
  }
  return positions;
}

function buildGraph(topology) {
  const positions = positionNodes(topology);
  elements.edgesLayer.replaceChildren();
  elements.nodesLayer.replaceChildren();
  state.paths.clear();
  state.nodeElements.clear();

  for (const edge of topology.edges) {
    const start = positions.get(edge.pre);
    const end = positions.get(edge.post);
    if (!start || !end) continue;
    const bend = (end.y - start.y) * 0.18;
    const d = `M ${start.x + 40} ${start.y} C 370 ${start.y + bend}, 560 ${end.y - bend}, ${end.x - 38} ${end.y}`;
    const path = svgElement("path", {
      d,
      class: "edge-path",
      "data-pre": edge.pre,
      "data-post": edge.post,
      "stroke-width": (1.5 + edge.weight / 145).toFixed(2),
    });
    const labelX = 500;
    const labelY = start.y * .45 + end.y * .55;
    const background = svgElement("rect", {x: labelX - 21, y: labelY - 11, width: 42, height: 22, rx: 7, class: "edge-weight-bg"});
    const label = svgElement("text", {x: labelX, y: labelY, class: "edge-weight", "data-edge-label": `${edge.pre}-${edge.post}`});
    label.textContent = String(edge.weight);
    const title = svgElement("title");
    title.textContent = `${edge.pre} 到 ${edge.post}，连接权重 ${edge.weight}`;
    path.append(title);
    elements.edgesLayer.append(path, background, label);
    state.paths.set(`${edge.pre}-${edge.post}`, path);
  }

  for (const node of topology.nodes) {
    const position = positions.get(node.id);
    if (!position) continue;
    const source = node.role === "source";
    const radius = source ? 38 : 32;
    const circumference = 2 * Math.PI * (radius + 7);
    const group = svgElement("g", {
      class: `node ${node.role}`,
      transform: `translate(${position.x} ${position.y})`,
      tabindex: "0",
      role: "img",
      "aria-label": `${source ? "DNge104 源" : "下游"}节点 ${node.id}，类型 ${node.type}`,
      "data-node-id": node.id,
    });
    const hit = svgElement("circle", {r: radius + 13, class: "node-hit"});
    const potential = svgElement("circle", {
      r: radius + 7,
      class: "node-potential",
      "stroke-dasharray": `0 ${circumference}`,
      "data-circumference": circumference,
    });
    const body = svgElement("circle", {r: radius, class: "node-body"});
    const idText = svgElement("text", {y: source ? 5 : 4, class: "node-id"});
    idText.textContent = String(node.id);
    const typeText = svgElement("text", {y: radius + 24, class: "node-type"});
    typeText.textContent = node.type;
    group.append(hit, potential, body, idText, typeText);
    if (node.engineering_side) {
      const side = svgElement("text", {y: -radius - 19, class: "node-side"});
      side.textContent = node.engineering_side === "left" ? "左源 · 工程标签" : "右源 · 工程标签";
      group.append(side);
    }
    elements.nodesLayer.append(group);
    state.nodeElements.set(node.id, group);
  }
}

function addPulse(path, key) {
  if (reducedMotion.matches) return;
  const pulse = svgElement("circle", {r: 7, class: "pulse"});
  const motion = svgElement("animateMotion", {
    dur: `${Math.max(.45, 1.05 / Number(elements.speed.value))}s`,
    repeatCount: "indefinite",
    path: path.getAttribute("d"),
  });
  const title = svgElement("title");
  title.textContent = `活动沿连接 ${key} 传播`;
  pulse.append(title, motion);
  elements.pulsesLayer.append(pulse);
}

function renderStep() {
  if (!state.run || !state.status) return;
  const simulation = state.run.simulation;
  const step = simulation.steps[state.stepIndex];
  const previous = state.stepIndex > 0 ? simulation.steps[state.stepIndex - 1] : null;
  const spikes = new Set(step.spikes);
  const previousSpikes = new Set(previous?.spikes || []);
  const sourceIds = new Set(state.status.topology.nodes.filter((node) => node.role === "source").map((node) => node.id));
  const downstreamThisStep = step.spikes.filter((id) => !sourceIds.has(id)).length;

  elements.pulsesLayer.replaceChildren();
  for (const edge of state.status.topology.edges) {
    const key = `${edge.pre}-${edge.post}`;
    const path = state.paths.get(key);
    if (!path) continue;
    const active = simulation.edges_enabled && previousSpikes.has(edge.pre);
    path.classList.toggle("is-disabled", !simulation.edges_enabled);
    path.classList.toggle("is-active", active);
    const label = elements.edgesLayer.querySelector(`[data-edge-label="${key}"]`);
    label?.classList.toggle("is-disabled", !simulation.edges_enabled);
    if (active) addPulse(path, key);
  }

  for (const node of state.status.topology.nodes) {
    const group = state.nodeElements.get(node.id);
    if (!group) continue;
    const potential = Number(step.pre_reset_potential[String(node.id)] || 0);
    const ring = group.querySelector(".node-potential");
    const circumference = Number(ring.dataset.circumference);
    ring.setAttribute("stroke-dasharray", `${potential * circumference} ${circumference}`);
    group.classList.toggle("is-spiking", spikes.has(node.id));
    group.setAttribute("aria-label", `${node.role === "source" ? "源" : "下游"}节点 ${node.id}，类型 ${node.type}，本步电位 ${potential.toFixed(3)}${spikes.has(node.id) ? "，本步放电" : ""}`);
  }

  const lastIndex = simulation.steps.length - 1;
  elements.timelineProgress.style.width = `${lastIndex ? state.stepIndex / lastIndex * 100 : 100}%`;
  elements.stepMetric.textContent = `${step.step} / ${lastIndex}`;
  elements.timeMetric.textContent = `${step.time_ms.toFixed(1)} ms`;
  elements.stimulusMetric.textContent = simulation.stimulus.nodes.length ? simulation.stimulus.nodes.join(" + ") : "无";
  elements.spikeMetric.textContent = `${downstreamThisStep} / ${simulation.summary.downstream_spike_count}`;
  elements.spikeDetail.textContent = "当前步 / 全程";
  elements.connectionMetric.textContent = simulation.edges_enabled ? "10 条开启" : "全部断开";
  elements.edgeStateText.textContent = simulation.edges_enabled ? "已开启 · 10 条" : "已断开 · 0 条传播";

  if (step.step === 0 && simulation.stimulus.nodes.length) {
    elements.stepNarrative.textContent = `第 0 步：外部脉冲注入工程源 ${simulation.stimulus.nodes.join("、")}，源节点放电。`;
  } else if (step.step === 0) {
    elements.stepNarrative.textContent = "第 0 步：没有外部刺激，所有节点保持静默。";
  } else if (!simulation.edges_enabled) {
    elements.stepNarrative.textContent = `第 ${step.step} 步：连接已断开，没有下游传播或下游放电。`;
  } else if (downstreamThisStep > 0) {
    elements.stepNarrative.textContent = `第 ${step.step} 步：权重驱动下游电位，${downstreamThisStep} 个下游节点越过工程阈值并放电。`;
  } else {
    elements.stepNarrative.textContent = `第 ${step.step} 步：电位按工程泄漏参数衰减，本步没有下游放电。`;
  }
}

function scheduleNextStep() {
  if (!state.playing || !state.run) return;
  const lastIndex = state.run.simulation.steps.length - 1;
  if (state.stepIndex >= lastIndex) {
    stopPlayback();
    return;
  }
  const delay = reducedMotion.matches ? 700 : 1150 / Number(elements.speed.value);
  state.timer = window.setTimeout(() => {
    state.stepIndex += 1;
    renderStep();
    scheduleNextStep();
  }, delay);
}

elements.controls.addEventListener("change", (event) => {
  if (event.target === elements.speed) return;
  requestSimulation();
});

elements.speed.addEventListener("input", () => {
  elements.speedValue.textContent = `${Number(elements.speed.value).toFixed(1).replace(".0", "")}×`;
  if (state.playing) {
    window.clearTimeout(state.timer);
    scheduleNextStep();
  }
});

elements.playButton.addEventListener("click", () => {
  if (!state.run) return;
  if (state.playing) {
    stopPlayback();
    return;
  }
  if (state.stepIndex >= state.run.simulation.steps.length - 1) {
    state.stepIndex = 0;
    renderStep();
  }
  state.playing = true;
  elements.playIcon.textContent = "Ⅱ";
  elements.playLabel.textContent = "暂停";
  scheduleNextStep();
});

elements.resetButton.addEventListener("click", () => {
  stopPlayback();
  state.stepIndex = 0;
  renderStep();
});

async function initialize() {
  setBusy(true, "正在载入…");
  try {
    await loadStatus();
    await requestSimulation();
  } catch (error) {
    setBusy(false, "载入失败");
    elements.verification.classList.add("is-error");
    elements.verificationText.textContent = "本地数据验证失败";
    elements.requestStatus.classList.add("is-error");
    elements.stepNarrative.textContent = error instanceof Error ? error.message : "未知错误";
  }
}

initialize();
