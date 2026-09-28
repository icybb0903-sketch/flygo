"use strict";

const API_URL = "/api/agents";
const VISIBLE_POLL_MS = 1000;
const HIDDEN_POLL_MS = 5000;
const REQUEST_TIMEOUT_MS = 8000;

const STATUS_LABELS = Object.freeze({
  waiting: "等待中",
  working: "工作中",
  testing: "测试中",
  completed: "已完成",
  failed: "失败",
  interrupted: "已中断",
});

const PRESENCE_LABELS = Object.freeze({
  present: "已实际观察到在线",
  not_present: "已实际观察到不在线",
  unavailable: "在线状态不可获取",
});

const GROUP_DEFINITIONS = Object.freeze([
  {
    key: "current",
    title: "当前真实运行",
    description: "只有最新 presence 观察为 present 且尚未结束的 Agent 才会出现在这里。",
    empty: "当前没有被实际观察到正在运行的 Agent。",
  },
  {
    key: "historical",
    title: "已完成或历史记录",
    description: "已完成，或最新观察确认已不在线；旧的“工作中”只是当时状态。",
    empty: "暂无历史 Agent 记录。",
  },
  {
    key: "failed_interrupted",
    title: "失败或中断",
    description: "明确记录为失败或中断的 Agent，保留错误和完整时间线。",
    empty: "暂无失败或中断记录。",
  },
  {
    key: "unavailable",
    title: "状态不可获取",
    description: "缺少足够的真实在线观察，页面不会猜测它们现在是否运行。",
    empty: "暂无状态不可获取的记录。",
  },
]);

const TEST_STATUS_LABELS = Object.freeze({
  running: "测试运行中",
  passed: "测试通过",
  failed: "测试失败",
});

const EVENT_TYPE_LABELS = Object.freeze({
  agent_created: "创建 Agent",
  task_assigned: "接收任务",
  status_changed: "状态变化",
  activity: "处理活动",
  files_changed: "修改文件",
  test_started: "开始测试",
  test_finished: "测试结果",
  agent_completed: "Agent 完成",
  agent_failed: "Agent 失败",
  agent_interrupted: "Agent 中断",
  runtime_observed: "运行状态观察",
  integration_started: "开始整合",
  integration_finished: "整合完成",
});

const elements = {
  groups: document.querySelector("#agentGroups"),
  empty: document.querySelector("#emptyState"),
  count: document.querySelector("#agentCount"),
  connectionDot: document.querySelector("#connectionDot"),
  connectionText: document.querySelector("#connectionText"),
  lastSync: document.querySelector("#lastSync"),
  sourceNote: document.querySelector("#sourceNote"),
  offlineBanner: document.querySelector("#offlineBanner"),
  offlineDetail: document.querySelector("#offlineDetail"),
  schemaVersion: document.querySelector("#schemaVersion"),
  liveStatus: document.querySelector("#liveStatus"),
};

let pollTimer = null;
let activeController = null;
let lastSuccessfulPayload = null;
let hasConnected = false;
let expandedAgents = new Set();
let pageIsActive = true;

function unavailable(value) {
  return value === null || value === undefined ? "不可获取" : String(value);
}

function displayValue(value) {
  if (value === null || value === undefined) return "不可获取";
  if (value === "unavailable") return "不可获取";
  if (Array.isArray(value)) {
    if (value.length === 0) return "暂无记录";
    return value.map((item) => displayValue(item)).join("、");
  }
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch (_error) {
      return String(value);
    }
  }
  if (value === "") return "（空字符串）";
  return String(value);
}

function displayTime(value) {
  if (value === null || value === undefined) return "不可获取";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function displayTest(value) {
  if (value === null || value === undefined) return "不可获取";
  if (typeof value !== "object" || Array.isArray(value)) return displayValue(value);
  const status = value.status === null || value.status === undefined
    ? "不可获取"
    : (TEST_STATUS_LABELS[value.status] || String(value.status));
  const parts = [status];
  if (value.summary !== null && value.summary !== undefined) parts.push(String(value.summary));
  if (value.command !== null && value.command !== undefined) parts.push(`命令：${value.command}`);
  return parts.join(" · ");
}

function makeElement(tagName, className, text) {
  const node = document.createElement(tagName);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function appendField(container, label, value, className) {
  const group = makeElement("div", className ? `field ${className}` : "field");
  group.append(makeElement("dt", null, label));
  group.append(makeElement("dd", null, displayValue(value)));
  container.append(group);
}

function statusLabel(status) {
  if (status === null || status === undefined) return "不可获取";
  if (status === "unavailable") return "不可获取";
  return STATUS_LABELS[status] || String(status);
}

function normalizedStatus(status) {
  return Object.prototype.hasOwnProperty.call(STATUS_LABELS, status) ? status : "unknown";
}

function normalizedGroup(group) {
  if (["current", "historical", "failed", "interrupted", "unavailable"].includes(group)) {
    return group;
  }
  return "unavailable";
}

function groupKeyFor(agent) {
  const group = normalizedGroup(agent.record_group);
  return ["failed", "interrupted"].includes(group) ? "failed_interrupted" : group;
}

function createStatus(agent) {
  const group = normalizedGroup(agent.record_group);
  let normalized = normalizedStatus(agent.status);
  let label = statusLabel(agent.status);

  if (group === "historical") {
    normalized = "historical";
    label = "历史记录";
  } else if (group === "unavailable") {
    normalized = "unavailable";
    label = "状态不可获取";
  } else if (group === "interrupted") {
    normalized = "interrupted";
    label = "已中断";
  }

  const badge = makeElement("span", `status status-${normalized}`);
  const dot = makeElement("i", null);
  dot.setAttribute("aria-hidden", "true");
  badge.append(dot, document.createTextNode(label));
  return badge;
}

function presenceLabel(value) {
  if (value === null || value === undefined) return "在线状态不可获取";
  return PRESENCE_LABELS[value] || `在线状态：${String(value)}`;
}

function eventLabel(key) {
  const labels = {
    at: "时间",
    timestamp: "时间",
    time: "时间",
    type: "类型",
    event: "事件",
    action: "动作",
    message: "说明",
    detail: "详情",
    source: "来源",
    files: "文件",
    result: "结果",
    event_type: "事件",
    agent_id: "Agent ID",
    agent_name: "Agent 名称",
    parent_id: "父 Agent",
    task: "任务",
    status: "状态",
    test: "测试",
    error: "错误",
    timestamp_utc: "时间",
    event_id: "事件 ID",
    schema_version: "Schema",
  };
  return labels[key] || key;
}

function createTimelineEvent(event, index) {
  const item = makeElement("li", "timeline-event");
  const marker = makeElement("span", "event-marker", String(index + 1));
  marker.setAttribute("aria-hidden", "true");
  const content = makeElement("div", "event-content");

  if (event === null || event === undefined || typeof event !== "object" || Array.isArray(event)) {
    content.append(makeElement("p", "event-value", displayValue(event)));
    item.append(marker, content);
    return item;
  }

  const entries = Object.entries(event);
  if (entries.length === 0) {
    content.append(makeElement("p", "event-value", "暂无记录"));
    item.append(marker, content);
    return item;
  }

  const sourceEntry = entries.find(([key]) => key === "source");
  if (sourceEntry) {
    const source = makeElement("span", "source-tag", `source · ${displayValue(sourceEntry[1])}`);
    content.append(source);
  }

  const fields = makeElement("dl", "event-fields");
  entries.forEach(([key, value]) => {
    const row = makeElement("div", "event-field");
    row.append(makeElement("dt", null, eventLabel(key)));
    let rendered;
    if (["at", "timestamp", "time", "timestamp_utc"].includes(key)) {
      rendered = displayTime(value);
    } else if (key === "test") {
      rendered = displayTest(value);
    } else if (key === "event_type" && value !== null && value !== undefined) {
      rendered = EVENT_TYPE_LABELS[value] || String(value);
    } else if (key === "status") {
      rendered = statusLabel(value);
    } else {
      rendered = displayValue(value);
    }
    row.append(makeElement("dd", null, rendered));
    fields.append(row);
  });
  content.append(fields);
  item.append(marker, content);
  return item;
}

function createTimeline(agent, agentKey) {
  const details = makeElement("details", "timeline-details");
  details.open = expandedAgents.has(agentKey);
  details.addEventListener("toggle", () => {
    if (details.open) expandedAgents.add(agentKey);
    else expandedAgents.delete(agentKey);
  });

  const timeline = Array.isArray(agent.timeline) ? agent.timeline : null;
  const count = timeline === null ? "不可获取" : String(timeline.length);
  const summary = makeElement("summary", null);
  summary.append(
    document.createTextNode("执行时间线"),
    makeElement("span", "timeline-count", count),
  );
  details.append(summary);

  if (timeline === null) {
    details.append(makeElement("p", "timeline-empty", "不可获取"));
  } else if (timeline.length === 0) {
    details.append(makeElement("p", "timeline-empty", "暂无记录"));
  } else {
    const list = makeElement("ol", "timeline-list");
    timeline.forEach((event, index) => list.append(createTimelineEvent(event, index)));
    details.append(list);
  }
  return details;
}

function createAgentCard(agent, agentKey) {
  const article = makeElement("article", "agent-card");
  const status = normalizedStatus(agent.status);
  const group = normalizedGroup(agent.record_group);
  article.dataset.status = status;
  article.dataset.group = group;

  const header = makeElement("header", "agent-header");
  const identity = makeElement("div", "agent-identity");
  const presenceClass = ["present", "not_present"].includes(agent.runtime_presence)
    ? agent.runtime_presence
    : "unavailable";
  identity.append(
    makeElement("p", "agent-id", displayValue(agent.agent_id)),
    makeElement("h3", null, displayValue(agent.name)),
    makeElement("p", `presence presence-${presenceClass}`, presenceLabel(agent.runtime_presence)),
  );
  header.append(identity, createStatus(agent));

  const task = makeElement("div", "task-block");
  task.append(makeElement("span", null, "任务"), makeElement("p", null, displayValue(agent.task)));

  const fields = makeElement("dl", "agent-fields");
  if (group !== "current") {
    appendField(fields, "最后记录状态（非当前在线证明）", statusLabel(agent.status), "field-wide");
  }
  appendField(fields, "当前", agent.current, "field-wide");
  appendField(fields, "模型", agent.model_name);
  appendField(fields, "推理强度", agent.reasoning_effort);
  appendField(fields, "在线观察", presenceLabel(agent.runtime_presence));
  appendField(fields, "观察时间", displayTime(agent.observed_at));
  appendField(fields, "文件", agent.files, "field-wide field-files");
  appendField(fields, "最近测试", displayTest(agent.last_test), "field-wide");
  appendField(fields, "开始时间", displayTime(agent.started_at));
  appendField(fields, "完成时间", displayTime(agent.completed_at));
  appendField(fields, "错误", agent.error, "field-wide field-error");

  article.append(header, task, fields, createTimeline(agent, agentKey));
  return article;
}

function buildAgentTree(agents) {
  const forest = document.createDocumentFragment();
  const records = agents.map((agent, index) => ({ agent, index, children: [] }));
  const byId = new Map();

  records.forEach((record) => {
    const id = record.agent.agent_id;
    if (id !== null && id !== undefined && !byId.has(String(id))) {
      byId.set(String(id), record);
    }
  });

  const roots = [];
  records.forEach((record) => {
    const parentId = record.agent.parent_id;
    const parent = parentId === null || parentId === undefined ? null : byId.get(String(parentId));
    if (parent && parent !== record) parent.children.push(record);
    else roots.push(record);
  });

  const rendered = new Set();

  function renderNode(record, ancestry) {
    const node = makeElement("div", "agent-node");
    const id = record.agent.agent_id === null || record.agent.agent_id === undefined
      ? `__index_${record.index}`
      : String(record.agent.agent_id);
    rendered.add(record.index);
    node.append(createAgentCard(record.agent, id));

    if (record.children.length > 0) {
      const children = makeElement("div", "agent-children");
      children.setAttribute("aria-label", `${displayValue(record.agent.name)} 的子 Agent`);
      const nextAncestry = new Set(ancestry);
      nextAncestry.add(id);
      record.children.forEach((child) => {
        const childId = child.agent.agent_id === null || child.agent.agent_id === undefined
          ? `__index_${child.index}`
          : String(child.agent.agent_id);
        if (!nextAncestry.has(childId) && !rendered.has(child.index)) {
          children.append(renderNode(child, nextAncestry));
        }
      });
      if (children.childElementCount > 0) node.append(children);
    }
    return node;
  }

  roots.forEach((record) => {
    if (!rendered.has(record.index)) forest.append(renderNode(record, new Set()));
  });
  records.forEach((record) => {
    if (!rendered.has(record.index)) forest.append(renderNode(record, new Set()));
  });
  return forest;
}

function createAgentGroup(definition, agents) {
  const section = makeElement("section", "agent-group");
  section.dataset.group = definition.key;
  section.setAttribute("aria-labelledby", `group-${definition.key}-title`);

  const header = makeElement("header", "agent-group-header");
  const heading = makeElement("div", "agent-group-heading");
  const title = makeElement("h3", null, definition.title);
  title.id = `group-${definition.key}-title`;
  heading.append(title, makeElement("p", null, definition.description));
  header.append(heading, makeElement("span", "group-count", String(agents.length)));

  const body = makeElement("div", "agent-group-body");
  if (agents.length === 0) {
    body.append(makeElement("p", "group-empty", definition.empty));
  } else {
    body.append(buildAgentTree(agents));
  }

  section.append(header, body);
  return section;
}

function validatePayload(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("响应不是 JSON 对象");
  }
  if (payload.ok !== true) throw new Error("服务端返回 ok=false");
  if (!Array.isArray(payload.agents)) throw new Error("agents 字段不是数组");
  return payload;
}

function setConnection(mode, message) {
  elements.connectionDot.className = `connection-dot is-${mode}`;
  elements.connectionText.textContent = message;
}

function renderPayload(payload) {
  const agents = payload.agents;
  elements.sourceNote.textContent = unavailable(payload.source_note);
  elements.schemaVersion.textContent = `Schema：${unavailable(payload.schema_version)}`;
  elements.lastSync.textContent = displayTime(payload.generated_at_utc);
  const grouped = new Map(GROUP_DEFINITIONS.map((definition) => [definition.key, []]));
  agents.forEach((agent) => grouped.get(groupKeyFor(agent)).push(agent));
  const currentCount = grouped.get("current").length;
  elements.count.textContent = `当前 ${currentCount} · 全部 ${agents.length}`;
  elements.groups.replaceChildren();

  if (agents.length === 0) {
    elements.empty.hidden = false;
  } else {
    elements.empty.hidden = true;
    GROUP_DEFINITIONS.forEach((definition) => {
      elements.groups.append(createAgentGroup(definition, grouped.get(definition.key)));
    });
  }
  elements.groups.setAttribute("aria-busy", "false");
}

function markSuccess(payload) {
  lastSuccessfulPayload = payload;
  renderPayload(payload);
  setConnection("online", "已连接");
  elements.offlineBanner.hidden = true;
  elements.liveStatus.textContent = hasConnected
    ? `同步成功，共 ${payload.agents.length} 个 Agent`
    : `已连接，共 ${payload.agents.length} 个 Agent`;
  hasConnected = true;
}

function markFailure(error) {
  setConnection("offline", "连接中断");
  elements.offlineBanner.hidden = false;
  elements.offlineDetail.textContent = lastSuccessfulPayload
    ? `正在重试；仍显示最后一次成功数据。原因：${error.message}`
    : `正在重试；尚无可保留的数据。原因：${error.message}`;
  elements.groups.setAttribute("aria-busy", "false");
  if (!lastSuccessfulPayload) {
    elements.count.textContent = "数据不可获取";
    elements.sourceNote.textContent = "不可获取";
  }
  elements.liveStatus.textContent = `Agent 数据连接中断：${error.message}`;
}

async function fetchAgents() {
  if (activeController) return;
  const controller = new AbortController();
  activeController = controller;
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(API_URL, {
      method: "GET",
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = validatePayload(await response.json());
    markSuccess(payload);
  } catch (error) {
    if (pageIsActive) {
      if (controller.signal.aborted) markFailure(new Error("请求超时"));
      else markFailure(error instanceof Error ? error : new Error("未知连接错误"));
    }
  } finally {
    window.clearTimeout(timeoutId);
    if (activeController === controller) activeController = null;
    scheduleNextPoll();
  }
}

function scheduleNextPoll() {
  window.clearTimeout(pollTimer);
  if (!pageIsActive) return;
  const delay = document.hidden ? HIDDEN_POLL_MS : VISIBLE_POLL_MS;
  pollTimer = window.setTimeout(fetchAgents, delay);
}

document.addEventListener("visibilitychange", () => {
  window.clearTimeout(pollTimer);
  if (!document.hidden && activeController === null) fetchAgents();
  else if (activeController === null) scheduleNextPoll();
});

window.addEventListener("pagehide", () => {
  pageIsActive = false;
  window.clearTimeout(pollTimer);
  if (activeController) activeController.abort();
});

window.addEventListener("pageshow", (event) => {
  if (event.persisted) {
    pageIsActive = true;
    fetchAgents();
  }
});

fetchAgents();
