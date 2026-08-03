"use strict";

(() => {
  const state = {
    page: "skills",
    filter: "all",
    apiStatus: "connecting",
    skills: [],
    selectedKey: null,
    run: {
      skillKey: null,
      runId: null,
      status: "idle",
      completedSteps: 0,
    },
  };

  const api = new window.DittobotApiClient();

  function element(id) {
    const found = document.getElementById(id);
    if (!found) throw new Error(`필수 UI 요소가 없습니다: ${id}`);
    return found;
  }

  const dom = {
    banner: element("api-banner"),
    connectionDot: element("connection-dot"),
    connectionLabel: element("connection-label"),
    skillList: element("skill-list"),
    skillEmpty: element("skill-empty"),
    detailTitle: element("detail-title"),
    detailSubtitle: element("detail-subtitle"),
    detailNodes: element("detail-nodes"),
    validateSkill: element("validate-skill"),
    activateSkill: element("activate-skill"),
    runStatus: element("run-status"),
    runTitle: element("run-title"),
    runSteps: element("run-steps"),
    estopChip: element("estop-chip"),
    forceChip: element("force-chip"),
    workspaceChip: element("workspace-chip"),
    estopRun: element("estop-run"),
    abortRun: element("abort-run"),
  };

  function create(tagName, { className = "", text = "" } = {}) {
    const node = document.createElement(tagName);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function errorText(error) {
    return error?.message || "알 수 없는 API 오류가 발생했습니다.";
  }

  function setBanner(message, kind = "info") {
    dom.banner.textContent = message;
    dom.banner.className = `banner${kind === "ok" ? " ok" : kind === "danger" ? " danger" : ""}`;
  }

  function normalizeSkill(row) {
    const graph = row.skill_graph || {};
    const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
    let uiState = "candidate";
    if (row.status === "active") uiState = "active";
    else if (row.validation_status === "passed") uiState = "tested";
    return {
      key: `${row.skill_id}@${row.version}`,
      id: row.skill_id,
      version: row.version,
      uiState,
      validationStatus: row.validation_status,
      status: row.status,
      description: row.description || graph.description || "",
      nodes: nodes.map((node) => ({
        operation: node.operation || node.node_id || "unknown",
        arguments: node.arguments || {},
        status: "idle",
      })),
    };
  }

  function selectedSkill() {
    return state.skills.find((skill) => skill.key === state.selectedKey) || null;
  }

  function runningSkill() {
    return state.skills.find((skill) => skill.key === state.run.skillKey) || null;
  }

  function tagText(uiState) {
    if (uiState === "active") return "운영 중";
    if (uiState === "tested") return "테스트 통과";
    return "미검증";
  }

  function renderConnection() {
    dom.connectionDot.className = "dot";
    if (state.apiStatus === "connected") {
      dom.connectionDot.classList.add("ok");
      dom.connectionLabel.textContent = "FastAPI · SQLite 연결됨";
    } else if (state.apiStatus === "error") {
      dom.connectionDot.classList.add("danger");
      dom.connectionLabel.textContent = "FastAPI 연결 안 됨";
    } else {
      dom.connectionLabel.textContent = "FastAPI 연결 확인 중";
    }
  }

  function showPage(page) {
    state.page = page;
    document.querySelectorAll(".page").forEach((section) => {
      section.hidden = section.id !== `page-${page}`;
    });
    document.querySelectorAll("[data-page]").forEach((button) => {
      if (button.closest("nav")) {
        button.setAttribute("aria-current", button.dataset.page === page ? "page" : "false");
      }
    });
    if (page === "detail") renderDetail();
    if (page === "monitor") renderMonitor();
  }

  function renderRegistry() {
    dom.skillList.replaceChildren();
    const visible = state.skills.filter((skill) => {
      if (state.filter === "all") return true;
      if (state.filter === "active") return skill.uiState === "active";
      return skill.uiState !== "active";
    });
    dom.skillEmpty.hidden = visible.length !== 0;

    visible.forEach((skill) => {
      const card = create("article", { className: "skill-card" });
      const summary = create("div");
      summary.append(create("h2", { text: skill.id }));
      const meta = create("div", { className: "meta" });
      meta.append(document.createTextNode(`v${skill.version} · 노드 ${skill.nodes.length}개`));
      meta.append(create("span", { className: `tag ${skill.uiState}`, text: tagText(skill.uiState) }));
      summary.append(meta);

      const actions = create("div", { className: "actions" });
      const detail = create("button", { className: "button", text: "상세" });
      detail.type = "button";
      detail.addEventListener("click", () => {
        state.selectedKey = skill.key;
        showPage("detail");
      });
      actions.append(detail);
      if (skill.uiState === "active") {
        const run = create("button", { className: "button primary", text: "Mock 실행" });
        run.type = "button";
        run.disabled = state.apiStatus !== "connected";
        run.addEventListener("click", () => startRun(skill.key));
        actions.append(run);
      }
      card.append(summary, actions);
      dom.skillList.append(card);
    });
  }

  function nodeArgumentsText(argumentsValue) {
    const entries = Object.entries(argumentsValue || {});
    if (!entries.length) return "인수 없음";
    return entries.map(([name, value]) => {
      const rendered = value && typeof value === "object" ? JSON.stringify(value) : String(value);
      return `${name}: ${rendered}`;
    }).join(" · ");
  }

  function renderDetail() {
    const skill = selectedSkill();
    dom.detailNodes.replaceChildren();
    dom.validateSkill.disabled = !skill || state.apiStatus !== "connected";
    dom.activateSkill.disabled = !skill || state.apiStatus !== "connected" || skill.uiState === "active";
    if (!skill) {
      dom.detailTitle.textContent = "스킬을 선택하세요";
      dom.detailSubtitle.textContent = "스킬 관리 화면에서 버전을 선택해 주세요.";
      return;
    }

    dom.detailTitle.textContent = skill.id;
    dom.detailSubtitle.textContent = `v${skill.version} · 노드 ${skill.nodes.length}개 · API는 스킬 전체 Mock 회귀 검증을 수행합니다.`;
    skill.nodes.forEach((skillNode) => {
      const node = create("div", { className: `node ${skillNode.status}`.trim() });
      const description = create("div");
      description.append(create("strong", { text: skillNode.operation }));
      description.append(create("p", { text: nodeArgumentsText(skillNode.arguments) }));
      const label = skillNode.status === "running" ? "검증 중" : skillNode.status === "ok" ? "통과" : skillNode.status === "error" ? "실패" : "대기";
      node.append(description, create("span", { className: "node-state", text: label }));
      dom.detailNodes.append(node);
    });
  }

  function renderMonitor() {
    const skill = runningSkill();
    const labels = {
      idle: "대기",
      starting: "준비 중",
      running: "실행 중",
      aborting: "중단 중",
      succeeded: "완료",
      failed: "실패",
      aborted: "중단됨",
      estop: "안전정지됨",
    };
    dom.runStatus.textContent = labels[state.run.status] || state.run.status;
    dom.runTitle.textContent = skill?.id || "실행 중인 스킬 없음";
    dom.runSteps.replaceChildren();

    if (!skill) {
      dom.runSteps.append(create("li", { text: "활성 스킬의 실행 버튼을 눌러주세요." }));
    } else {
      skill.nodes.forEach((node, index) => {
        let className = "";
        let prefix = "○";
        if (index < state.run.completedSteps) {
          className = "done";
          prefix = "✓";
        } else if (index === state.run.completedSteps && ["starting", "running"].includes(state.run.status)) {
          className = "current";
          prefix = "●";
        }
        dom.runSteps.append(create("li", { className, text: `${prefix} ${node.operation}` }));
      });
    }

    const supervising = ["starting", "running", "aborting"].includes(state.run.status);
    dom.forceChip.textContent = `${supervising ? "●" : "○"} FORCE SUPERVISOR ${supervising ? "ACTIVE" : "IDLE"}`;
    dom.workspaceChip.textContent = `${supervising ? "●" : "○"} WORKSPACE MONITOR ${supervising ? "ACTIVE" : "IDLE"}`;
    dom.estopChip.textContent = state.run.status === "estop" ? "● E-STOP REQUESTED" : "● E-STOP READY";
    dom.estopChip.style.color = state.run.status === "estop" ? "var(--danger)" : "";
    const canAbort = Boolean(state.run.runId) && ["starting", "running", "aborting"].includes(state.run.status);
    dom.abortRun.disabled = !canAbort;
    dom.estopRun.disabled = !canAbort;
  }

  function renderAll() {
    renderConnection();
    renderRegistry();
    if (state.page === "detail") renderDetail();
    if (state.page === "monitor") renderMonitor();
  }

  async function loadRegistry(message = null) {
    state.apiStatus = "connecting";
    renderConnection();
    setBanner("FastAPI 연결 확인 중…");
    try {
      const [, registry] = await Promise.all([api.health(), api.listSkills()]);
      state.skills = (registry.skills || []).map(normalizeSkill);
      state.selectedKey = state.skills.some((skill) => skill.key === state.selectedKey)
        ? state.selectedKey
        : state.skills[0]?.key || null;
      state.apiStatus = "connected";
      setBanner(message || `FastAPI · SQLite 연결됨 · ${state.skills.length}개 버전`, "ok");
    } catch (error) {
      state.apiStatus = "error";
      state.skills = [];
      state.selectedKey = null;
      setBanner(`API 연결 실패: ${errorText(error)}`, "danger");
    }
    renderAll();
  }

  async function validateSelected() {
    const skill = selectedSkill();
    if (!skill || state.apiStatus !== "connected") return false;
    skill.nodes.forEach((node) => { node.status = "running"; });
    setBanner(`${skill.id} 전체 Mock 회귀 검증 중…`);
    renderDetail();
    try {
      const result = await api.validateSkill(skill.id, skill.version);
      skill.nodes.forEach((node) => { node.status = result.passed ? "ok" : "error"; });
      if (result.passed) {
        skill.validationStatus = "passed";
        if (skill.uiState !== "active") skill.uiState = "tested";
        setBanner(`${skill.id} 검증 통과`, "ok");
      } else {
        setBanner(`${skill.id} 검증 실패: ${(result.errors || []).join(", ")}`, "danger");
      }
      renderAll();
      return Boolean(result.passed);
    } catch (error) {
      skill.nodes.forEach((node) => { node.status = "error"; });
      setBanner(`검증 요청 실패: ${errorText(error)}`, "danger");
      renderDetail();
      return false;
    }
  }

  async function activateSelected() {
    const skill = selectedSkill();
    if (!skill || skill.uiState === "active") return;
    const validated = skill.validationStatus === "passed" || await validateSelected();
    if (!validated) return;
    try {
      setBanner(`${skill.id} 버전 활성화 중…`);
      await api.activateSkill(skill.id, skill.version);
      await loadRegistry(`${skill.id} 활성화 완료`);
      showPage("skills");
    } catch (error) {
      setBanner(`활성화 실패: ${errorText(error)}`, "danger");
    }
  }

  function newRunId() {
    const suffix = window.crypto?.randomUUID
      ? window.crypto.randomUUID().replaceAll("-", "").slice(0, 12)
      : Math.random().toString(36).slice(2, 14);
    return `ui-${Date.now().toString(36)}-${suffix}`;
  }

  async function startRun(skillKey) {
    const skill = state.skills.find((item) => item.key === skillKey);
    if (!skill || skill.uiState !== "active" || state.apiStatus !== "connected") return;
    const runId = newRunId();
    state.run = { skillKey, runId, status: "starting", completedSteps: 0 };
    setBanner(`${skill.id} Scene 캡처 및 preflight 준비 중…`);
    showPage("monitor");
    renderMonitor();
    try {
      const scene = await api.captureScene();
      const binding = await api.bindRuntime(skill.id, skill.version, scene.scene_id);
      const preflight = await api.preflightRuntime(skill.id, skill.version, scene.scene_id, binding.bindings);
      if (!preflight.passed) {
        throw new Error(`preflight 거부: ${(preflight.errors || []).join(", ")}`);
      }
      state.run.status = "running";
      setBanner(`${skill.id} Mock 실행 중…`);
      renderMonitor();
      const result = await api.executeRuntime({
        skillId: skill.id,
        version: skill.version,
        sceneId: scene.scene_id,
        bindings: binding.bindings,
        runId,
      });
      state.run.runId = result.run_id;
      state.run.status = "succeeded";
      state.run.completedSteps = skill.nodes.length;
      setBanner(`${skill.id} 실행 ${result.status}`, "ok");
    } catch (error) {
      state.run.status = "failed";
      setBanner(`실행 실패: ${errorText(error)}`, "danger");
    }
    renderMonitor();
  }

  async function abortRun(reason) {
    if (!state.run.runId) return;
    state.run.status = "aborting";
    setBanner("중단 요청 전송 중…");
    renderMonitor();
    try {
      await api.abortRuntime(state.run.runId, reason);
      state.run.status = reason === "emergency_stop" ? "estop" : "aborted";
      setBanner(reason === "emergency_stop" ? "안전정지 요청 완료" : "실행 중단 요청 완료");
    } catch (error) {
      state.run.status = "failed";
      setBanner(`중단 요청 실패: ${errorText(error)}`, "danger");
    }
    renderMonitor();
  }

  document.querySelectorAll("[data-page]").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.page));
  });
  document.querySelectorAll("[data-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach((item) => {
        item.setAttribute("aria-pressed", String(item === button));
      });
      renderRegistry();
    });
  });
  dom.validateSkill.addEventListener("click", validateSelected);
  dom.activateSkill.addEventListener("click", activateSelected);
  dom.estopRun.addEventListener("click", () => abortRun("emergency_stop"));
  dom.abortRun.addEventListener("click", () => abortRun("operator_request"));

  renderAll();
  loadRegistry();
})();
