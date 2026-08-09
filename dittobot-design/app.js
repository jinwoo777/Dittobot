"use strict";

(() => {
  const state = {
    page: "skills",
    filter: "all",
    apiStatus: "connecting",
    skills: [],
    drafts: [],
    taskFlowCatalog: { objects: [] },
    selectedKey: null,
    selectedDraftId: null,
    run: {
      skillKey: null,
      runId: null,
      status: "idle",
      completedSteps: 0,
    },
    camera: {
      state: "stopped",
      recording: null,
      recordingId: null,
      lastRecording: null,
      lastError: null,
      frameNumber: null,
      maximumTimestampSkewMs: null,
    },
    calibration: {
      capabilities: null,
      session: null,
      latestResult: null,
      legacyTransform: null,
      loading: false,
      lastError: null,
    },
    jog: {
      status: null,
      loading: false,
      lastError: null,
      targetPositionsDeg: null,
      targetDirty: false,
    },
    arucoExperiment: {
      status: null,
      loading: false,
      lastError: null,
    },
    recordingReview: {
      recordings: [],
      selectedId: null,
      frameIndex: 0,
      playing: false,
      playbackTimer: null,
      capabilities: null,
      loading: false,
      loaded: false,
      draft: null,
      fingerTrackingRecordingId: null,
      fingerObservations: [],
      fingerTrackingSettings: null,
      surfaceHint: null,
    },
    geometryTeaching: {
      mode: null,
      draftId: null,
      points: [],
      samples: [],
      busy: false,
    },
    editor: {
      createMode: "recording",
      catalog: [],
      blocks: [],
      bindings: {},
      loading: false,
      workspace: null,
      selectedBlocklyBlockId: null,
      blocklyError: null,
      blocklyCatalogSignature: "",
      parameterNodeId: null,
      parameterArguments: null,
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
    executionMode: element("execution-mode"),
    skillList: element("skill-list"),
    skillEmpty: element("skill-empty"),
    taskFlowCatalog: element("task-flow-catalog"),
    taskFlowList: element("task-flow-list"),
    draftInspector: element("draft-inspector"),
    draftInspectorTitle: element("draft-inspector-title"),
    draftInspectorSummary: element("draft-inspector-summary"),
    draftEvidence: element("draft-evidence"),
    draftPrimitiveList: element("draft-primitive-list"),
    draftAmbiguityList: element("draft-ambiguity-list"),
    draftReadinessList: element("draft-readiness-list"),
    draftNextAction: element("draft-next-action"),
    closeDraftInspector: element("close-draft-inspector"),
    reviewDraftRecording: element("review-draft-recording"),
    autoSurfaceCalibration: element("auto-surface-calibration"),
    startSurfaceCalibration: element("start-surface-calibration"),
    startPathTeaching: element("start-path-teaching"),
    autoTcpPath: element("auto-tcp-path"),
    registerDraftCandidate: element("register-draft-candidate"),
    detailTitle: element("detail-title"),
    detailSubtitle: element("detail-subtitle"),
    detailNodes: element("detail-nodes"),
    validateSkill: element("validate-skill"),
    activateSkill: element("activate-skill"),
    deactivateSkill: element("deactivate-skill"),
    runStatus: element("run-status"),
    runTitle: element("run-title"),
    runSteps: element("run-steps"),
    estopChip: element("estop-chip"),
    cameraChip: element("camera-chip"),
    workspaceChip: element("workspace-chip"),
    modeChip: element("mode-chip"),
    rgbStream: element("rgb-stream"),
    depthStream: element("depth-stream"),
    rgbPlaceholder: element("rgb-placeholder"),
    depthPlaceholder: element("depth-placeholder"),
    startCamera: element("start-camera"),
    stopCamera: element("stop-camera"),
    recordDuration: element("record-duration"),
    startRecording: element("start-recording"),
    stopRecording: element("stop-recording"),
    cameraStatus: element("camera-status"),
    startHandeyeCalibration: element("start-handeye-calibration"),
    abortHandeyeCalibration: element("abort-handeye-calibration"),
    importLegacyHandeye: element("import-legacy-handeye"),
    calibrationProgress: element("calibration-progress"),
    calibrationStatus: element("calibration-status"),
    calibrationResult: element("calibration-result"),
    jogModeNotice: element("jog-mode-notice"),
    jogOperatorId: element("jog-operator-id"),
    jogWorkspaceCleared: element("jog-workspace-cleared"),
    jogEstopReady: element("jog-estop-ready"),
    jogDirectMotionAck: element("jog-direct-motion-ack"),
    jogEnable: element("jog-enable"),
    jogRefresh: element("jog-refresh"),
    jogStop: element("jog-stop"),
    jogStatus: element("jog-status"),
    jogGates: element("jog-gates"),
    jogStepDeg: element("jog-step-deg"),
    jogJointList: element("jog-joint-list"),
    jogLoadCurrent: element("jog-load-current"),
    jogMoveJ: element("jog-movej"),
    arucoExperimentModeNotice: element("aruco-experiment-mode-notice"),
    arucoExperimentOperatorId: element("aruco-experiment-operator-id"),
    arucoObjectWidthMm: element("aruco-object-width-mm"),
    arucoWidthModel: element("aruco-width-model"),
    arucoWorkspaceCleared: element("aruco-workspace-cleared"),
    arucoEstopReady: element("aruco-estop-ready"),
    arucoDirectMotionAck: element("aruco-direct-motion-ack"),
    arucoExperimentEnable: element("aruco-experiment-enable"),
    arucoExperimentRefresh: element("aruco-experiment-refresh"),
    arucoExperimentStop: element("aruco-experiment-stop"),
    arucoExperimentStatus: element("aruco-experiment-status"),
    arucoExperimentGates: element("aruco-experiment-gates"),
    arucoMoveReference: element("aruco-move-reference"),
    arucoMoveZTest: element("aruco-move-z-test"),
    arucoWorkspaceSummary: element("aruco-workspace-summary"),
    recordingSelect: element("recording-select"),
    refreshRecordings: element("refresh-recordings"),
    recordedRgbFrame: element("recorded-rgb-frame"),
    recordedRgbView: element("recorded-rgb-view"),
    recordedPointOverlay: element("recorded-point-overlay"),
    recordedDepthFrame: element("recorded-depth-frame"),
    recordedRgbPlaceholder: element("recorded-rgb-placeholder"),
    recordedDepthPlaceholder: element("recorded-depth-placeholder"),
    toggleRecordingPlayback: element("toggle-recording-playback"),
    recordingFrameIndex: element("recording-frame-index"),
    recordingFrameLabel: element("recording-frame-label"),
    recordingReviewStatus: element("recording-review-status"),
    geometryTeaching: element("geometry-teaching"),
    geometryTeachingTitle: element("geometry-teaching-title"),
    geometryTeachingInstruction: element("geometry-teaching-instruction"),
    surfaceAnchorLabel: element("surface-anchor-label"),
    surfaceAnchorId: element("surface-anchor-id"),
    geometryPointSummary: element("geometry-point-summary"),
    resetGeometryPoints: element("reset-geometry-points"),
    addPathSample: element("add-path-sample"),
    saveGeometryEvidence: element("save-geometry-evidence"),
    cancelGeometryTeaching: element("cancel-geometry-teaching"),
    recordingSkillForm: element("recording-skill-form"),
    openaiCapability: element("openai-capability"),
    draftSkillName: element("draft-skill-name"),
    draftInstruction: element("draft-instruction"),
    draftKeyframeCount: element("draft-keyframe-count"),
    analyzeRecording: element("analyze-recording"),
    draftStatus: element("draft-status"),
    draftResult: element("draft-result"),
    createRecordingTab: element("create-recording-tab"),
    createBlockTab: element("create-block-tab"),
    recordingCreateMode: element("recording-create-mode"),
    blockCreateMode: element("block-create-mode"),
    blockSkillForm: element("block-skill-form"),
    blockSkillId: element("block-skill-id"),
    blockSkillName: element("block-skill-name"),
    blockSkillDescription: element("block-skill-description"),
    blockSkillType: element("block-skill-type"),
    blockOperationSelect: element("block-operation-select"),
    addSkillBlock: element("add-skill-block"),
    previewSkillBlocks: element("preview-skill-blocks"),
    createBlockCandidate: element("create-block-candidate"),
    skillBlockList: element("skill-block-list"),
    blocklyDiv: element("blocklyDiv"),
    blocklyStatus: element("blockly-status"),
    blocklyParameterEditor: element("blockly-parameter-editor"),
    blocklyParameterTitle: element("blockly-parameter-title"),
    blocklyParameterFields: element("blockly-parameter-fields"),
    blockBindingList: element("block-binding-list"),
    blockEditorResult: element("block-editor-result"),
    parameterEditor: element("parameter-editor"),
    parameterNodeSelect: element("parameter-node-select"),
    parameterFields: element("parameter-fields"),
    createParameterCandidate: element("create-parameter-candidate"),
    parameterEditorResult: element("parameter-editor-result"),
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
      checksum: row.graph_checksum_sha256 || "",
      description: row.description || graph.description || "",
      graph,
      nodes: nodes.map((node) => ({
        nodeId: node.node_id || "unknown",
        operation: node.operation || node.node_id || "unknown",
        arguments: node.arguments || {},
        status: "idle",
      })),
    };
  }

  function normalizeDraft(row) {
    const draft = row.draft || {};
    const transport = row.transport || {};
    const readiness = row.promotion_readiness || {};
    const tcp = draft.tcp_proxy_observation || null;
    const sceneObservation = draft.scene_observation || null;
    return {
      draftId: row.draft_id,
      sourceRecordingId: row.source_recording_id,
      suggestedSkillId: draft.suggested_skill_id || row.draft_id,
      displayName: draft.display_name || draft.suggested_skill_id || "분석 초안",
      taskDescription: draft.task_description || "",
      observedSummary: draft.observed_task_summary || "",
      confidence: Number(draft.confidence || 0),
      primitiveSequence: Array.isArray(draft.primitive_sequence) ? draft.primitive_sequence : [],
      ambiguities: Array.isArray(draft.unresolved_ambiguities) ? draft.unresolved_ambiguities : [],
      keyframeCount: Number(row.keyframe_count || 0),
      traceFrameCount: Number(transport.trace_frame_count || 0),
      transport,
      tcp,
      sceneObservation,
      readiness,
      promotionEvidence: row.promotion_evidence || {},
      createdAtNs: Number(row.created_at_ns || 0),
    };
  }

  function selectedSkill() {
    return state.skills.find((skill) => skill.key === state.selectedKey) || null;
  }

  function selectedDraft() {
    return state.drafts.find((draft) => draft.draftId === state.selectedDraftId) || null;
  }

  function runningSkill() {
    return state.skills.find((skill) => skill.key === state.run.skillKey) || null;
  }

  function selectedRecording() {
    return state.recordingReview.recordings.find(
      (recording) => recording.recording_id === state.recordingReview.selectedId,
    ) || null;
  }

  function tagText(uiState) {
    if (uiState === "draft") return "분석 초안";
    if (uiState === "active") return "운영 중";
    if (uiState === "tested") return "테스트 통과";
    return "미검증";
  }

  function renderConnection() {
    const hardwareJog = state.jog.status?.capabilities?.mode === "hardware";
    const hardwareAruco = state.arucoExperiment.status?.capabilities?.mode === "hardware";
    const hardware = hardwareJog || hardwareAruco;
    dom.executionMode.textContent = hardwareAruco
      ? "HARDWARE ARUCO"
      : hardwareJog ? "HARDWARE JOG" : "MOCK API";
    dom.executionMode.style.color = hardware ? "var(--danger)" : "";
    dom.executionMode.style.background = hardware ? "var(--danger-soft)" : "";
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
    if (page !== "create") stopRecordingPlayback();
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
    if (page === "jog") {
      renderJog();
      refreshJogStatus({ quiet: true });
    }
    if (page === "aruco-experiment") {
      renderArucoExperiment();
      refreshArucoExperimentStatus({ quiet: true });
    }
    if (page === "create") {
      renderCreateMode();
      if (state.editor.createMode === "recording") {
        renderRecordingReview();
        if (!state.recordingReview.loaded) loadRecordings();
      }
    }
  }

  async function deleteInactiveSkill(skill, button) {
    const versions = state.skills.filter((item) => item.id === skill.id);
    if (versions.some((item) => item.uiState === "active")) {
      setBanner("활성 스킬은 삭제할 수 없습니다.", "danger");
      return;
    }
    const confirmed = window.confirm(
      `${skill.id}의 후보 버전 ${versions.length}개를 모두 삭제하시겠습니까?\n\n삭제한 레지스트리 데이터는 복구할 수 없습니다.`,
    );
    if (!confirmed) return;
    button.disabled = true;
    setBanner(`${skill.id} 삭제 중…`);
    try {
      const result = await api.deleteSkill(skill.id);
      if (state.selectedKey?.startsWith(`${skill.id}@`)) state.selectedKey = null;
      await loadRegistry(
        `${result.skill_id} 삭제 완료 · 버전 ${result.deleted_versions}개 제거`,
      );
    } catch (error) {
      button.disabled = false;
      setBanner(`스킬 삭제 실패: ${errorText(error)}`, "danger");
    }
  }

  async function deactivateActiveSkill(skill, button) {
    const confirmed = window.confirm(
      `${skill.id} v${skill.version}을 비활성화하시겠습니까?\n\n실행 대상에서 제외되고 retired 상태로 보존됩니다.`,
    );
    if (!confirmed) return;
    button.disabled = true;
    setBanner(`${skill.id} v${skill.version} 비활성화 중…`);
    try {
      const result = await api.deactivateSkill(skill.id);
      if (state.run.skillKey === skill.key) {
        state.run.skillKey = null;
        state.run.runId = null;
        state.run.status = "idle";
      }
      await loadRegistry(`${result.skill_id} v${result.version} 비활성화 완료`);
    } catch (error) {
      button.disabled = false;
      setBanner(`스킬 비활성화 실패: ${errorText(error)}`, "danger");
    }
  }

  function renderRegistry() {
    dom.skillList.replaceChildren();
    const visibleSkills = state.skills.filter((skill) => {
      if (state.filter === "draft") return false;
      if (state.filter === "all") return true;
      if (state.filter === "active") return skill.uiState === "active";
      return skill.uiState !== "active";
    });
    const visibleDrafts = state.filter === "all" || state.filter === "draft"
      ? state.drafts
      : [];
    dom.skillEmpty.hidden = visibleSkills.length + visibleDrafts.length !== 0;
    dom.skillEmpty.textContent = state.filter === "draft"
      ? "저장된 분석 초안이 없습니다. 스킬 만들기에서 RGB-D 녹화를 분석해 주세요."
      : "등록된 스킬 또는 분석 초안이 없습니다.";

    visibleDrafts.forEach((draft) => {
      const card = create("article", { className: "skill-card draft-card" });
      const summary = create("div");
      summary.append(create("h2", { text: draft.suggestedSkillId }));
      const meta = create("div", { className: "meta" });
      meta.append(document.createTextNode(
        `첫 RGB 1장 + 손끝 trace ${draft.traceFrameCount}프레임 · 신뢰도 ${draft.confidence.toFixed(2)}`,
      ));
      meta.append(create("span", { className: "tag draft", text: tagText("draft") }));
      summary.append(meta);
      summary.append(create("p", { text: draft.observedSummary || draft.taskDescription }));

      const actions = create("div", { className: "actions" });
      const detail = create("button", { className: "button", text: "승격 준비도" });
      detail.type = "button";
      detail.addEventListener("click", () => {
        state.selectedDraftId = draft.draftId;
        renderRegistry();
        dom.draftInspector.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
      actions.append(detail);
      card.append(summary, actions);
      dom.skillList.append(card);
    });

    visibleSkills.forEach((skill) => {
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
      const hasActiveVersion = state.skills.some(
        (item) => item.id === skill.id && item.uiState === "active",
      );
      const deleteButton = create("button", {
        className: "button danger",
        text: "삭제",
      });
      deleteButton.type = "button";
      deleteButton.disabled = state.apiStatus !== "connected" || hasActiveVersion;
      deleteButton.title = hasActiveVersion
        ? "활성 버전이 있는 스킬은 삭제할 수 없습니다."
        : "이 스킬 ID의 모든 후보 버전을 삭제합니다.";
      deleteButton.addEventListener("click", () => {
        deleteInactiveSkill(skill, deleteButton);
      });
      if (skill.uiState === "active") {
        const deactivateButton = create("button", {
          className: "button danger",
          text: "비활성화",
        });
        deactivateButton.type = "button";
        deactivateButton.disabled = state.apiStatus !== "connected";
        deactivateButton.addEventListener("click", () => {
          deactivateActiveSkill(skill, deactivateButton);
        });
        actions.append(deactivateButton);
      }
      actions.append(deleteButton);
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
    renderDraftInspector();
  }

  function renderTaskFlowCatalog() {
    const objects = Array.isArray(state.taskFlowCatalog?.objects)
      ? state.taskFlowCatalog.objects
      : [];
    dom.taskFlowCatalog.hidden = objects.length === 0;
    dom.taskFlowList.replaceChildren();
    objects.forEach((objectItem) => {
      const card = create("article", { className: "task-flow-object" });
      const grip = objectItem.active_grip_profile;
      card.append(create("h3", {
        text: `${objectItem.display_name || objectItem.canonical_id} · Grip ${grip ? grip.version : "미활성"}`,
      }));
      const actions = create("ul", { className: "task-flow-actions" });
      (objectItem.actions || []).forEach((action) => {
        const item = create("li");
        const mappedEnd = action.mapped_end_motion?.end_motion_id || "End 미매핑";
        item.append(
          create("span", {
            text: `${action.display_name || action.canonical_id} → ${mappedEnd}`,
          }),
          create("span", {
            className: action.executable ? "ready" : "blocked",
            text: action.executable ? "조합 가능" : `차단 ${action.blockers?.length || 0}`,
          }),
        );
        actions.append(item);
      });
      card.append(actions);
      dom.taskFlowList.append(card);
    });
  }

  function renderDraftInspector() {
    const draft = selectedDraft();
    dom.draftInspector.hidden = !draft;
    if (!draft) return;
    dom.draftInspectorTitle.textContent = draft.suggestedSkillId;
    dom.draftInspectorSummary.textContent = draft.taskDescription || draft.observedSummary;
    dom.draftEvidence.replaceChildren();
    const evidence = [
      {
        label: draft.traceFrameCount > 0
          ? `첫 RGB 1장 + 전체 손끝 trace ${draft.traceFrameCount}프레임`
          : "전체 손끝 trace 재분석 필요",
        passed: draft.traceFrameCount > 0,
      },
      {
        label: draft.tcp?.detected
          ? "첫 RGB의 LLM fingertip 관찰됨 · 상태 판정은 로컬 MediaPipe 우선"
          : "첫 RGB의 LLM fingertip은 advisory 미검출",
        passed: Boolean(draft.tcp?.detected),
      },
      {
        label: draft.sceneObservation?.person_hand
          ? `손 ${draft.sceneObservation.person_hand.shape}` : "손 모양 재분석 필요",
        passed: Boolean(draft.sceneObservation?.person_hand?.detected),
      },
      {
        label: draft.sceneObservation?.tool
          ? `툴 ${draft.sceneObservation.tool.shape}` : "툴 모양 재분석 필요",
        passed: Boolean(draft.sceneObservation?.tool?.detected),
      },
      {
        label: draft.sceneObservation?.work_surface
          ? `작업대 ${draft.sceneObservation.work_surface.shape}` : "작업대 ROI 재분석 필요",
        passed: Boolean(draft.sceneObservation?.work_surface?.detected),
      },
      { label: `semantic confidence ${draft.confidence.toFixed(2)}`, passed: true },
      { label: "실행 geometry는 로컬 task-plane 기준", passed: true },
    ];
    evidence.forEach((item) => {
      dom.draftEvidence.append(create("span", {
        className: `evidence-chip${item.passed ? " ok" : ""}`,
        text: item.label,
      }));
    });
    dom.draftPrimitiveList.replaceChildren();
    const primitives = draft.primitiveSequence.length
      ? draft.primitiveSequence
      : [{ operation: "관찰된 primitive 없음", confidence: 0 }];
    primitives.forEach((primitive) => {
      const confidence = Number(primitive.confidence || 0).toFixed(2);
      dom.draftPrimitiveList.append(create("li", {
        text: `${primitive.operation} · confidence ${confidence}`,
      }));
    });
    dom.draftAmbiguityList.replaceChildren();
    const ambiguities = draft.ambiguities.length
      ? draft.ambiguities
      : ["추가로 보고된 모호성 없음"];
    ambiguities.forEach((ambiguity) => {
      dom.draftAmbiguityList.append(create("li", { text: ambiguity }));
    });

    dom.draftReadinessList.replaceChildren();
    const checks = Array.isArray(draft.readiness.checks) ? draft.readiness.checks : [];
    checks.forEach((check) => {
      const advisory = check.required === false || check.blocking === false;
      const item = create("li", {
        className: `readiness-item${check.passed ? " passed" : ""}${advisory ? " advisory" : ""}`,
      });
      item.append(create("span", {
        className: "readiness-mark",
        text: check.passed ? "✓" : check.pending ? "…" : advisory ? "i" : "!",
      }));
      const description = create("div");
      description.append(create("strong", { text: check.label || check.id }));
      description.append(create("small", { text: check.detail || "" }));
      item.append(description);
      dom.draftReadinessList.append(item);
    });
    dom.draftNextAction.textContent = draft.readiness.next_action
      || "보정된 TF와 pose trajectory가 필요합니다.";
    const checkById = new Map(checks.map((check) => [check.id, check]));
    const calibrationReady = Boolean(checkById.get("operator_task_plane")?.passed);
    const trajectoryReady = Boolean(checkById.get("metric_trajectory")?.passed)
      && Boolean(checkById.get("anchor_geometry")?.passed);
    const candidateReady = Boolean(checkById.get("mock_validation")?.passed);
    dom.autoSurfaceCalibration.disabled = false;
    dom.autoSurfaceCalibration.textContent = calibrationReady
      ? "Depth 평면 보조 힌트 다시 계산" : "Depth 평면 보조 힌트";
    dom.startSurfaceCalibration.disabled = false;
    dom.startSurfaceCalibration.textContent = calibrationReady ? "task-plane TF 다시 보정" : "수동 3점 task-plane TF";
    dom.startPathTeaching.disabled = !calibrationReady;
    dom.startPathTeaching.title = calibrationReady
      ? "녹화 프레임에서 두 fingertip을 직접 지정"
      : "먼저 수동 3점 camera → task-plane TF를 보정하세요.";
    dom.autoTcpPath.disabled = !calibrationReady;
    dom.autoTcpPath.title = calibrationReady
      ? "MediaPipe landmark 4·8을 각자의 aligned depth로 3D 복원"
      : "먼저 수동 3점 camera → task-plane TF를 보정하세요.";
    dom.registerDraftCandidate.disabled = !draft.readiness.can_register_candidate;
    dom.registerDraftCandidate.textContent = candidateReady
      ? "Candidate 등록 완료"
      : trajectoryReady
        ? "Candidate로 등록"
        : "Candidate로 등록";
    dom.registerDraftCandidate.title = draft.readiness.can_register_candidate
      ? "Candidate SkillGraph로 등록"
      : "승격 체크리스트의 필수 증거가 아직 부족합니다.";
  }

  function nodeArgumentsText(argumentsValue) {
    const entries = Object.entries(argumentsValue || {});
    if (!entries.length) return "인수 없음";
    return entries.map(([name, value]) => {
      const rendered = value && typeof value === "object" ? JSON.stringify(value) : String(value);
      return `${name}: ${rendered}`;
    }).join(" · ");
  }

  function primitiveByOperation(operation) {
    return state.editor.catalog.find((item) => item.operation_name === operation) || null;
  }

  function cloneValue(value) {
    return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
  }

  function collectBlockBindingHints() {
    const hints = new Map();
    const kindByAnchorType = {
      object: "object",
      tool: "tool",
      surface: "surface",
      fixture: "surface",
      workspace_region: "workspace",
    };
    const kindByField = {
      tool: "tool",
      surface: "surface",
      region: "workspace",
    };
    const mergeHint = (variable, kind = null) => {
      if (!/^\$[A-Za-z][A-Za-z0-9_]*$/.test(variable)) return;
      if (!hints.has(variable)) {
        hints.set(variable, kind);
        return;
      }
      const previous = hints.get(variable);
      if (previous === "conflict") return;
      if (previous && kind && previous !== kind) hints.set(variable, "conflict");
      else if (!previous && kind) hints.set(variable, kind);
    };
    const visit = (value, fieldName = null) => {
      if (Array.isArray(value)) {
        value.forEach((item) => visit(item, fieldName));
        return;
      }
      if (value && typeof value === "object") {
        const anchorId = value.anchor_id;
        if (typeof anchorId === "string") {
          mergeHint(anchorId, kindByAnchorType[value.anchor_type] || null);
        }
        Object.entries(value).forEach(([key, child]) => visit(child, key));
        return;
      }
      if (typeof value === "string") mergeHint(value, kindByField[fieldName] || null);
    };
    state.editor.blocks.forEach((block) => visit(block.arguments));
    return [...hints.entries()].sort(([left], [right]) => left.localeCompare(right));
  }

  function blockBindingDraft(variable, hint) {
    const conventionalKinds = {
      $tool: "tool",
      $surface: "surface",
      $object: "object",
      $workspace: "workspace",
    };
    const existing = state.editor.bindings[variable] || {};
    const entityKind = existing.entity_kind
      || (hint !== "conflict" ? hint : null)
      || conventionalKinds[variable]
      || "object";
    const draft = {
      variable,
      entity_kind: entityKind,
      instance_id: existing.instance_id || "",
      class_name: existing.class_name || "",
      role: existing.role || "",
      minimum_confidence: Number(existing.minimum_confidence ?? 0.7),
      minimum_visible_fraction: Number(existing.minimum_visible_fraction ?? 0.5),
      must_be_attached: Object.hasOwn(existing, "must_be_attached")
        ? existing.must_be_attached
        : (entityKind === "tool" ? true : null),
    };
    state.editor.bindings[variable] = draft;
    return draft;
  }

  function renderBlockBindings() {
    const hints = collectBlockBindingHints();
    dom.blockBindingList.replaceChildren();
    if (!hints.length) {
      dom.blockBindingList.append(create("p", {
        className: "empty",
        text: "현재 블록에는 $placeholder binding이 없습니다.",
      }));
      return;
    }
    hints.forEach(([variable, hint]) => {
      const binding = blockBindingDraft(variable, hint);
      const card = create("article", { className: "block-card" });
      const head = create("div", { className: "block-card-head" });
      head.append(create("strong", { text: variable }));
      head.append(create("span", {
        className: "tag",
        text: hint === "conflict" ? "역할 충돌" : `추론 ${hint || "없음"}`,
      }));
      const grid = create("div", { className: "binding-grid" });

      const kindLabel = create("label");
      kindLabel.append(create("span", { text: "entity kind" }));
      const kindSelect = create("select");
      ["object", "tool", "surface", "workspace"].forEach((kind) => {
        const option = create("option", { text: kind });
        option.value = kind;
        option.selected = binding.entity_kind === kind;
        kindSelect.append(option);
      });
      kindSelect.addEventListener("change", () => {
        binding.entity_kind = kindSelect.value;
      });
      kindLabel.append(kindSelect);
      grid.append(kindLabel);

      const attachedLabel = create("label");
      attachedLabel.append(create("span", { text: "must be attached" }));
      const attachedSelect = create("select");
      [["", "지정 안 함"], ["true", "true"], ["false", "false"]]
        .forEach(([value, label]) => {
          const option = create("option", { text: label });
          option.value = value;
          option.selected = value === (
            binding.must_be_attached === null ? "" : String(binding.must_be_attached)
          );
          attachedSelect.append(option);
        });
      attachedSelect.addEventListener("change", () => {
        binding.must_be_attached = attachedSelect.value === ""
          ? null
          : attachedSelect.value === "true";
      });
      attachedLabel.append(attachedSelect);
      grid.append(attachedLabel);

      [["instance_id", "instance ID"], ["class_name", "class name"], ["role", "role"]]
        .forEach(([key, label]) => {
          const wrapper = create("label");
          wrapper.append(create("span", { text: label }));
          const input = create("input");
          input.type = "text";
          input.value = binding[key];
          input.addEventListener("change", () => {
            binding[key] = input.value.trim();
          });
          wrapper.append(input);
          grid.append(wrapper);
        });

      [["minimum_confidence", "minimum confidence"], ["minimum_visible_fraction", "minimum visible fraction"]]
        .forEach(([key, label]) => {
          const wrapper = create("label");
          wrapper.append(create("span", { text: label }));
          const input = create("input");
          input.type = "number";
          input.min = "0";
          input.max = "1";
          input.step = "0.05";
          input.value = String(binding[key]);
          input.addEventListener("change", () => {
            binding[key] = Number(input.value);
          });
          wrapper.append(input);
          grid.append(wrapper);
        });
      card.append(head, grid);
      dom.blockBindingList.append(card);
    });
  }

  function serializedBlockBindings() {
    const bindings = {};
    collectBlockBindingHints().forEach(([variable, hint]) => {
      if (hint === "conflict") {
        throw new Error(`${variable}가 서로 다른 entity 역할로 사용되었습니다.`);
      }
      const draft = blockBindingDraft(variable, hint);
      const binding = {
        variable,
        entity_kind: draft.entity_kind,
        minimum_confidence: draft.minimum_confidence,
        minimum_visible_fraction: draft.minimum_visible_fraction,
      };
      ["instance_id", "class_name", "role"].forEach((key) => {
        if (draft[key]) binding[key] = draft[key];
      });
      if (draft.must_be_attached !== null) {
        binding.must_be_attached = draft.must_be_attached;
      }
      bindings[variable] = binding;
    });
    return bindings;
  }

  function resolveEditorSchema(schema, rootSchema) {
    if (schema?.$ref?.startsWith("#/$defs/")) {
      return rootSchema?.$defs?.[schema.$ref.split("/").at(-1)] || schema;
    }
    if (Array.isArray(schema?.anyOf)) {
      const nonNull = schema.anyOf.find((item) => item?.type !== "null");
      return nonNull ? resolveEditorSchema(nonNull, rootSchema) : schema;
    }
    return schema || {};
  }

  function defaultEditorValue(schema, rootSchema, fieldName = "") {
    const resolved = resolveEditorSchema(schema, rootSchema);
    if (Object.hasOwn(resolved, "default")) return cloneValue(resolved.default);
    if (Array.isArray(resolved.enum)) return resolved.enum[0];
    if (resolved.type === "object" || resolved.properties) {
      const result = {};
      (resolved.required || []).forEach((key) => {
        result[key] = defaultEditorValue(resolved.properties[key], rootSchema, key);
      });
      return result;
    }
    if (resolved.type === "array") {
      return Array.from(
        { length: Number(resolved.minItems || 0) },
        () => defaultEditorValue(resolved.items || {}, rootSchema, fieldName),
      );
    }
    if (resolved.type === "boolean") return false;
    if (resolved.type === "integer") return Number(resolved.minimum || 1);
    if (resolved.type === "number") {
      if (fieldName === "w") return 1;
      return Number(resolved.minimum || 0);
    }
    return "";
  }

  function editorEnumOptions(fieldName, schema, primitive) {
    const resolved = resolveEditorSchema(schema, primitive.typed_parameter_schema);
    if (Array.isArray(resolved.enum)) return resolved.enum;
    const profileKey = {
      motion_profile_id: "motion_profile_ids",
      force_profile_id: "force_profile_ids",
      recovery_profile_id: "recovery_profile_ids",
    }[fieldName];
    return profileKey ? primitive.approved_profile_ids?.[profileKey] || [] : [];
  }

  function renderEditorControl({ parent, schema, value, setValue, fieldName, primitive }) {
    const rootSchema = primitive.typed_parameter_schema;
    const resolved = resolveEditorSchema(schema, rootSchema);
    const wrapper = create("label", { className: "schema-field" });
    wrapper.append(create("span", { text: fieldName.replaceAll("_", " ") || "item" }));
    const options = editorEnumOptions(fieldName, schema, primitive);
    if (resolved.type === "object" || resolved.properties) {
      const objectValue = value && typeof value === "object" && !Array.isArray(value) ? value : {};
      const objectContainer = create("div", { className: "schema-object" });
      Object.entries(resolved.properties || {}).forEach(([childName, childSchema]) => {
        renderEditorControl({
          parent: objectContainer,
          schema: childSchema,
          value: objectValue[childName],
          fieldName: childName,
          primitive,
          setValue: (childValue) => {
            const next = { ...objectValue, [childName]: childValue };
            setValue(next);
          },
        });
      });
      wrapper.append(objectContainer);
    } else if (resolved.type === "array") {
      const values = Array.isArray(value) ? value : [];
      const arrayContainer = create("div", { className: "schema-array" });
      values.forEach((item, index) => {
        const itemContainer = create("div", { className: "schema-array-item" });
        renderEditorControl({
          parent: itemContainer,
          schema: resolved.items || {},
          value: item,
          fieldName: `${fieldName} ${index + 1}`,
          primitive,
          setValue: (itemValue) => {
            const next = values.slice();
            next[index] = itemValue;
            setValue(next);
          },
        });
        const remove = create("button", { className: "button", text: "항목 삭제" });
        remove.type = "button";
        remove.disabled = values.length <= Number(resolved.minItems || 0);
        remove.addEventListener("click", () => setValue(values.filter((_, itemIndex) => itemIndex !== index)));
        itemContainer.append(remove);
        arrayContainer.append(itemContainer);
      });
      const add = create("button", { className: "button", text: "항목 추가" });
      add.type = "button";
      add.disabled = values.length >= Number(resolved.maxItems || 128);
      add.addEventListener("click", () => setValue([
        ...values,
        defaultEditorValue(resolved.items || {}, rootSchema, fieldName),
      ]));
      arrayContainer.append(add);
      wrapper.append(arrayContainer);
    } else if (options.length) {
      const select = create("select");
      options.forEach((optionValue) => {
        const option = create("option", { text: String(optionValue) });
        option.value = String(optionValue);
        option.selected = optionValue === value;
        select.append(option);
      });
      select.addEventListener("change", () => setValue(select.value));
      wrapper.append(select);
    } else if (resolved.type === "boolean") {
      const input = create("input");
      input.type = "checkbox";
      input.checked = Boolean(value);
      input.addEventListener("change", () => setValue(input.checked));
      wrapper.append(input);
    } else {
      const input = create("input");
      const numeric = ["number", "integer"].includes(resolved.type);
      input.type = numeric ? "number" : "text";
      if (numeric) {
        if (resolved.minimum !== undefined) input.min = String(resolved.minimum);
        if (resolved.maximum !== undefined) input.max = String(resolved.maximum);
        input.step = resolved.type === "integer" ? "1" : "any";
      }
      input.value = value ?? "";
      input.addEventListener("change", () => {
        setValue(numeric ? Number(input.value) : input.value);
      });
      wrapper.append(input);
    }
    parent.append(wrapper);
  }

  function renderMoveJArgumentEditor(parent, primitive, argumentsValue, onChange) {
    const values = argumentsValue || {};
    const positionsRad = Array.isArray(values.target_joint_positions_rad)
      && values.target_joint_positions_rad.length === 6
      ? values.target_joint_positions_rad.map(Number)
      : Array(6).fill(0);
    const jointGrid = create("div", { className: "joint-angle-grid" });
    positionsRad.forEach((positionRad, index) => {
      const wrapper = create("label", { className: "schema-field" });
      wrapper.append(create("span", { text: `joint${index + 1} (deg)` }));
      const input = create("input");
      input.type = "number";
      input.min = "-360";
      input.max = "360";
      input.step = "0.1";
      input.value = String(Number((positionRad * 180 / Math.PI).toFixed(3)));
      input.addEventListener("change", () => {
        const degrees = Number(input.value);
        if (!Number.isFinite(degrees)) return;
        const nextPositions = positionsRad.slice();
        nextPositions[index] = degrees * Math.PI / 180;
        onChange({
          ...values,
          target_joint_positions_rad: nextPositions,
        });
      });
      wrapper.append(input);
      jointGrid.append(wrapper);
    });
    parent.append(jointGrid);
    parent.append(create("p", {
      className: "notice",
      text: "입력은 degree로 표시되며 SkillGraph에는 radian으로 저장됩니다.",
    }));
    const profileSchema = primitive.typed_parameter_schema.properties?.motion_profile_id;
    if (profileSchema) {
      renderEditorControl({
        parent,
        schema: profileSchema,
        value: values.motion_profile_id,
        fieldName: "motion_profile_id",
        primitive,
        setValue: (profileId) => onChange({ ...values, motion_profile_id: profileId }),
      });
    }
  }

  function renderArgumentEditor(parent, primitive, argumentsValue, onChange) {
    parent.replaceChildren();
    const schema = primitive?.typed_parameter_schema;
    if (!schema) {
      parent.append(create("p", { className: "notice", text: "primitive schema를 찾을 수 없습니다." }));
      return;
    }
    if (primitive.operation_name === "motion.move_j") {
      renderMoveJArgumentEditor(parent, primitive, argumentsValue, onChange);
      return;
    }
    const values = argumentsValue || {};
    Object.entries(schema.properties || {}).forEach(([fieldName, fieldSchema]) => {
      renderEditorControl({
        parent,
        schema: fieldSchema,
        value: values[fieldName],
        fieldName,
        primitive,
        setValue: (fieldValue) => onChange({ ...values, [fieldName]: fieldValue }),
      });
    });
  }

  const HIDDEN_BLOCKLY_CATEGORIES = new Set(["grasp", "gripper", "workspace"]);

  function blocklyToolboxPrimitives() {
    return state.editor.catalog.filter((primitive) => {
      const category = primitive.operation_name.split(".")[0];
      return !HIDDEN_BLOCKLY_CATEGORIES.has(category);
    });
  }

  function renderCreateMode() {
    const blockMode = state.editor.createMode === "block";
    dom.recordingCreateMode.hidden = blockMode;
    dom.blockCreateMode.hidden = !blockMode;
    dom.createRecordingTab.setAttribute("aria-selected", String(!blockMode));
    dom.createRecordingTab.setAttribute("aria-pressed", String(!blockMode));
    dom.createBlockTab.setAttribute("aria-selected", String(blockMode));
    dom.createBlockTab.setAttribute("aria-pressed", String(blockMode));
    if (blockMode) {
      renderSkillBlocks();
      window.requestAnimationFrame(() => {
        if (state.editor.workspace && window.Blockly) {
          window.Blockly.svgResize(state.editor.workspace);
        }
      });
    }
  }

  function renderPrimitiveOptions() {
    const selected = dom.blockOperationSelect.value;
    const toolboxPrimitives = blocklyToolboxPrimitives();
    dom.blockOperationSelect.replaceChildren();
    toolboxPrimitives.forEach((primitive) => {
      const option = create("option", {
        text: `${primitive.operation_name} · ${primitive.description}`,
      });
      option.value = primitive.operation_name;
      option.selected = primitive.operation_name === selected;
      dom.blockOperationSelect.append(option);
    });
    dom.addSkillBlock.disabled = !toolboxPrimitives.length || !window.Blockly;
  }

  function blocklyTypeForOperation(operation) {
    return `dittobot_${operation.replace(/[^A-Za-z0-9_]/g, "_")}`;
  }

  function writeBlocklyBlockData(blocklyBlock, editorBlock) {
    blocklyBlock.data = JSON.stringify({
      operation: editorBlock.operation,
      arguments: cloneValue(editorBlock.arguments || {}),
    });
  }

  function readBlocklyBlockData(blocklyBlock) {
    const primitive = state.editor.catalog.find(
      (item) => blocklyTypeForOperation(item.operation_name) === blocklyBlock.type,
    );
    if (!primitive) return null;
    let stored = {};
    try {
      stored = JSON.parse(blocklyBlock.data || "{}");
    } catch (_error) {
      stored = {};
    }
    return {
      blockId: blocklyBlock.id,
      operation: primitive.operation_name,
      arguments: cloneValue(stored.arguments || primitive.default_arguments || {}),
    };
  }

  function registerBlocklyPrimitive(primitive) {
    const blockType = blocklyTypeForOperation(primitive.operation_name);
    window.Blockly.Blocks[blockType] = {
      init() {
        this.appendDummyInput()
          .appendField(primitive.operation_name);
        this.appendDummyInput()
          .appendField(primitive.description || "승인된 primitive");
        this.setPreviousStatement(true);
        this.setNextStatement(true);
        this.setColour({
          motion: 215,
          gripper: 25,
          contact: 345,
          control: 120,
        }[primitive.operation_name.split(".")[0]] || 265);
        this.setTooltip(primitive.description || primitive.operation_name);
        this.setHelpUrl("");
        this.data = JSON.stringify({
          operation: primitive.operation_name,
          arguments: cloneValue(primitive.default_arguments || {}),
        });
      },
    };
  }

  function buildBlocklyToolbox() {
    const groups = new Map();
    blocklyToolboxPrimitives().forEach((primitive) => {
      const group = primitive.operation_name.split(".")[0] || "primitive";
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push({
        kind: "block",
        type: blocklyTypeForOperation(primitive.operation_name),
      });
    });
    return {
      kind: "categoryToolbox",
      contents: [...groups.entries()].map(([name, contents]) => ({
        kind: "category",
        name: name.toUpperCase(),
        colour: {
          motion: "#4c77d9",
          gripper: "#d97732",
          contact: "#c43c78",
          control: "#4c9b63",
        }[name] || "#7755aa",
        contents,
      })),
    };
  }

  function syncBlocksFromBlockly() {
    const workspace = state.editor.workspace;
    if (!workspace) return;
    const allBlocks = workspace.getAllBlocks(false);
    if (!allBlocks.length) {
      state.editor.blocks = [];
      state.editor.selectedBlocklyBlockId = null;
      state.editor.blocklyError = null;
      return;
    }
    const topBlocks = workspace.getTopBlocks(true);
    if (topBlocks.length !== 1) {
      state.editor.blocks = [];
      state.editor.blocklyError = "모든 primitive를 위에서 아래로 하나의 체인에 연결해 주세요.";
      return;
    }
    const ordered = [];
    let current = topBlocks[0];
    while (current) {
      const block = readBlocklyBlockData(current);
      if (block) ordered.push(block);
      current = current.getNextBlock();
    }
    if (ordered.length !== allBlocks.length) {
      state.editor.blocks = [];
      state.editor.blocklyError = "분리된 블록이 있습니다. 하나의 순차 체인만 만들 수 있습니다.";
      return;
    }
    state.editor.blocks = ordered;
    state.editor.blocklyError = null;
    if (!ordered.some((block) => block.blockId === state.editor.selectedBlocklyBlockId)) {
      state.editor.selectedBlocklyBlockId = ordered[0]?.blockId || null;
    }
  }

  function renderBlocklyParameterEditor() {
    const selected = state.editor.blocks.find(
      (block) => block.blockId === state.editor.selectedBlocklyBlockId,
    ) || state.editor.blocks[0] || null;
    dom.blocklyParameterEditor.hidden = !selected;
    dom.blocklyParameterFields.replaceChildren();
    if (!selected) return;
    state.editor.selectedBlocklyBlockId = selected.blockId;
    const index = state.editor.blocks.indexOf(selected);
    dom.blocklyParameterTitle.textContent = `${index + 1}. ${selected.operation}`;
    const primitive = primitiveByOperation(selected.operation);
    renderArgumentEditor(
      dom.blocklyParameterFields,
      primitive,
      selected.arguments,
      (argumentsValue) => {
        selected.arguments = argumentsValue;
        const blocklyBlock = state.editor.workspace?.getBlockById(selected.blockId);
        if (blocklyBlock) writeBlocklyBlockData(blocklyBlock, selected);
        renderSkillBlocks();
      },
    );
  }

  function initializeBlockly() {
    if (state.editor.workspace || !window.Blockly || !state.editor.catalog.length) return;
    state.editor.catalog.forEach(registerBlocklyPrimitive);
    state.editor.workspace = window.Blockly.inject(dom.blocklyDiv, {
      toolbox: buildBlocklyToolbox(),
      media: "./vendor/blockly/media/",
      trashcan: true,
      move: { scrollbars: true, drag: true, wheel: true },
      zoom: { controls: true, wheel: true, startScale: 0.9, maxScale: 1.4, minScale: 0.5 },
      grid: { spacing: 24, length: 3, colour: "#d7d3e4", snap: true },
    });
    state.editor.blocklyCatalogSignature = blocklyToolboxPrimitives()
      .map((primitive) => primitive.operation_name)
      .join("|");
    state.editor.workspace.addChangeListener((event) => {
      if (event.type === "selected") {
        // Clicking the parameter panel is outside Blockly, so Blockly emits a
        // deselection with newElementId=null. Preserve the last real block
        // selection instead of falling back to the first block in the chain.
        if (!event.newElementId) return;
        state.editor.selectedBlocklyBlockId = event.newElementId;
      } else if (event.isUiEvent) {
        return;
      }
      syncBlocksFromBlockly();
      renderSkillBlocks();
    });
  }

  function renderSkillBlocks() {
    renderPrimitiveOptions();
    initializeBlockly();
    if (!window.Blockly) {
      state.editor.blocklyError = "Blockly 라이브러리를 불러오지 못했습니다.";
    } else if (state.editor.workspace && state.editor.catalog.length) {
      const signature = blocklyToolboxPrimitives()
        .map((primitive) => primitive.operation_name)
        .join("|");
      if (signature !== state.editor.blocklyCatalogSignature) {
        state.editor.catalog.forEach(registerBlocklyPrimitive);
        state.editor.workspace.updateToolbox(buildBlocklyToolbox());
        state.editor.blocklyCatalogSignature = signature;
      }
    }
    renderBlocklyParameterEditor();
    renderBlockBindings();
    const hasBindingConflict = collectBlockBindingHints()
      .some(([, hint]) => hint === "conflict");
    const ready = state.apiStatus === "connected"
      && state.editor.blocks.length > 0
      && !state.editor.loading
      && !state.editor.blocklyError
      && !hasBindingConflict;
    dom.previewSkillBlocks.disabled = !ready;
    dom.createBlockCandidate.disabled = !ready;
    dom.blocklyStatus.textContent = state.editor.blocklyError
      || (state.editor.blocks.length
        ? `${state.editor.blocks.length}개 primitive가 순차 연결되었습니다. 블록을 선택해 파라미터를 편집하세요.`
        : "toolbox에서 primitive 블록을 추가해 주세요.");
    dom.blocklyStatus.classList.toggle("danger", Boolean(state.editor.blocklyError));
  }

  function blockEditorPayload() {
    if (!dom.blockSkillForm.reportValidity()) return null;
    let bindings;
    try {
      bindings = serializedBlockBindings();
    } catch (error) {
      setBanner(`Binding 검증 실패: ${errorText(error)}`, "danger");
      return null;
    }
    return {
      skill_id: dom.blockSkillId.value.trim(),
      name: dom.blockSkillName.value.trim(),
      description: dom.blockSkillDescription.value.trim(),
      skill_type: dom.blockSkillType.value,
      blocks: state.editor.blocks.map((block) => ({
        operation: block.operation,
        arguments: cloneValue(block.arguments),
      })),
      bindings,
    };
  }

  function renderParameterEditor(skill) {
    dom.parameterEditor.hidden = !skill || !state.editor.catalog.length;
    if (!skill || !state.editor.catalog.length) return;
    if (state.editor.parameterSkillKey !== skill.key) {
      state.editor.parameterSkillKey = skill.key;
      state.editor.parameterNodeId = skill.nodes[0]?.nodeId || null;
      state.editor.parameterArguments = cloneValue(skill.nodes[0]?.arguments || {});
    }
    dom.parameterNodeSelect.replaceChildren();
    skill.nodes.forEach((node) => {
      const option = create("option", { text: `${node.nodeId} · ${node.operation}` });
      option.value = node.nodeId;
      option.selected = node.nodeId === state.editor.parameterNodeId;
      dom.parameterNodeSelect.append(option);
    });
    const node = skill.nodes.find((item) => item.nodeId === state.editor.parameterNodeId);
    renderArgumentEditor(
      dom.parameterFields,
      primitiveByOperation(node?.operation),
      state.editor.parameterArguments,
      (argumentsValue) => {
        state.editor.parameterArguments = argumentsValue;
        renderParameterEditor(skill);
      },
    );
    dom.createParameterCandidate.disabled = !node || !skill.checksum || state.editor.loading;
  }

  function renderDetail() {
    const skill = selectedSkill();
    dom.detailNodes.replaceChildren();
    dom.validateSkill.disabled = !skill || state.apiStatus !== "connected";
    dom.activateSkill.disabled = !skill || state.apiStatus !== "connected" || skill.uiState === "active";
    dom.activateSkill.hidden = Boolean(skill && skill.uiState === "active");
    dom.deactivateSkill.hidden = !skill || skill.uiState !== "active";
    dom.deactivateSkill.disabled = !skill || state.apiStatus !== "connected";
    if (!skill) {
      dom.detailTitle.textContent = "스킬을 선택하세요";
      dom.detailSubtitle.textContent = "스킬 관리 화면에서 버전을 선택해 주세요.";
      dom.parameterEditor.hidden = true;
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
    renderParameterEditor(skill);
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

    const setChip = (chip, text, tone = "") => {
      chip.textContent = text;
      chip.classList.remove("ok", "warn", "danger");
      if (tone) chip.classList.add(tone);
    };
    const aruco = state.arucoExperiment.status;
    const capabilities = aruco?.capabilities || {};
    const hardwareAruco = capabilities.mode === "hardware";
    const arucoEnabled = aruco?.enabled === true;
    const arucoError = state.arucoExperiment.lastError || aruco?.last_error;
    const referenceReady = Boolean(
      capabilities.reference && !capabilities.reference_error,
    );
    const runtimeReady = Boolean(aruco?.runtime_workspace);

    if (state.run.status === "estop") {
      setChip(dom.estopChip, "● E-STOP REQUESTED", "danger");
    } else if (arucoEnabled) {
      setChip(dom.estopChip, "● OPERATOR E-STOP ACK", "ok");
    } else {
      setChip(dom.estopChip, "○ E-STOP CHECK ON ENABLE");
    }

    if (state.camera.state === "streaming") {
      setChip(dom.cameraChip, "● REALSENSE STREAMING", "ok");
    } else if (state.camera.state === "starting") {
      setChip(dom.cameraChip, "● REALSENSE STARTING", "warn");
    } else if (state.camera.state === "error") {
      setChip(dom.cameraChip, "● REALSENSE ERROR", "danger");
    } else {
      setChip(dom.cameraChip, "○ REALSENSE STOPPED");
    }

    if (capabilities.reference_error) {
      setChip(dom.workspaceChip, "● WORKSPACE ERROR", "danger");
    } else if (aruco?.reference_captured && runtimeReady) {
      setChip(dom.workspaceChip, "● BASE/PLANE WORKSPACE ACTIVE", "ok");
    } else if (runtimeReady) {
      setChip(dom.workspaceChip, "● WIDTH WORKSPACE READY", "ok");
    } else if (referenceReady) {
      setChip(dom.workspaceChip, "● FROZEN REFERENCE VALID", "ok");
    } else {
      setChip(dom.workspaceChip, "○ FIXED WORKSPACE CHECKING");
    }

    if (state.apiStatus !== "connected") {
      setChip(dom.modeChip, "● FASTAPI DISCONNECTED", "danger");
    } else if (arucoError) {
      setChip(dom.modeChip, "● ARUCO SESSION ERROR", "danger");
    } else if (hardwareAruco && capabilities.hardware_authorized === true) {
      setChip(dom.modeChip, "● HARDWARE ARUCO GATES READY", "ok");
    } else if (hardwareAruco) {
      setChip(dom.modeChip, "● HARDWARE GATES CLOSED", "danger");
    } else {
      setChip(dom.modeChip, "○ MOCK MODE");
    }
    const canAbort = Boolean(state.run.runId) && ["starting", "running", "aborting"].includes(state.run.status);
    dom.abortRun.disabled = !canAbort;
    dom.estopRun.disabled = !canAbort;
  }

  function renderJog() {
    const payload = state.jog.status;
    const capabilities = payload?.capabilities || {};
    const enabled = payload?.enabled === true;
    const hardware = capabilities.mode === "hardware";
    const busy = state.jog.loading;
    const positions = Array.isArray(payload?.joint_positions_deg)
      ? payload.joint_positions_deg
      : [0, 0, 0, 0, 0, 0];
    if (!Array.isArray(state.jog.targetPositionsDeg)
      || state.jog.targetPositionsDeg.length !== 6) {
      state.jog.targetPositionsDeg = positions.map((value) => Number(value));
      state.jog.targetDirty = false;
    }

    dom.jogModeNotice.textContent = hardware
      ? "실제 M0609 하드웨어 조그 모드입니다. MOVEJ를 누르면 입력한 6축 목표로 즉시 이동합니다."
      : "MOCK 조그 모드입니다. 화면의 관절값만 변경되며 실제 로봇은 움직이지 않습니다.";
    renderConnection();
    dom.jogModeNotice.style.color = hardware ? "var(--danger)" : "";
    dom.jogStatus.textContent = state.jog.lastError
      ? `오류 · ${state.jog.lastError}`
      : enabled
        ? `● 활성 · ${payload.operator_id || "operator"} · ${payload.adapter_name || capabilities.mode}`
        : "○ 비활성 · 안전 확인 후 활성화하세요.";
    dom.jogStatus.style.color = state.jog.lastError
      ? "var(--danger)"
      : enabled ? "var(--ok)" : "";

    const failedGates = Array.isArray(capabilities.failed_gates)
      ? capabilities.failed_gates
      : [];
    dom.jogGates.textContent = hardware
      ? `실제 로봇 gate: 모두 통과\nMOVEJ 속도/가속도: 승인된 ${capabilities.motion_profile_id || "joint_safe"} 프로파일`
      : `실제 로봇은 비활성화됨${failedGates.length ? `\n닫힌 gate:\n- ${failedGates.join("\n- ")}` : ""}\n현재 조작은 MOCK 전용`;

    dom.jogEnable.disabled = busy || enabled || state.apiStatus !== "connected";
    dom.jogRefresh.disabled = busy || state.apiStatus !== "connected";
    dom.jogStop.disabled = busy || !enabled;
    dom.jogOperatorId.disabled = enabled || busy;
    dom.jogWorkspaceCleared.disabled = enabled || busy;
    dom.jogEstopReady.disabled = enabled || busy;
    dom.jogDirectMotionAck.disabled = enabled || busy;
    dom.jogStepDeg.disabled = busy;
    dom.jogLoadCurrent.disabled = busy || !payload;
    dom.jogMoveJ.disabled = busy || !enabled;
    if (document.activeElement?.classList.contains("jog-target-angle")) return;
    dom.jogJointList.replaceChildren();
    for (let jointIndex = 1; jointIndex <= 6; jointIndex += 1) {
      const row = create("div", { className: "jog-joint" });
      const limits = capabilities.joint_limits_deg?.[jointIndex - 1];
      const name = create("strong", { text: `J${jointIndex}` });
      if (limits) name.title = `허용 범위 ${limits.minimum}° .. ${limits.maximum}°`;
      const currentAngle = create("span", {
        className: "jog-angle",
        text: `현재 ${Number(positions[jointIndex - 1] || 0).toFixed(3)}°`,
      });
      const target = create("input", { className: "jog-target-angle" });
      target.type = "number";
      target.step = "0.1";
      target.inputMode = "decimal";
      target.setAttribute("aria-label", `J${jointIndex} 목표 각도 (degree)`);
      if (limits) {
        target.min = String(limits.minimum);
        target.max = String(limits.maximum);
      }
      target.value = String(state.jog.targetPositionsDeg[jointIndex - 1]);
      target.disabled = !enabled || busy;
      target.addEventListener("input", () => {
        state.jog.targetPositionsDeg[jointIndex - 1] = target.value;
        state.jog.targetDirty = true;
      });
      const minus = create("button", { className: "button", text: "− 설정" });
      const plus = create("button", { className: "button", text: "+ 설정" });
      minus.type = "button";
      plus.type = "button";
      minus.disabled = !enabled || busy;
      plus.disabled = !enabled || busy;
      minus.addEventListener("click", () => moveJogJoint(jointIndex, -1));
      plus.addEventListener("click", () => moveJogJoint(jointIndex, 1));
      row.append(name, currentAngle, target, minus, plus);
      dom.jogJointList.append(row);
    }
  }

  async function refreshJogStatus({ quiet = false } = {}) {
    if (state.apiStatus !== "connected") return;
    try {
      const wasEnabled = state.jog.status?.enabled === true;
      state.jog.status = await api.jogStatus();
      if (!state.jog.targetDirty
        && Array.isArray(state.jog.status?.joint_positions_deg)) {
        state.jog.targetPositionsDeg = state.jog.status.joint_positions_deg.map(Number);
      }
      if (wasEnabled && state.jog.status?.enabled !== true) {
        dom.jogWorkspaceCleared.checked = false;
        dom.jogEstopReady.checked = false;
        dom.jogDirectMotionAck.checked = false;
      }
      state.jog.lastError = null;
    } catch (error) {
      state.jog.lastError = errorText(error);
      if (!quiet) setBanner(`조그 상태 확인 실패: ${errorText(error)}`, "danger");
    }
    renderJog();
  }

  async function enableJog() {
    if (
      !dom.jogWorkspaceCleared.checked
      || !dom.jogEstopReady.checked
      || !dom.jogDirectMotionAck.checked
    ) {
      setBanner("조그 활성화 전에 세 가지 안전 항목을 모두 확인하세요.", "danger");
      return;
    }
    state.jog.loading = true;
    state.jog.lastError = null;
    renderJog();
    try {
      state.jog.status = await api.enableJog({
        operator_id: dom.jogOperatorId.value.trim() || "ui_operator",
        workspace_cleared: true,
        estop_ready: true,
        acknowledge_direct_motion: true,
      });
      state.jog.targetPositionsDeg = state.jog.status.joint_positions_deg.map(Number);
      state.jog.targetDirty = false;
      const hardware = state.jog.status?.capabilities?.mode === "hardware";
      setBanner(
        hardware ? "실제 로봇 조그가 활성화되었습니다. 로봇 주변에 접근하지 마세요." : "MOCK 조그가 활성화되었습니다.",
        hardware ? "danger" : "ok",
      );
    } catch (error) {
      state.jog.lastError = errorText(error);
      setBanner(`조그 활성화 실패: ${errorText(error)}`, "danger");
    } finally {
      state.jog.loading = false;
      renderJog();
    }
  }

  function moveJogJoint(jointIndex, direction) {
    const step = Number(dom.jogStepDeg.value);
    if (!Number.isFinite(step) || step < 0.1 || step > 5) {
      setBanner("목표각 설정 간격은 0.1° 이상 5° 이하여야 합니다.", "danger");
      return;
    }
    const targetIndex = jointIndex - 1;
    const currentTarget = Number(state.jog.targetPositionsDeg[targetIndex]);
    const limits = state.jog.status?.capabilities?.joint_limits_deg?.[targetIndex];
    const nextTarget = Number((currentTarget + direction * step).toFixed(3));
    if (!Number.isFinite(nextTarget)
      || (limits && (nextTarget < limits.minimum || nextTarget > limits.maximum))) {
      setBanner(`J${jointIndex} 목표각이 허용 범위를 벗어납니다.`, "danger");
      return;
    }
    state.jog.targetPositionsDeg[targetIndex] = nextTarget;
    state.jog.targetDirty = true;
    renderJog();
  }

  function loadCurrentJogTargets() {
    const positions = state.jog.status?.joint_positions_deg;
    if (!Array.isArray(positions) || positions.length !== 6) return;
    state.jog.targetPositionsDeg = positions.map(Number);
    state.jog.targetDirty = false;
    renderJog();
    setBanner("현재 관절각을 MOVEJ 목표로 불러왔습니다.");
  }

  async function executeJogMoveJ() {
    const rawTargets = state.jog.targetPositionsDeg;
    const hasBlankTarget = rawTargets?.some(
      (value) => typeof value === "string" && value.trim() === "",
    );
    const targets = rawTargets?.map(Number);
    if (hasBlankTarget || !Array.isArray(targets) || targets.length !== 6
      || targets.some((value) => !Number.isFinite(value))) {
      setBanner("J1–J6 목표 각도를 모두 숫자로 입력하세요.", "danger");
      return;
    }
    const limits = state.jog.status?.capabilities?.joint_limits_deg || [];
    const invalidIndex = targets.findIndex((value, index) => {
      const limit = limits[index];
      return limit && (value < limit.minimum || value > limit.maximum);
    });
    if (invalidIndex >= 0) {
      const limit = limits[invalidIndex];
      setBanner(
        `J${invalidIndex + 1} 목표각은 ${limit.minimum}°–${limit.maximum}° 안이어야 합니다.`,
        "danger",
      );
      return;
    }
    const hardware = state.jog.status?.capabilities?.mode === "hardware";
    if (hardware && !window.confirm(
      `입력한 목표각 [${targets.join(", ")}]°으로 MOVEJ를 실행할까요?`,
    )) return;
    state.jog.loading = true;
    state.jog.lastError = null;
    renderJog();
    try {
      state.jog.status = await api.moveJogJoints(targets);
      state.jog.targetPositionsDeg = state.jog.status.joint_positions_deg.map(Number);
      state.jog.targetDirty = false;
      setBanner(`MOVEJ 완료 · [${targets.join(", ")}]°`, "ok");
    } catch (error) {
      state.jog.lastError = errorText(error);
      setBanner(`MOVEJ 실패: ${errorText(error)}`, "danger");
    } finally {
      state.jog.loading = false;
      renderJog();
    }
  }

  async function stopJog() {
    const hardware = state.jog.status?.capabilities?.mode === "hardware";
    if (hardware && !window.confirm("로봇 stop을 요청하고 조그를 비활성화할까요?")) return;
    state.jog.loading = true;
    renderJog();
    try {
      state.jog.status = await api.stopJog("ui_operator_request");
      state.jog.lastError = null;
      state.jog.targetPositionsDeg = state.jog.status.joint_positions_deg.map(Number);
      state.jog.targetDirty = false;
      dom.jogWorkspaceCleared.checked = false;
      dom.jogEstopReady.checked = false;
      dom.jogDirectMotionAck.checked = false;
      setBanner("조그 정지 및 비활성화 완료", "ok");
    } catch (error) {
      state.jog.lastError = errorText(error);
      setBanner(`조그 정지 실패: ${errorText(error)}`, "danger");
    } finally {
      state.jog.loading = false;
      renderJog();
    }
  }

  function renderArucoExperiment() {
    const payload = state.arucoExperiment.status;
    const capabilities = payload?.capabilities || {};
    const enabled = payload?.enabled === true;
    const hardware = capabilities.mode === "hardware";
    const busy = state.arucoExperiment.loading;
    const reference = capabilities.reference || null;
    const runtime = payload?.runtime_workspace || null;
    const failedGates = Array.isArray(capabilities.failed_gates)
      ? capabilities.failed_gates
      : [];

    dom.arucoExperimentModeNotice.textContent = capabilities.reference_error
      ? `Frozen reference 오류: ${capabilities.reference_error}`
      : hardware
        ? "실제 M0609 ArUco 실험 모드입니다. 두 이동 버튼은 로봇을 즉시 움직입니다."
        : "MOCK ArUco 실험 모드입니다. 계산과 runtime NPZ는 실제와 같지만 로봇은 움직이지 않습니다.";
    dom.arucoExperimentModeNotice.style.color = (
      hardware || capabilities.reference_error
    ) ? "var(--danger)" : "";
    dom.arucoExperimentStatus.textContent = state.arucoExperiment.lastError
      ? `오류 · ${state.arucoExperiment.lastError}`
      : enabled
        ? `● 활성 · ${payload.operator_id || "operator"} · ${payload.last_action || "대기"}`
        : "○ 비활성 · 물체 폭과 안전 확인 후 활성화하세요.";
    dom.arucoExperimentStatus.style.color = state.arucoExperiment.lastError
      ? "var(--danger)"
      : enabled ? "var(--ok)" : "";
    dom.arucoExperimentGates.textContent = hardware
      ? `실제 로봇 gate: 모두 통과\nactive TCP: ${capabilities.expected_tcp_name}\nbase 반경: ${capabilities.maximum_base_radius_m} m 이하`
      : `실제 로봇은 비활성화됨${failedGates.length ? `\n닫힌 gate:\n- ${failedGates.join("\n- ")}` : ""}\n현재 이동은 MOCK 전용`;

    const legacyModel = dom.arucoWidthModel.value === "legacy-half-factor";
    dom.arucoObjectWidthMm.max = legacyModel ? "55" : "110";
    dom.arucoExperimentEnable.disabled = busy || enabled || state.apiStatus !== "connected"
      || Boolean(capabilities.reference_error);
    dom.arucoExperimentRefresh.disabled = busy || state.apiStatus !== "connected";
    dom.arucoExperimentStop.disabled = busy || !enabled;
    dom.arucoMoveReference.disabled = busy || !enabled;
    dom.arucoMoveZTest.disabled = busy || !enabled || !payload?.reference_captured
      || payload?.z_test_completed;
    for (const input of [
      dom.arucoExperimentOperatorId,
      dom.arucoObjectWidthMm,
      dom.arucoWidthModel,
      dom.arucoWorkspaceCleared,
      dom.arucoEstopReady,
      dom.arucoDirectMotionAck,
    ]) input.disabled = enabled || busy;

    dom.arucoWorkspaceSummary.textContent = JSON.stringify({
      frame: reference?.frame_name || null,
      fixed_reference_sha256: reference?.checksum_sha256 || null,
      reference_joint_deg: reference?.reference_joint_deg || null,
      plane_to_camera_z_range_m: reference?.plane_to_camera_z_range_m || null,
      reference_tcp_plane_xyz_m: reference?.reference_tcp_plane_xyz_m || null,
      active_tcp_name: payload?.active_tcp_name || null,
      object_width_mm: runtime?.object_width_mm ?? null,
      theta_deg: runtime ? Number(runtime.theta_deg).toFixed(3) : null,
      opening_offset_mm: runtime ? (Number(runtime.opening_offset_m) * 1000).toFixed(3) : null,
      tcp_z_bounds_plane_m: runtime
        ? [runtime.z_min_plane_m, runtime.z_max_plane_m]
        : null,
      reference_captured: payload?.reference_captured || false,
      z_test_completed: payload?.z_test_completed || false,
      tcp_base_xyz_m: payload?.tcp_base_xyz_m || null,
      unsafe_targets_are_clamped: capabilities.unsafe_targets_are_clamped ?? false,
    }, null, 2);
    renderConnection();
  }

  async function refreshArucoExperimentStatus({ quiet = false } = {}) {
    if (state.apiStatus !== "connected") return;
    try {
      const wasEnabled = state.arucoExperiment.status?.enabled === true;
      state.arucoExperiment.status = await api.arucoExperimentStatus();
      if (wasEnabled && state.arucoExperiment.status?.enabled !== true) {
        dom.arucoWorkspaceCleared.checked = false;
        dom.arucoEstopReady.checked = false;
        dom.arucoDirectMotionAck.checked = false;
      }
      state.arucoExperiment.lastError = null;
    } catch (error) {
      state.arucoExperiment.lastError = errorText(error);
      if (!quiet) setBanner(`ArUco 실험 상태 확인 실패: ${errorText(error)}`, "danger");
    }
    renderArucoExperiment();
  }

  async function enableArucoExperiment() {
    if (
      !dom.arucoWorkspaceCleared.checked
      || !dom.arucoEstopReady.checked
      || !dom.arucoDirectMotionAck.checked
    ) {
      setBanner("ArUco 실험 활성화 전에 세 가지 안전 항목을 모두 확인하세요.", "danger");
      return;
    }
    const widthMm = Number(dom.arucoObjectWidthMm.value);
    const maximumWidthMm = dom.arucoWidthModel.value === "legacy-half-factor" ? 55 : 110;
    if (!Number.isFinite(widthMm) || widthMm < 0 || widthMm > maximumWidthMm) {
      setBanner(`현재 폭 모델에서 물체 폭은 0–${maximumWidthMm} mm여야 합니다.`, "danger");
      return;
    }
    const hardware = state.arucoExperiment.status?.capabilities?.mode === "hardware";
    if (hardware && !window.confirm(
      "실제 M0609 실험 세션을 활성화합니다.\n\n"
      + `물체 폭: ${widthMm} mm\n`
      + "다음 단계에서 기준 자세 MoveJ와 plane +Z 20 mm MoveL이 실행됩니다.\n"
      + "작업공간 비움과 E-stop 준비를 다시 확인했습니까?",
    )) return;
    state.arucoExperiment.loading = true;
    state.arucoExperiment.lastError = null;
    renderArucoExperiment();
    try {
      state.arucoExperiment.status = await api.enableArucoExperiment({
        operator_id: dom.arucoExperimentOperatorId.value.trim() || "ui_operator",
        workspace_cleared: true,
        estop_ready: true,
        acknowledge_direct_motion: true,
        object_width_mm: widthMm,
        width_model: dom.arucoWidthModel.value,
      });
      setBanner(
        hardware
          ? "실제 ArUco 실험 활성화 완료 · 기준 자세 버튼을 누르세요."
          : "MOCK ArUco 실험 활성화 및 runtime workspace 생성 완료",
        hardware ? "danger" : "ok",
      );
    } catch (error) {
      state.arucoExperiment.lastError = errorText(error);
      setBanner(`ArUco 실험 활성화 실패: ${errorText(error)}`, "danger");
    } finally {
      state.arucoExperiment.loading = false;
      renderArucoExperiment();
    }
  }

  async function moveArucoReference() {
    const hardware = state.arucoExperiment.status?.capabilities?.mode === "hardware";
    if (hardware && !window.confirm(
      "M0609를 [0, 0, 90, 0, 90, -90]° 기준 자세로 이동합니다.\n"
      + "로봇 주변이 비어 있고 E-stop을 잡고 있습니까?",
    )) return;
    state.arucoExperiment.loading = true;
    renderArucoExperiment();
    try {
      state.arucoExperiment.status = await api.moveArucoReference();
      state.arucoExperiment.lastError = null;
      setBanner("기준 자세 도달 및 frozen plane의 base 결합 완료", "ok");
    } catch (error) {
      state.arucoExperiment.lastError = errorText(error);
      setBanner(`기준 자세 이동 실패: ${errorText(error)}`, "danger");
    } finally {
      state.arucoExperiment.loading = false;
      renderArucoExperiment();
    }
  }

  async function moveArucoPlaneZTest() {
    const hardware = state.arucoExperiment.status?.capabilities?.mode === "hardware";
    if (hardware && !window.confirm(
      "TCP를 frozen plane +Z(카메라/테이블 반대 방향)로 정확히 20 mm MoveL 합니다.\n"
      + "이 동작은 한 번만 허용됩니다. 실행할까요?",
    )) return;
    state.arucoExperiment.loading = true;
    renderArucoExperiment();
    try {
      state.arucoExperiment.status = await api.moveArucoPlaneZTest();
      state.arucoExperiment.lastError = null;
      setBanner("Plane +Z 20 mm 검증 동작 완료", "ok");
    } catch (error) {
      state.arucoExperiment.lastError = errorText(error);
      setBanner(`Plane +Z 검증 거절/실패: ${errorText(error)}`, "danger");
    } finally {
      state.arucoExperiment.loading = false;
      renderArucoExperiment();
    }
  }

  async function stopArucoExperiment() {
    const hardware = state.arucoExperiment.status?.capabilities?.mode === "hardware";
    if (hardware && !window.confirm("로봇 quick stop을 요청하고 실험을 비활성화할까요?")) return;
    state.arucoExperiment.loading = true;
    renderArucoExperiment();
    try {
      state.arucoExperiment.status = await api.stopArucoExperiment(
        "ui_operator_request",
      );
      state.arucoExperiment.lastError = null;
      dom.arucoWorkspaceCleared.checked = false;
      dom.arucoEstopReady.checked = false;
      dom.arucoDirectMotionAck.checked = false;
      setBanner("ArUco 실험 정지 및 비활성화 완료", "ok");
    } catch (error) {
      state.arucoExperiment.lastError = errorText(error);
      setBanner(`ArUco 실험 정지 실패: ${errorText(error)}`, "danger");
    } finally {
      state.arucoExperiment.loading = false;
      renderArucoExperiment();
    }
  }

  function renderCamera() {
    const camera = state.camera;
    const streaming = camera.state === "streaming";
    if (streaming && dom.rgbStream.dataset.streaming !== "true") {
      dom.rgbStream.src = api.cameraStreamUrl("rgb");
      dom.depthStream.src = api.cameraStreamUrl("depth");
      dom.rgbStream.dataset.streaming = "true";
      dom.depthStream.dataset.streaming = "true";
    } else if (!streaming && dom.rgbStream.dataset.streaming === "true") {
      dom.rgbStream.removeAttribute("src");
      dom.depthStream.removeAttribute("src");
      delete dom.rgbStream.dataset.streaming;
      delete dom.depthStream.dataset.streaming;
    }
    dom.rgbStream.hidden = !streaming;
    dom.depthStream.hidden = !streaming;
    dom.rgbPlaceholder.hidden = streaming;
    dom.depthPlaceholder.hidden = streaming;
    dom.startCamera.disabled = state.apiStatus !== "connected" || ["starting", "streaming"].includes(camera.state);
    dom.stopCamera.disabled = !["starting", "streaming", "error"].includes(camera.state);
    dom.startRecording.disabled = !streaming || Boolean(camera.recording);
    dom.stopRecording.disabled = !camera.recording;
    dom.recordDuration.disabled = Boolean(camera.recording);

    if (camera.recording) {
      dom.cameraStatus.textContent = `● 녹화 중 · ${camera.recording.recording_id} · 저장 ${camera.recording.frame_count}프레임 · 누락 ${camera.recording.dropped_frame_count}프레임`;
      dom.cameraStatus.style.color = "var(--danger)";
    } else if (camera.lastRecording) {
      dom.cameraStatus.textContent = `녹화 ${camera.lastRecording.status} · ${camera.lastRecording.frame_count}프레임 · ${camera.lastRecording.duration_s.toFixed(1)}초 · ${camera.lastRecording.manifest_uri || "manifest 생성 중"}`;
      dom.cameraStatus.style.color = camera.lastRecording.status === "failed" ? "var(--danger)" : "var(--ok)";
    } else if (camera.state === "streaming") {
      const skewLimit = Number.isFinite(Number(camera.maximumTimestampSkewMs))
        ? ` · 동기화 한도 ${Number(camera.maximumTimestampSkewMs).toFixed(0)} ms`
        : "";
      dom.cameraStatus.textContent = `RealSense RGB + 정렬 Depth 스트리밍 중 · 프레임 ${camera.frameNumber ?? "대기"}${skewLimit}`;
      dom.cameraStatus.style.color = "var(--ok)";
    } else if (camera.state === "error") {
      dom.cameraStatus.textContent = `RealSense 오류: ${camera.lastError || "장치 또는 Python 바인딩을 확인하세요."}`;
      dom.cameraStatus.style.color = "var(--danger)";
    } else {
      dom.cameraStatus.textContent = "RealSense 정지됨 · 사용자가 시작할 때만 카메라가 열립니다.";
      dom.cameraStatus.style.color = "";
    }
    renderCalibration();
    if (state.page === "monitor") renderMonitor();
  }

  function renderCalibration() {
    const calibration = state.calibration;
    const capabilities = calibration.capabilities || {};
    const session = calibration.session;
    const active = ["starting", "moving_to_reference", "running", "aborting"]
      .includes(session?.status);
    const poseCount = Number(capabilities.pose_count || session?.planned_pose_count || 21);
    const accepted = Number(session?.accepted_observation_count || 0);
    const rejected = Number(session?.rejected_observation_count || 0);
    dom.calibrationProgress.max = Math.max(1, poseCount);
    dom.calibrationProgress.value = Math.min(accepted, poseCount);
    dom.startHandeyeCalibration.disabled = (
      calibration.loading
      || active
      || state.camera.state !== "streaming"
      || capabilities.hardware_authorized !== true
    );
    dom.abortHandeyeCalibration.disabled = !active;
    const legacyCapability = capabilities.legacy_npy || {};
    dom.importLegacyHandeye.disabled = (
      calibration.loading
      || active
      || capabilities.hardware_authorized !== true
      || legacyCapability.available !== true
    );
    const failedGates = Array.isArray(capabilities.failed_gates) ? capabilities.failed_gates : [];
    if (calibration.lastError) {
      dom.calibrationStatus.textContent = `보정 상태 오류: ${calibration.lastError}`;
      dom.calibrationStatus.style.color = "var(--danger)";
    } else if (session) {
      dom.calibrationStatus.textContent = `${session.status} · ${accepted}/${poseCount}개 승인 · ${rejected}개 거절 · ${session.message || "대기"}`;
      dom.calibrationStatus.style.color = session.status === "passed"
        ? "var(--ok)"
        : ["failed", "aborted"].includes(session.status)
          ? "var(--danger)"
          : "";
    } else if (failedGates.length) {
      dom.calibrationStatus.textContent = `선택적 재보정 비활성 · frozen ArUco 실행에는 영향 없음 · ${failedGates.join(", ")}`;
      dom.calibrationStatus.style.color = "";
    } else {
      dom.calibrationStatus.textContent = state.camera.state === "streaming"
        ? "Calibrate 준비됨 · 기준 자세와 안전 조건을 확인하세요."
        : "먼저 RealSense RGB-D를 시작하세요.";
      dom.calibrationStatus.style.color = "";
    }
    const result = calibration.legacyTransform || calibration.latestResult;
    dom.calibrationResult.hidden = !result;
    dom.calibrationResult.textContent = result ? JSON.stringify({
      passed: result.passed,
      evidence_type: result.evidence_type,
      active_tcp_name: result.active_tcp_name,
      expected_tcp_name: result.expected_tcp_name,
      transform_convention: result.transform_convention,
      flange_to_camera: result.flange_to_camera,
      metrics: result.metrics,
      failures: result.failures,
      artifact_uri: result.artifact_uri,
    }, null, 2) : "";
  }

  function stopRecordingPlayback() {
    const review = state.recordingReview;
    if (review.playbackTimer !== null) {
      window.clearInterval(review.playbackTimer);
      review.playbackTimer = null;
    }
    review.playing = false;
  }

  function geometryPointFromEvent(event) {
    const image = dom.recordedRgbFrame;
    if (!image.naturalWidth || !image.naturalHeight) return null;
    const rect = image.getBoundingClientRect();
    const scale = Math.min(
      rect.width / image.naturalWidth,
      rect.height / image.naturalHeight,
    );
    const renderedWidth = image.naturalWidth * scale;
    const renderedHeight = image.naturalHeight * scale;
    const offsetX = (rect.width - renderedWidth) / 2;
    const offsetY = (rect.height - renderedHeight) / 2;
    const imageX = event.clientX - rect.left - offsetX;
    const imageY = event.clientY - rect.top - offsetY;
    if (imageX < 0 || imageY < 0 || imageX > renderedWidth || imageY > renderedHeight) {
      return null;
    }
    return {
      x_px: imageX / scale,
      y_px: imageY / scale,
    };
  }

  function fingerObservationForFrame(frameIndex) {
    const review = state.recordingReview;
    if (review.fingerTrackingRecordingId !== review.selectedId) return null;
    return review.fingerObservations.find(
      (item) => Number(item.frame_index) === Number(frameIndex),
    ) || null;
  }

  function renderGeometryMarkers() {
    dom.recordedPointOverlay.replaceChildren();
    const image = dom.recordedRgbFrame;
    if (!image.naturalWidth) return;
    const scale = Math.min(
      image.clientWidth / image.naturalWidth,
      image.clientHeight / image.naturalHeight,
    );
    const offsetX = (image.clientWidth - image.naturalWidth * scale) / 2;
    const offsetY = (image.clientHeight - image.naturalHeight * scale) / 2;
    const observation = fingerObservationForFrame(state.recordingReview.frameIndex);
    if (observation) {
      [
        {
          kind: "thumb",
          label: "T",
          pixel: observation.thumb_pixel_xy,
          depthM: observation.thumb_depth_m,
        },
        {
          kind: "index",
          label: "I",
          pixel: observation.index_pixel_xy,
          depthM: observation.index_depth_m,
        },
      ].forEach((tip) => {
        if (!Array.isArray(tip.pixel) || tip.pixel.length !== 2) return;
        const depthLabel = Number.isFinite(Number(tip.depthM))
          ? `${Number(tip.depthM).toFixed(3)}m`
          : "depth ?";
        const marker = create("span", {
          className: `finger-marker ${tip.kind}`,
          text: `${tip.label} (${Number(tip.pixel[0])},${Number(tip.pixel[1])}) · ${depthLabel}`,
        });
        marker.style.left = `${offsetX + Number(tip.pixel[0]) * scale}px`;
        marker.style.top = `${offsetY + Number(tip.pixel[1]) * scale}px`;
        dom.recordedPointOverlay.append(marker);
      });
      const distanceCm = Number.isFinite(Number(observation.distance_m))
        ? `${(Number(observation.distance_m) * 100).toFixed(2)} cm`
        : "거리 불확실";
      const candidate = observation.candidate_state || "uncertain";
      const progress = Number(observation.stabilization_progress_frames || 0);
      const required = Number(observation.required_stable_frames || 3);
      const confirmed = observation.stable_state || "미확정";
      const readout = create("span", {
        className: `finger-readout${observation.status === "uncertain" ? " uncertain" : ""}`,
        text: observation.status === "uncertain"
          ? `distance ${distanceCm} · candidate uncertain · stable ${progress}/${required} · confirmed ${confirmed} · ${observation.invalid_reason || "RGB-D fingertip unavailable"}`
          : `distance ${distanceCm} · candidate ${candidate} · stable ${progress}/${required} · confirmed ${confirmed}`,
      });
      dom.recordedPointOverlay.append(readout);
    }

    const teaching = state.geometryTeaching;
    if (!teaching.mode) return;
    const labels = teaching.mode === "surface_calibration" ? ["O", "X", "Y"] : ["A", "B"];
    teaching.points.forEach((point, index) => {
      const marker = create("span", { className: "point-marker", text: labels[index] || String(index + 1) });
      marker.style.left = `${offsetX + point.x_px * scale}px`;
      marker.style.top = `${offsetY + point.y_px * scale}px`;
      dom.recordedPointOverlay.append(marker);
    });
  }

  function renderGeometryTeaching() {
    const teaching = state.geometryTeaching;
    const calibrationMode = teaching.mode === "surface_calibration";
    const pathMode = teaching.mode === "manual_tcp_path";
    dom.geometryTeaching.hidden = !teaching.mode;
    dom.recordedRgbFrame.classList.toggle("teaching-image", Boolean(teaching.mode));
    if (!teaching.mode) {
      return;
    }
    dom.geometryTeachingTitle.textContent = calibrationMode
      ? "camera → task-plane TF 보정"
      : "두 fingertip TCP 경로 티칭";
    dom.geometryTeachingInstruction.textContent = calibrationMode
      ? "한 프레임의 같은 평면에서 원점(O), +X 방향점(X), +Y 방향점(Y)을 순서대로 클릭하세요. 세 점은 각각 3 cm 이상 떨어뜨리세요."
      : "현재 프레임에서 집게로 사용할 두 fingertip 끝 A와 B를 클릭하고 ‘현재 프레임 샘플 추가’를 누르세요. 서로 다른 프레임에서 최소 2개를 저장합니다.";
    dom.surfaceAnchorLabel.hidden = !calibrationMode;
    dom.addPathSample.hidden = !pathMode;
    dom.addPathSample.disabled = teaching.busy || teaching.points.length !== 2;
    dom.saveGeometryEvidence.disabled = teaching.busy || (
      calibrationMode ? teaching.points.length !== 3 : teaching.samples.length < 2
    );
    dom.resetGeometryPoints.disabled = teaching.busy || teaching.points.length === 0;
    dom.cancelGeometryTeaching.disabled = teaching.busy;
    dom.geometryPointSummary.textContent = calibrationMode
      ? `선택한 점 ${teaching.points.length}/3 · 기준 프레임 ${state.recordingReview.frameIndex}`
      : `현재 fingertip ${teaching.points.length}/2 · 저장된 경로 샘플 ${teaching.samples.length}/2 이상`;
    renderGeometryMarkers();
  }

  function renderRecordingFrame() {
    const review = state.recordingReview;
    const recording = selectedRecording();
    const frameCount = Number(recording?.frame_count || 0);
    const hasFrames = frameCount > 0;
    const maximumIndex = Math.max(0, frameCount - 1);
    review.frameIndex = Math.min(Math.max(0, review.frameIndex), maximumIndex);
    dom.recordingFrameIndex.max = String(maximumIndex);
    dom.recordingFrameIndex.value = String(review.frameIndex);
    dom.recordingFrameIndex.disabled = !hasFrames;
    dom.toggleRecordingPlayback.disabled = !hasFrames || Boolean(state.geometryTeaching.mode);
    dom.toggleRecordingPlayback.textContent = review.playing ? "일시정지" : "재생";
    dom.recordingFrameLabel.textContent = hasFrames
      ? `${review.frameIndex + 1} / ${frameCount}`
      : "0 / 0";

    dom.recordedRgbFrame.hidden = !hasFrames;
    dom.recordedDepthFrame.hidden = !hasFrames;
    dom.recordedRgbPlaceholder.hidden = hasFrames;
    dom.recordedDepthPlaceholder.hidden = hasFrames;
    if (hasFrames && recording) {
      dom.recordedRgbFrame.src = api.recordingFrameUrl(
        recording.recording_id, review.frameIndex, "rgb",
      );
      dom.recordedDepthFrame.src = api.recordingFrameUrl(
        recording.recording_id, review.frameIndex, "depth",
      );
    } else {
      dom.recordedRgbFrame.removeAttribute("src");
      dom.recordedDepthFrame.removeAttribute("src");
    }
    renderGeometryTeaching();
  }

  function renderRecordingCapability() {
    const capabilities = state.recordingReview.capabilities;
    if (!capabilities) {
      dom.openaiCapability.textContent = "OpenAI 연결 상태 확인 중…";
      return;
    }
    const live = capabilities.openai_mode === "live";
    const ready = !live || capabilities.api_key_configured;
    dom.openaiCapability.textContent = live
      ? ready
        ? `GPT LIVE 준비됨 · ${capabilities.model} · 첫 RGB 1장 + 전체 엄지/검지 trace · Depth/거리/상태는 로컬 유지`
        : "OPENAI_MODE=live이지만 서버에 OPENAI_API_KEY가 없습니다."
      : `현재 MOCK 분석 모드 · 실제 GPT 전송은 OPENAI_MODE=live에서만 수행됩니다. · ${capabilities.model}`;
    dom.openaiCapability.style.color = ready ? "" : "var(--danger)";
    dom.draftKeyframeCount.max = "1";
    dom.draftKeyframeCount.value = "1";
  }

  function renderRecordingReview() {
    const review = state.recordingReview;
    const recording = selectedRecording();
    renderRecordingFrame();
    renderRecordingCapability();
    const capabilities = review.capabilities;
    const liveReady = capabilities?.openai_mode !== "live"
      || Boolean(capabilities?.api_key_configured);
    dom.analyzeRecording.disabled = review.loading
      || state.apiStatus !== "connected"
      || !recording
      || !liveReady;
    if (review.loading) {
      dom.draftStatus.textContent = "첫 RGB 1장과 전체 녹화 엄지/검지 trace를 준비하는 중…";
    } else if (!recording) {
      dom.draftStatus.textContent = "녹화를 선택해 주세요.";
    }
    dom.draftResult.hidden = !review.draft;
    if (review.draft) {
      dom.draftResult.textContent = JSON.stringify(review.draft, null, 2);
    }
  }

  async function loadRecordings() {
    const review = state.recordingReview;
    stopRecordingPlayback();
    dom.recordingReviewStatus.textContent = "저장된 RGB-D manifest를 불러오는 중…";
    try {
      const [catalog, capabilities] = await Promise.all([
        api.listCameraRecordings(),
        api.recordingSkillDraftCapabilities(),
      ]);
      review.recordings = Array.isArray(catalog.recordings) ? catalog.recordings : [];
      review.capabilities = capabilities;
      review.loaded = true;
      review.selectedId = review.recordings.some(
        (recording) => recording.recording_id === review.selectedId,
      ) ? review.selectedId : review.recordings[0]?.recording_id || null;
      review.frameIndex = 0;
      dom.recordingSelect.replaceChildren();
      if (!review.recordings.length) {
        const option = create("option", { text: "완료된 RGB-D 녹화가 없습니다" });
        option.value = "";
        dom.recordingSelect.append(option);
        dom.recordingSelect.disabled = true;
        dom.recordingReviewStatus.textContent = "실시간 모니터링에서 모션 녹화를 먼저 완료해 주세요.";
      } else {
        review.recordings.forEach((recording) => {
          const option = create("option", {
            text: `${recording.recording_id} · ${recording.frame_count}프레임 · ${Number(recording.duration_s || 0).toFixed(1)}초`,
          });
          option.value = recording.recording_id;
          option.selected = recording.recording_id === review.selectedId;
          dom.recordingSelect.append(option);
        });
        dom.recordingSelect.disabled = false;
        dom.recordingReviewStatus.textContent = `${review.recordings.length}개 녹화 · RGB/Depth는 로컬에서만 재생됩니다.`;
      }
      if (catalog.invalid_recording_count) {
        dom.recordingReviewStatus.textContent += ` · 손상된 manifest ${catalog.invalid_recording_count}개 제외`;
      }
    } catch (error) {
      review.loaded = false;
      review.recordings = [];
      review.selectedId = null;
      dom.recordingReviewStatus.textContent = `녹화 목록 실패: ${errorText(error)}`;
    }
    renderRecordingReview();
  }

  function toggleRecordingPlayback() {
    const review = state.recordingReview;
    const recording = selectedRecording();
    if (!recording) return;
    if (review.playing) {
      stopRecordingPlayback();
      renderRecordingFrame();
      return;
    }
    const frameCount = Number(recording.frame_count || 0);
    if (frameCount < 1) return;
    if (review.frameIndex >= frameCount - 1) review.frameIndex = 0;
    review.playing = true;
    const framesPerSecond = Math.min(30, Math.max(1, Number(recording.recording_fps || 10)));
    review.playbackTimer = window.setInterval(() => {
      if (review.frameIndex >= frameCount - 1) {
        stopRecordingPlayback();
      } else {
        review.frameIndex += 1;
      }
      renderRecordingFrame();
    }, Math.round(1000 / framesPerSecond));
    renderRecordingFrame();
  }

  async function createRecordingSkillDraft(event) {
    event.preventDefault();
    if (!dom.recordingSkillForm.reportValidity()) return;
    const recording = selectedRecording();
    if (!recording) return;
    const review = state.recordingReview;
    review.loading = true;
    review.draft = null;
    renderRecordingReview();
    try {
      const result = await api.createRecordingSkillDraft({
        recordingId: recording.recording_id,
        nameHint: dom.draftSkillName.value.trim(),
        operatorInstruction: dom.draftInstruction.value.trim(),
        keyframeCount: Number(dom.draftKeyframeCount.value),
      });
      review.draft = {
        draft_id: result.draft_id,
        status: result.status,
        source_recording_id: result.source_recording_id,
        keyframe_indices: result.keyframe_indices,
        transport: result.transport,
        openai: result.openai,
        draft: result.draft,
        artifact_uri: result.artifact_uri,
        executable: result.executable,
        requires_pose_trajectory: result.requires_pose_trajectory,
      };
      const draftCatalog = await api.listSkillDrafts();
      state.drafts = (draftCatalog.drafts || []).map(normalizeDraft);
      dom.draftStatus.style.color = "var(--ok)";
      dom.draftStatus.textContent = result.openai?.mode === "live"
        ? result.transport?.fallback_used
          ? "GPT 분석 완료 · 첫 RGB 파일 1장 + 전체 엄지/검지 trace 전송 · Depth는 로컬 유지"
          : `GPT 분석 완료 · ${result.openai.model} · 첫 RGB 1장 + 전체 ${result.transport?.trace_frame_count || 0}프레임 손끝 trace 전송`
        : "Mock 분석 완료 · 실제 모드에서도 첫 RGB 1장과 전체 손끝 trace만 전송되고 Depth는 로컬에 남습니다.";
    } catch (error) {
      dom.draftStatus.textContent = `스킬 초안 생성 실패: ${errorText(error)}`;
      dom.draftStatus.style.color = "var(--danger)";
    } finally {
      review.loading = false;
    }
    renderRecordingReview();
  }

  function setCreateMode(mode) {
    state.editor.createMode = mode;
    if (mode === "recording" && !state.recordingReview.loaded) loadRecordings();
    renderCreateMode();
  }

  function addSelectedSkillBlock() {
    const primitive = primitiveByOperation(dom.blockOperationSelect.value);
    if (!primitive) return;
    initializeBlockly();
    const workspace = state.editor.workspace;
    if (!workspace) return;
    const editorBlock = {
      operation: primitive.operation_name,
      arguments: cloneValue(primitive.default_arguments || {}),
    };
    window.Blockly.Events.disable();
    let blocklyBlock;
    try {
      blocklyBlock = workspace.newBlock(blocklyTypeForOperation(primitive.operation_name));
      editorBlock.blockId = blocklyBlock.id;
      writeBlocklyBlockData(blocklyBlock, editorBlock);
      blocklyBlock.initSvg();
      blocklyBlock.render();
      const topBlocks = workspace.getTopBlocks(true);
      const root = topBlocks.find((block) => block.id !== blocklyBlock.id);
      let tail = root || null;
      while (tail?.getNextBlock()) tail = tail.getNextBlock();
      if (tail?.nextConnection && blocklyBlock.previousConnection) {
        tail.nextConnection.connect(blocklyBlock.previousConnection);
      } else {
        blocklyBlock.moveBy(48, 48);
      }
    } finally {
      window.Blockly.Events.enable();
    }
    state.editor.selectedBlocklyBlockId = blocklyBlock.id;
    blocklyBlock.select();
    syncBlocksFromBlockly();
    dom.blockEditorResult.hidden = true;
    renderSkillBlocks();
  }

  async function previewSkillBlocks() {
    const payload = blockEditorPayload();
    if (!payload || !state.editor.blocks.length) return;
    state.editor.loading = true;
    renderSkillBlocks();
    setBanner("순차 블록 schema·profile·anchor·compiler 검증 중…");
    try {
      const result = await api.previewSkillBlocks(payload);
      dom.blockEditorResult.hidden = false;
      dom.blockEditorResult.textContent = JSON.stringify(result, null, 2);
      setBanner(
        result.valid ? "블록 미리보기 검증 통과 · 아직 저장되지 않았습니다." : `블록 검증 실패: ${(result.errors || []).join("; ")}`,
        result.valid ? "ok" : "danger",
      );
    } catch (error) {
      setBanner(`블록 미리보기 실패: ${errorText(error)}`, "danger");
    } finally {
      state.editor.loading = false;
      renderSkillBlocks();
    }
  }

  async function createBlockCandidate() {
    const payload = blockEditorPayload();
    if (!payload || !state.editor.blocks.length) return;
    if (!window.confirm("부모/시연 없이 Mock-only 비활성 Candidate를 생성하고 전체 Mock 회귀 검증을 실행할까요?")) return;
    state.editor.loading = true;
    renderSkillBlocks();
    setBanner("순차 블록 Candidate 컴파일 및 Mock 회귀 검증 중…");
    try {
      const result = await api.createSkillBlockCandidate(payload);
      dom.blockEditorResult.hidden = false;
      dom.blockEditorResult.textContent = JSON.stringify(result, null, 2);
      await loadRegistry(
        result.mock_validation_passed
          ? `${payload.skill_id} Candidate Mock 검증 통과`
          : `${payload.skill_id} Candidate 저장됨 · Mock 검증 경고를 확인하세요.`,
      );
      state.selectedKey = `${result.candidate.skill_id}@${result.candidate.version}`;
      showPage("detail");
    } catch (error) {
      setBanner(`블록 Candidate 생성 실패: ${errorText(error)}`, "danger");
    } finally {
      state.editor.loading = false;
      renderSkillBlocks();
    }
  }

  function selectParameterNode() {
    const skill = selectedSkill();
    if (!skill) return;
    state.editor.parameterNodeId = dom.parameterNodeSelect.value;
    const node = skill.nodes.find((item) => item.nodeId === state.editor.parameterNodeId);
    state.editor.parameterArguments = cloneValue(node?.arguments || {});
    dom.parameterEditorResult.hidden = true;
    renderParameterEditor(skill);
  }

  async function createParameterCandidate() {
    const skill = selectedSkill();
    const node = skill?.nodes.find((item) => item.nodeId === state.editor.parameterNodeId);
    if (!skill || !node || !skill.checksum) return;
    if (!window.confirm("현재 버전은 그대로 두고, 이 인수로 새 child Candidate를 만들까요?")) return;
    state.editor.loading = true;
    renderParameterEditor(skill);
    setBanner(`${skill.id}@${skill.version} checksum 확인 및 child Candidate 생성 중…`);
    try {
      const result = await api.createSkillParameterCandidate(skill.id, skill.version, {
        expected_parent_checksum_sha256: skill.checksum,
        edits: [{
          node_id: node.nodeId,
          arguments: cloneValue(state.editor.parameterArguments),
        }],
      });
      dom.parameterEditorResult.hidden = false;
      dom.parameterEditorResult.textContent = JSON.stringify(result, null, 2);
      await loadRegistry(
        result.parent_unchanged
          ? `${skill.id} 부모 checksum 보존 · ${result.candidate.version} 생성 완료`
          : `${skill.id} parent 불변 검증 실패`,
      );
      state.selectedKey = `${result.candidate.skill_id}@${result.candidate.version}`;
      state.editor.parameterSkillKey = null;
      showPage("detail");
    } catch (error) {
      setBanner(`Parameter Candidate 생성 실패: ${errorText(error)}`, "danger");
    } finally {
      state.editor.loading = false;
      renderDetail();
    }
  }

  function renderAll() {
    renderConnection();
    renderTaskFlowCatalog();
    renderRegistry();
    if (state.page === "detail") renderDetail();
    if (state.page === "monitor") renderMonitor();
    if (state.page === "jog") renderJog();
    if (state.page === "aruco-experiment") renderArucoExperiment();
    if (state.page === "create") {
      renderCreateMode();
      if (state.editor.createMode === "recording") renderRecordingReview();
    }
    renderCamera();
  }

  async function refreshCameraStatus({ quiet = true } = {}) {
    if (state.apiStatus !== "connected") return;
    try {
      const payload = await api.cameraStatus();
      const previousRecordingId = state.camera.recordingId;
      state.camera = {
        ...state.camera,
        state: payload.state,
        recording: payload.recording,
        recordingId: payload.recording?.recording_id || previousRecordingId,
        lastError: payload.last_error,
        frameNumber: payload.frame_number,
        maximumTimestampSkewMs: payload.maximum_timestamp_skew_ms,
      };
      if (previousRecordingId && !payload.recording) {
        try {
          state.camera.lastRecording = await api.getCameraRecording(previousRecordingId);
          state.camera.recordingId = null;
        } catch {
          // The writer may still be finalizing; the next status poll retries.
        }
      }
      renderCamera();
    } catch (error) {
      if (!quiet) setBanner(`카메라 상태 확인 실패: ${errorText(error)}`, "danger");
    }
  }

  async function refreshHandeyeCalibrationStatus({ quiet = true } = {}) {
    if (state.apiStatus !== "connected") return;
    try {
      const payload = await api.handeyeCalibrationStatus();
      state.calibration.capabilities = payload.capabilities || null;
      state.calibration.session = payload.session || null;
      state.calibration.latestResult = payload.latest_result || null;
      state.calibration.legacyTransform = payload.legacy_transform || null;
      state.calibration.lastError = null;
    } catch (error) {
      state.calibration.lastError = errorText(error);
      if (!quiet) setBanner(`보정 상태 확인 실패: ${errorText(error)}`, "danger");
    }
    renderCalibration();
  }

  async function startHandeyeCalibration() {
    const accepted = window.confirm(
      "실제 M0609가 움직입니다.\n\n"
      + "• J1/J2가 0°에 있고 Calibration 중 고정됨\n"
      + "• 시작하면 J3–J6가 먼저 [90,0,90,0]으로 자동 이동함\n"
      + "• 10×7/25mm Chessboard가 고정됨\n"
      + "• 작업공간이 비어 있고 E-stop 사용 가능\n"
      + "• J1/J2 고정 pose 계획을 검토함\n\n"
      + "위 조건을 모두 확인하고 Calibration을 시작할까요?",
    );
    if (!accepted) return;
    state.calibration.loading = true;
    state.calibration.lastError = null;
    renderCalibration();
    try {
      const payload = await api.startHandeyeCalibration();
      state.calibration.capabilities = payload.capabilities || null;
      state.calibration.session = payload.session || null;
      state.calibration.latestResult = payload.latest_result || null;
      setBanner("Eye-in-hand calibration 시작 · 로봇 주변에 접근하지 마세요.");
    } catch (error) {
      state.calibration.lastError = errorText(error);
      setBanner(`Calibration 시작 실패: ${errorText(error)}`, "danger");
    } finally {
      state.calibration.loading = false;
    }
    renderCalibration();
  }

  async function abortHandeyeCalibration() {
    if (!window.confirm("Calibration을 즉시 중단하고 로봇 stop을 요청할까요?")) return;
    try {
      const payload = await api.abortHandeyeCalibration("ui_operator_request");
      state.calibration.session = payload.session || state.calibration.session;
      setBanner("Calibration 중단 요청을 전송했습니다.", "danger");
    } catch (error) {
      setBanner(`Calibration 중단 실패: ${errorText(error)}`, "danger");
    }
    renderCalibration();
  }

  async function importLegacyHandeyeNpy() {
    const accepted = window.confirm(
      "원본 T_gripper2camera.npy는 덮어쓰지 않고 별도 artifact로 복사합니다.\n\n"
      + "현재 활성 TCP와 flange pose를 읽어 T_flange_camera 후보로 변환하고 기존 Chessboard 관측으로 검증합니다.\n"
      + "결과는 Candidate/Mock 증거일 뿐 실제 로봇 실행이나 ROS TF를 승인하지 않습니다. 계속할까요?",
    );
    if (!accepted) return;
    state.calibration.loading = true;
    state.calibration.lastError = null;
    renderCalibration();
    try {
      const payload = await api.importLegacyHandeyeNpy();
      state.calibration.capabilities = payload.capabilities || null;
      state.calibration.legacyTransform = payload.legacy_transform || null;
      const result = state.calibration.legacyTransform;
      setBanner(
        result?.passed
          ? "NPY TF 후보 검증 통과 · 별도 물리 검증 전까지 Candidate 전용"
          : `NPY TF 후보 검증 실패 · ${(result?.failures || []).join("; ")}`,
        result?.passed ? "ok" : "danger",
      );
      await loadRegistry();
    } catch (error) {
      state.calibration.lastError = errorText(error);
      setBanner(`NPY TF 복사·검증 실패: ${errorText(error)}`, "danger");
    } finally {
      state.calibration.loading = false;
    }
    renderCalibration();
  }

  async function startCameraPreview() {
    state.camera.state = "starting";
    state.camera.lastError = null;
    setBanner("RealSense RGB-D 시작 중…");
    renderCamera();
    try {
      const payload = await api.startCameraPreview();
      state.camera = {
        ...state.camera,
        state: payload.state,
        recording: payload.recording,
        lastError: payload.last_error,
        frameNumber: payload.frame_number,
        maximumTimestampSkewMs: payload.maximum_timestamp_skew_ms,
      };
      setBanner("RealSense RGB + 정렬 Depth 스트리밍 시작", "ok");
    } catch (error) {
      state.camera.state = "error";
      state.camera.lastError = errorText(error);
      setBanner(`카메라 시작 실패: ${errorText(error)}`, "danger");
    }
    renderCamera();
  }

  async function stopCameraPreview() {
    setBanner("RealSense 종료 중…");
    try {
      const payload = await api.stopCameraPreview();
      state.camera.state = payload.state;
      state.camera.recording = null;
      state.camera.frameNumber = null;
      setBanner("RealSense 카메라를 종료했습니다.", "ok");
    } catch (error) {
      setBanner(`카메라 종료 실패: ${errorText(error)}`, "danger");
    }
    renderCamera();
  }

  async function startCameraRecording() {
    const maximumDurationS = Number(dom.recordDuration.value);
    if (!Number.isFinite(maximumDurationS) || maximumDurationS < 1 || maximumDurationS > 60) {
      setBanner("녹화 최대 시간은 1–60초여야 합니다.", "danger");
      return;
    }
    try {
      const recording = await api.startCameraRecording(maximumDurationS);
      state.camera.recording = recording;
      state.camera.recordingId = recording.recording_id;
      state.camera.lastRecording = null;
      setBanner(`RGB-D 모션 녹화 시작 · 최대 ${maximumDurationS}초`);
    } catch (error) {
      setBanner(`녹화 시작 실패: ${errorText(error)}`, "danger");
    }
    renderCamera();
  }

  async function stopCameraRecording() {
    const recordingId = state.camera.recording?.recording_id || state.camera.recordingId;
    if (!recordingId) return;
    try {
      setBanner("RGB-D 녹화 파일 정리 중…");
      const recording = await api.stopCameraRecording(recordingId);
      state.camera.recording = null;
      state.camera.recordingId = null;
      state.camera.lastRecording = recording;
      state.recordingReview.loaded = false;
      setBanner(`RGB-D 녹화 완료 · ${recording.frame_count}프레임`, "ok");
    } catch (error) {
      setBanner(`녹화 중지 실패: ${errorText(error)}`, "danger");
    }
    renderCamera();
  }

  async function loadRegistry(message = null) {
    state.apiStatus = "connecting";
    renderConnection();
    setBanner("FastAPI 연결 확인 중…");
    try {
      const [, registry, draftCatalog, editorCatalog] = await Promise.all([
        api.health(),
        api.listSkills(),
        api.listSkillDrafts(),
        api.skillEditorCatalog(),
      ]);
      state.skills = (registry.skills || []).map(normalizeSkill);
      state.drafts = (draftCatalog.drafts || []).map(normalizeDraft);
      state.taskFlowCatalog = registry.task_flow_catalog || { objects: [] };
      state.editor.catalog = Array.isArray(editorCatalog.primitives)
        ? editorCatalog.primitives
        : [];
      state.selectedKey = state.skills.some((skill) => skill.key === state.selectedKey)
        ? state.selectedKey
        : state.skills[0]?.key || null;
      state.apiStatus = "connected";
      setBanner(
        message || `FastAPI 연결됨 · 등록 버전 ${state.skills.length}개 · 분석 초안 ${state.drafts.length}개`,
        "ok",
      );
    } catch (error) {
      state.apiStatus = "error";
      state.skills = [];
      state.drafts = [];
      state.taskFlowCatalog = { objects: [] };
      state.selectedKey = null;
      state.selectedDraftId = null;
      setBanner(`API 연결 실패: ${errorText(error)}`, "danger");
    }
    renderAll();
  }

  function reviewSelectedDraftRecording() {
    const draft = selectedDraft();
    if (!draft?.sourceRecordingId) return;
    state.recordingReview.selectedId = draft.sourceRecordingId;
    state.recordingReview.loaded = false;
    state.recordingReview.draft = null;
    showPage("create");
  }

  function beginDraftGeometryTeaching(mode) {
    const draft = selectedDraft();
    if (!draft?.sourceRecordingId) return;
    const surfaceHint = state.recordingReview.surfaceHint;
    const useSurfaceHint = mode === "surface_calibration"
      && surfaceHint?.draftId === draft.draftId
      && surfaceHint?.recordingId === draft.sourceRecordingId
      && Array.isArray(surfaceHint.points)
      && surfaceHint.points.length === 3;
    state.geometryTeaching = {
      mode,
      draftId: draft.draftId,
      points: useSurfaceHint
        ? surfaceHint.points.map((point) => ({ ...point }))
        : [],
      samples: [],
      busy: false,
    };
    state.recordingReview.selectedId = draft.sourceRecordingId;
    state.recordingReview.loaded = false;
    state.recordingReview.frameIndex = useSurfaceHint ? surfaceHint.frameIndex : 0;
    state.recordingReview.draft = null;
    stopRecordingPlayback();
    showPage("create");
    setBanner(
      mode === "surface_calibration"
        ? useSurfaceHint
          ? "RANSAC 보조점 O·X·Y를 확인한 뒤 필요하면 조정하고 직접 확정하세요."
          : "녹화 RGB에서 표면 원점, +X, +Y를 지정하세요."
        : "서로 다른 프레임에서 두 fingertip을 지정해 TCP 경로를 만드세요.",
    );
    renderRecordingReview();
  }

  function cancelGeometryTeaching() {
    state.geometryTeaching = {
      mode: null,
      draftId: null,
      points: [],
      samples: [],
      busy: false,
    };
    renderRecordingReview();
  }

  function handleRecordedRgbClick(event) {
    const teaching = state.geometryTeaching;
    if (!teaching.mode || teaching.busy) return;
    const point = geometryPointFromEvent(event);
    if (!point) {
      setBanner("실제 RGB 이미지 영역 안을 클릭해 주세요.", "danger");
      return;
    }
    const maximumPoints = teaching.mode === "surface_calibration" ? 3 : 2;
    if (teaching.points.length >= maximumPoints) return;
    teaching.points.push(point);
    renderGeometryTeaching();
  }

  function resetCurrentGeometryPoints() {
    state.geometryTeaching.points = [];
    renderGeometryTeaching();
  }

  function addCurrentPathSample() {
    const teaching = state.geometryTeaching;
    if (teaching.mode !== "manual_tcp_path" || teaching.points.length !== 2) return;
    const frameIndex = state.recordingReview.frameIndex;
    const sample = {
      frame_index: frameIndex,
      jaw_tip_a_px: teaching.points[0],
      jaw_tip_b_px: teaching.points[1],
    };
    const existingIndex = teaching.samples.findIndex((item) => item.frame_index === frameIndex);
    if (existingIndex >= 0) teaching.samples[existingIndex] = sample;
    else teaching.samples.push(sample);
    teaching.samples.sort((left, right) => left.frame_index - right.frame_index);
    teaching.points = [];
    renderGeometryTeaching();
    setBanner(`TCP 경로 샘플 ${teaching.samples.length}개 저장됨`, "ok");
  }

  async function saveGeometryEvidence() {
    const teaching = state.geometryTeaching;
    if (!teaching.mode || !teaching.draftId || teaching.busy) return;
    if (teaching.mode === "surface_calibration"
      && (!dom.surfaceAnchorId.reportValidity() || teaching.points.length !== 3)) return;
    if (teaching.mode === "manual_tcp_path" && teaching.samples.length < 2) return;
    teaching.busy = true;
    renderGeometryTeaching();
    try {
      if (teaching.mode === "surface_calibration") {
        await api.calibrateDraftSurface(teaching.draftId, {
          frame_index: state.recordingReview.frameIndex,
          surface_anchor_id: dom.surfaceAnchorId.value.trim(),
          origin_px: teaching.points[0],
          positive_x_px: teaching.points[1],
          positive_y_px: teaching.points[2],
          operator_confirmed: true,
        });
        cancelGeometryTeaching();
        await loadRegistry("수동 3점 camera → task-plane TF 보정 완료 · 실제 로봇 TF는 별도 검증 필요");
      } else {
        await api.createDraftTcpTrajectory(teaching.draftId, {
          method: "manual_two_fingertip",
          annotations: teaching.samples,
          operator_confirmed: true,
        });
        cancelGeometryTeaching();
        await loadRegistry("표면 상대 두 fingertip TCP 경로 생성 완료");
      }
      showPage("skills");
    } catch (error) {
      teaching.busy = false;
      setBanner(`로컬 RGB-D 증거 생성 실패: ${errorText(error)}`, "danger");
      renderGeometryTeaching();
    }
  }

  async function autoExtractSurfacePlane() {
    const draft = selectedDraft();
    if (!draft) return;
    const confirmed = window.confirm(
      "GPT의 작업대 영역은 힌트로만 사용하고 원본 aligned Depth로 보조 평면을 계산합니다. 이 결과는 최종 TF가 아니며, 이후 원점·+X·+Y를 직접 지정해야 합니다. 계속할까요?",
    );
    if (!confirmed) return;
    setBanner("원본 Depth에서 수동 3점 선택을 보조할 작업대 평면 힌트 계산 중…");
    dom.autoSurfaceCalibration.disabled = true;
    try {
      const result = await api.autoCalibrateDraftSurface(draft.draftId, {
        frame_index: null,
        surface_anchor_id: "teaching_surface",
        operator_confirmed: true,
      });
      const assist = result.manual_click_assist || {};
      const suggestedPoints = [
        assist.origin_px,
        assist.positive_x_px,
        assist.positive_y_px,
      ];
      const validSuggestedPoints = suggestedPoints.every((point) => point
        && Number.isFinite(Number(point.x_px))
        && Number.isFinite(Number(point.y_px)));
      state.recordingReview.surfaceHint = validSuggestedPoints
        ? {
          draftId: draft.draftId,
          recordingId: draft.sourceRecordingId,
          frameIndex: Number(result.frame_index || 0),
          points: suggestedPoints.map((point) => ({
            x_px: Number(point.x_px),
            y_px: Number(point.y_px),
          })),
        }
        : null;
      const inlier = Number(result.diagnostics?.inlier_ratio || 0);
      await loadRegistry(
        `Depth 평면 보조 힌트 생성 · inlier ${(inlier * 100).toFixed(1)}% · 수동 3점에서 제안점을 확인·조정하세요`,
      );
    } catch (error) {
      setBanner(`Depth 평면 자동 추출 실패: ${errorText(error)} · 3점 표면 TF 보정을 사용하세요.`, "danger");
      renderDraftInspector();
    }
  }

  async function autoExtractTcpPath() {
    const draft = selectedDraft();
    if (!draft) return;
    const confirmed = window.confirm(
      "MediaPipe가 전체 녹화 RGB에서 검출한 엄지(4)·검지(8)를 각자의 aligned Depth로 3D 복원합니다. 3 cm 임계값과 3-frame 안정화는 로컬 설정만 사용합니다. 결과는 Candidate/Mock 전용입니다. 계속할까요?",
    );
    if (!confirmed) return;
    setBanner("MediaPipe 전체 프레임 + 로컬 aligned Depth로 TCP 경로 복원 중…");
    dom.autoTcpPath.disabled = true;
    try {
      const result = await api.createDraftTcpTrajectory(draft.draftId, {
        method: "mediapipe_rgbd",
        annotations: [],
        operator_confirmed: true,
      });
      const review = state.recordingReview;
      review.fingerTrackingRecordingId = result.recording_id || draft.sourceRecordingId;
      review.fingerObservations = Array.isArray(result.finger_observations)
        ? result.finger_observations
        : [];
      review.fingerTrackingSettings = result.finger_tracking_settings || null;
      review.selectedId = draft.sourceRecordingId;
      review.frameIndex = 0;
      await loadRegistry(
        `MediaPipe TCP 경로 ${result.quality?.sample_count || 0}개 · overlay ${review.fingerObservations.length}프레임`,
      );
      showPage("create");
      renderRecordingReview();
    } catch (error) {
      setBanner(`MediaPipe RGB-D 경로 적용 실패: ${errorText(error)} · optional dependency 또는 depth 정렬 상태를 확인하세요.`, "danger");
      renderDraftInspector();
    }
  }

  async function registerSelectedDraftCandidate() {
    const draft = selectedDraft();
    if (!draft?.readiness.can_register_candidate) return;
    const confirmed = window.confirm(
      "이 Candidate는 surface-relative 경로의 컴파일/Mock 검증만 수행하며 실제 로봇 하드웨어 검증은 포함하지 않습니다. 등록할까요?",
    );
    if (!confirmed) return;
    dom.registerDraftCandidate.disabled = true;
    setBanner(`${draft.suggestedSkillId} Candidate 생성 및 Mock 회귀 검증 중…`);
    try {
      const result = await api.registerDraftCandidate(draft.draftId);
      await loadRegistry(
        result.mock_validation_passed
          ? `${result.skill_id}@${result.version} Candidate Mock 검증 통과`
          : `${result.skill_id}@${result.version} Candidate Mock 검증 실패`,
      );
    } catch (error) {
      setBanner(`Candidate 등록 실패: ${errorText(error)}`, "danger");
      renderDraftInspector();
    }
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
  dom.deactivateSkill.addEventListener("click", () => {
    const skill = selectedSkill();
    if (skill) deactivateActiveSkill(skill, dom.deactivateSkill);
  });
  dom.closeDraftInspector.addEventListener("click", () => {
    state.selectedDraftId = null;
    renderRegistry();
  });
  dom.reviewDraftRecording.addEventListener("click", reviewSelectedDraftRecording);
  dom.autoSurfaceCalibration.addEventListener("click", autoExtractSurfacePlane);
  dom.startSurfaceCalibration.addEventListener("click", () => {
    beginDraftGeometryTeaching("surface_calibration");
  });
  dom.startPathTeaching.addEventListener("click", () => {
    beginDraftGeometryTeaching("manual_tcp_path");
  });
  dom.autoTcpPath.addEventListener("click", autoExtractTcpPath);
  dom.registerDraftCandidate.addEventListener("click", registerSelectedDraftCandidate);
  dom.estopRun.addEventListener("click", () => abortRun("emergency_stop"));
  dom.abortRun.addEventListener("click", () => abortRun("operator_request"));
  dom.startCamera.addEventListener("click", startCameraPreview);
  dom.stopCamera.addEventListener("click", stopCameraPreview);
  dom.startRecording.addEventListener("click", startCameraRecording);
  dom.stopRecording.addEventListener("click", stopCameraRecording);
  dom.startHandeyeCalibration.addEventListener("click", startHandeyeCalibration);
  dom.abortHandeyeCalibration.addEventListener("click", abortHandeyeCalibration);
  dom.importLegacyHandeye.addEventListener("click", importLegacyHandeyeNpy);
  dom.jogEnable.addEventListener("click", enableJog);
  dom.jogRefresh.addEventListener("click", () => refreshJogStatus({ quiet: false }));
  dom.jogStop.addEventListener("click", stopJog);
  dom.jogLoadCurrent.addEventListener("click", loadCurrentJogTargets);
  dom.jogMoveJ.addEventListener("click", executeJogMoveJ);
  dom.arucoExperimentEnable.addEventListener("click", enableArucoExperiment);
  dom.arucoExperimentRefresh.addEventListener(
    "click",
    () => refreshArucoExperimentStatus({ quiet: false }),
  );
  dom.arucoExperimentStop.addEventListener("click", stopArucoExperiment);
  dom.arucoMoveReference.addEventListener("click", moveArucoReference);
  dom.arucoMoveZTest.addEventListener("click", moveArucoPlaneZTest);
  dom.arucoWidthModel.addEventListener("change", renderArucoExperiment);
  dom.recordingSelect.addEventListener("change", () => {
    stopRecordingPlayback();
    cancelGeometryTeaching();
    state.recordingReview.selectedId = dom.recordingSelect.value || null;
    state.recordingReview.frameIndex = 0;
    state.recordingReview.draft = null;
    dom.draftStatus.style.color = "";
    renderRecordingReview();
  });
  dom.refreshRecordings.addEventListener("click", loadRecordings);
  dom.toggleRecordingPlayback.addEventListener("click", toggleRecordingPlayback);
  dom.recordingFrameIndex.addEventListener("input", () => {
    stopRecordingPlayback();
    state.geometryTeaching.points = [];
    state.recordingReview.frameIndex = Number(dom.recordingFrameIndex.value);
    renderRecordingFrame();
  });
  dom.recordedRgbFrame.addEventListener("click", handleRecordedRgbClick);
  dom.recordedRgbFrame.addEventListener("load", renderGeometryMarkers);
  dom.resetGeometryPoints.addEventListener("click", resetCurrentGeometryPoints);
  dom.addPathSample.addEventListener("click", addCurrentPathSample);
  dom.saveGeometryEvidence.addEventListener("click", saveGeometryEvidence);
  dom.cancelGeometryTeaching.addEventListener("click", cancelGeometryTeaching);
  dom.recordingSkillForm.addEventListener("submit", createRecordingSkillDraft);
  dom.createRecordingTab.addEventListener("click", () => setCreateMode("recording"));
  dom.createBlockTab.addEventListener("click", () => setCreateMode("block"));
  dom.addSkillBlock.addEventListener("click", addSelectedSkillBlock);
  dom.previewSkillBlocks.addEventListener("click", previewSkillBlocks);
  dom.createBlockCandidate.addEventListener("click", createBlockCandidate);
  dom.parameterNodeSelect.addEventListener("change", selectParameterNode);
  dom.createParameterCandidate.addEventListener("click", createParameterCandidate);
  window.addEventListener("resize", renderGeometryMarkers);

  const normalLogo = document.getElementById("normal-logo");
  const shinyLogo = document.getElementById("shiny-logo");

  function spawnSparkle(side) {
    const sparkle = document.createElement("img");

    sparkle.src = "assets/sparkle.png";
    sparkle.className = "sparkle";

    // 왼쪽 / 오른쪽 시작 위치
    const x = side === "left"
      ? 18 + Math.random() * 8
      : 62 + Math.random() * 8;

    const y = 25 + Math.random() * 10;

    sparkle.style.left = `${x}px`;
    sparkle.style.top = `${y}px`;

    // 랜덤 크기
    const size = 12 + Math.random() * 14;
    sparkle.style.width = `${size}px`;
    sparkle.style.height = `${size}px`;

    // 랜덤 이동
    sparkle.style.setProperty("--dx", `${(Math.random() - 0.5) * 30}px`);
    sparkle.style.setProperty("--dy", `${-20 - Math.random() * 20}px`);

    // 랜덤 회전
    sparkle.style.setProperty("--rot", `${Math.random() * 360}deg`);

    // 랜덤 속도
    sparkle.style.setProperty(
      "--duration",
      `${0.5 + Math.random() * 0.4}s`
    );

    document.querySelector(".brand").appendChild(sparkle);

    sparkle.addEventListener("animationend", () => sparkle.remove());
  }

  function toggleShinyLogo() {
    const shiny = shinyLogo.hidden;

    shinyLogo.hidden = !shiny;
    normalLogo.hidden = shiny;

    // 일반 → Shiny로 바뀔 때만 반짝이 생성
    if (shiny) {
      spawnSparkle("left");

      setTimeout(() => {
        spawnSparkle("right");
      }, 120);
    }
  }
  normalLogo?.addEventListener("dblclick", toggleShinyLogo);
  shinyLogo?.addEventListener("dblclick", toggleShinyLogo);

  renderAll();
  loadRegistry().then(() => Promise.all([
    refreshCameraStatus(),
    refreshHandeyeCalibrationStatus(),
    refreshJogStatus({ quiet: true }),
    refreshArucoExperimentStatus({ quiet: true }),
  ])).catch(() => undefined);
  window.setInterval(() => {
    if (state.camera.state === "streaming" || state.camera.recordingId) {
      refreshCameraStatus();
    }
    if (["starting", "moving_to_reference", "running", "aborting"]
      .includes(state.calibration.session?.status)) {
      refreshHandeyeCalibrationStatus();
    }
    if (state.page === "jog" || state.jog.status?.enabled) {
      refreshJogStatus({ quiet: true });
    }
    if (state.page === "aruco-experiment" || state.arucoExperiment.status?.enabled) {
      refreshArucoExperimentStatus({ quiet: true });
    }
  }, 1000);
})();
