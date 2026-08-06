"use strict";

(() => {
  const state = {
    page: "skills",
    filter: "all",
    apiStatus: "connecting",
    skills: [],
    drafts: [],
    selectedKey: null,
    selectedDraftId: null,
    expandedNode: null,
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
    },
    calibration: {
      capabilities: null,
      session: null,
      latestResult: null,
      legacyTransform: null,
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
    },
    geometryTeaching: {
      mode: null,
      draftId: null,
      points: [],
      samples: [],
      busy: false,
    },
    manualSkill: {
      nodes: [],
      loading: false,
    },
    primitiveCatalog: [],
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
    runStatus: element("run-status"),
    runTitle: element("run-title"),
    runSteps: element("run-steps"),
    estopChip: element("estop-chip"),
    forceChip: element("force-chip"),
    workspaceChip: element("workspace-chip"),
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
    estopRun: element("estop-run"),
    abortRun: element("abort-run"),
    manualSkillForm: element("manual-skill-form"),
    manualSkillId: element("manual-skill-id"),
    manualSkillName: element("manual-skill-name"),
    manualSkillDescription: element("manual-skill-description"),
    manualSkillStatus: element("manual-skill-status"),
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

    const skillNames = {
      pick_tool: "공구 잡기",
      lift_tool: "공구 들어올리기",
      move_tool: "공구 이동하기",
      place_tool: "공구 내려놓기",
      insert_tool: "공구 삽입하기",
      hand_over_tool: "공구 전달하기",
      pick_object: "물체 잡기",
      lift_object: "물체 들어올리기",
      place_object: "물체 내려놓기",
    };

    const displayName =
      skillNames[row.skill_id]
      || skillNames[graph.skill_id]
      || row.description
      || graph.description
      || row.skill_id;

    let uiState = "candidate";
    if (row.status === "active") uiState = "active";
    else if (row.validation_status === "passed") uiState = "tested";

    return {
      key: `${row.skill_id}@${row.version}`,
      id: row.skill_id,
      displayName,
      version: row.version,
      uiState,
      validationStatus: row.validation_status,
      status: row.status,
      description: row.description || graph.description || "",
      nodes: nodes.map((node) => ({
        nodeId: node.node_id,
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
      pairCount: Number(transport.keyframe_pair_count || 0),
      transport,
      tcp,
      sceneObservation,
      readiness,
      promotionEvidence: row.promotion_evidence || {},
      createdAtNs: Number(row.created_at_ns || 0),
    };
  }

  function renderManualNodes() {
    dom.manualNodeList.replaceChildren();

    state.manualSkill.nodes.forEach((node, index) => {
      const item = create("div", {
        className: "manual-node",
      });

      item.append(
        create("strong", {
          text: `${index + 1}. ${node.operation}`,
        }),
      );

      const remove = create("button", {
        className: "button danger",
        text: "삭제",
      });

      remove.type = "button";

      remove.addEventListener("click", () => {
        state.manualSkill.nodes.splice(index, 1);
        renderManualNodes();
      });

      item.append(remove);
      dom.manualNodeList.append(item);
    });
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
    if (page === "create") {
      renderRecordingReview();
      if (!state.recordingReview.loaded) loadRecordings();
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
      const evidenceCount = draft.pairCount || draft.keyframeCount;
      meta.append(document.createTextNode(
        `${evidenceCount}개 RGB-D 쌍 · 신뢰도 ${draft.confidence.toFixed(2)}`,
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
      summary.append(create("h2", { text: skill.displayName || skill.id }));
      const meta = create("div", { className: "meta" });
      meta.append(document.createTextNode(`v${skill.version} · 노드 ${skill.nodes.length}개`));
      meta.append(create("span", { className: `tag ${skill.uiState}`, text: tagText(skill.uiState) }));
      summary.append(meta);

      const actions = create("div", { className: "actions" });
      const detail = create("button", { className: "button", text: "상세" });
      const deleteButton = create("button", {
        className: "button danger",
        text: "삭제",
      });

      deleteButton.type = "button";

      deleteButton.addEventListener("click", async () => {
        const confirmed = window.confirm(
          `${skill.id} v${skill.version}을 삭제하시겠습니까?\n\n삭제한 스킬은 복구할 수 없습니다.`,
        );

        if (!confirmed) return;

        deleteButton.disabled = true;
        setBanner(`${skill.id} v${skill.version} 삭제 중…`);

        try {
          await api.deleteSkill(skill.id);

          if (state.selectedKey === skill.key) {
            state.selectedKey = null;
          }

          await loadRegistry(`${skill.id} v${skill.version} 삭제 완료`);
        } catch (error) {
          deleteButton.disabled = false;
          setBanner(`스킬 삭제 실패: ${errorText(error)}`, "danger");
        }
      });

      detail.type = "button";
      detail.addEventListener("click", () => {
        state.selectedKey = skill.key;
        showPage("detail");
      });

      actions.append(detail, deleteButton);

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

  function renderDraftInspector() {
    const draft = selectedDraft();
    dom.draftInspector.hidden = !draft;
    if (!draft) return;
    dom.draftInspectorTitle.textContent = draft.suggestedSkillId;
    dom.draftInspectorSummary.textContent = draft.taskDescription || draft.observedSummary;
    dom.draftEvidence.replaceChildren();
    const evidence = [
      {
        label: draft.pairCount > 0 ? `RGB-D ${draft.pairCount}쌍` : "RGB-D 재분석 필요",
        passed: draft.pairCount > 0,
      },
      {
        label: draft.tcp?.trajectory_status
          ? `GPT TCP ${draft.tcp.trajectory_status} · 유효 ${draft.tcp.valid_landmark_frame_count || 0}프레임`
          : draft.tcp?.detected ? "두 손가락 TCP 프록시 관찰됨" : "TCP 재분석 필요",
        passed: Boolean(draft.tcp?.usable_for_local_depth_path),
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
      { label: "실행 좌표 없음", passed: false },
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
        text: check.passed ? "✓" : advisory ? "i" : "!",
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
    const calibrationReady = Boolean(checkById.get("calibrated_transform")?.passed);
    const trajectoryReady = Boolean(checkById.get("trusted_pose_trajectory")?.passed);
    const candidateReady = Boolean(checkById.get("mock_validation")?.passed);
    dom.autoSurfaceCalibration.disabled = false;
    dom.autoSurfaceCalibration.textContent = calibrationReady
      ? "Depth 평면 다시 추출" : "Depth 평면 자동 추출";
    dom.startSurfaceCalibration.disabled = false;
    dom.startSurfaceCalibration.textContent = calibrationReady ? "표면 TF 다시 보정" : "표면 TF 보정";
    dom.startPathTeaching.disabled = !calibrationReady;
    dom.startPathTeaching.title = calibrationReady
      ? "녹화 프레임에서 두 fingertip을 직접 지정"
      : "먼저 camera → surface TF를 보정하세요.";
    dom.autoTcpPath.disabled = !calibrationReady;
    dom.autoTcpPath.title = calibrationReady
      ? "GPT 정규화 fingertip 좌표를 로컬 aligned depth로 3D 복원"
      : "먼저 camera → surface TF를 보정하세요.";
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

  function renderDetail() {
    const skill = selectedSkill();
    dom.detailNodes.replaceChildren();
    dom.validateSkill.disabled =
      !skill || state.apiStatus !== "connected";

    dom.activateSkill.disabled =
      !skill ||
      state.apiStatus !== "connected" ||
      skill.uiState === "active";

    if (!skill) {
      dom.detailTitle.textContent = "스킬을 선택하세요";
      dom.detailSubtitle.textContent =
        "스킬 관리 화면에서 버전을 선택해 주세요.";
      return;
    }

    dom.detailTitle.textContent =
      skill.displayName || skill.id;

    dom.detailSubtitle.textContent =
      `v${skill.version} · 노드 ${skill.nodes.length}개`;

    skill.nodes.forEach((skillNode, index) => {
      const block = create("div", {
        className: `skill-node-block ${skillNode.status}`.trim(),
      });
      const expanded = state.expandedNode === skillNode.id;
      // =========================
      // 헤더
      // =========================

      const header = create("div", {
        className: "skill-node-header",
      });

      const titleArea = create("div");

      titleArea.append(
        create("span", {
          className: "skill-node-number",
          text: `${index + 1}`,
        }),
      );

      titleArea.append(
        create("strong", {
          className: "skill-node-operation",
          text: skillNode.operation,
        }),
      );

      header.append(titleArea);

      const arrow = create("span", {
        className: "skill-node-arrow",
        text: expanded ? "▲" : "▼",
      });
      header.append(arrow);

      const body = create("div", {
        className: "skill-node-body",
      });

      body.hidden = !expanded;

      header.style.cursor = "pointer";
      header.addEventListener("click", () => {

          if (state.expandedNode === skillNode.id) {
              state.expandedNode = null;
          } else {
              state.expandedNode = skillNode.id;
          }

          renderDetail();
      });

      // =========================
      // 파라미터 영역
      // =========================
      const parameters = create("div", {
        className: "skill-node-parameters",
      });
      parameters.hidden = !expanded;

      const args = skillNode.arguments || {};
      const summary = create("div", {
        className: "skill-node-summary",
      });  
      Object.entries(args).forEach(([key, value]) => {
        let text = value;
        if (typeof value === "object" && value !== null) {
          if (value.anchor_id) {
            text = value.anchor_id;
          } else {
            text = JSON.stringify(value);
          }
        }
        summary.append(
          create("div", {
            className: "skill-node-summary-item",
            text: `${key} : ${text}`,
          })
        );
      });

      const primitive = getPrimitiveMetadata(skillNode.operation);

      const parameterSchema =
        primitive?.typed_parameter_schema ||
        primitive?.parameter_schema ||
        {};

      const properties =
        parameterSchema.properties || {};

      const definitions =
        parameterSchema.$defs ||
        parameterSchema.definitions ||
        {};

      // 실제 저장된 arguments를 기준으로 표시
      const entries = Object.entries(args);

      if (!entries.length) {
        parameters.append(
          create("div", {
            className: "empty-parameters",
            text: "파라미터 없음",
          }),
        );
      }

      entries.forEach(([name, value]) => {
      const rawSchema = properties[name] || {};

      const schema = resolveParameterSchema(
        rawSchema,
        definitions,
      );

      const row = createParameterEditor(
        name,
        schema,
        value,
      );

      const input = row.querySelector(
        "[data-parameter-name]",
      );

      if (!input) {
        parameters.append(row);
        return;
      }

      if (schema.type === "boolean") {
        input.addEventListener("change", () => {
          saveNodeParameter(
            skill,
            skillNode,
            name,
            input.checked,
          );
        });
      } else {
        input.addEventListener("change", () => {
          let newValue = input.value;

          if (schema.type === "number") {
            newValue = Number(input.value);

            if (!Number.isFinite(newValue)) {
              setBanner(
                `${name}은 숫자여야 합니다.`,
                "danger",
              );
              return;
            }
          }

          else if (schema.type === "integer") {
            newValue =
              Number.parseInt(input.value, 10);

            if (!Number.isFinite(newValue)) {
              setBanner(
                `${name}은 정수여야 합니다.`,
                "danger",
              );
              return;
            }
          }

          else if (
            typeof value === "object" &&
            value !== null
          ) {
            try {
              newValue = JSON.parse(input.value);
            } catch {
              setBanner(
                `${name}의 JSON 형식이 올바르지 않습니다.`,
                "danger",
              );
              return;
            }
          }

          saveNodeParameter(
            skill,
            skillNode,
            name,
            newValue,
          );
        });
      }

      parameters.append(row);
    });


      // =========================
      // 버튼
      // =========================

      const actions = create("div", {
        className: "skill-node-actions",
      });
      actions.hidden = !expanded;

      const saveButton = create("button", {
        className: "button primary",
        text: "파라미터 적용",
      });

      saveButton.type = "button";

      saveButton.addEventListener("click", () => {
        const success = updateSkillNodeParameters(
          skillNode,
          parameters,
        );

        if (success) {
          setBanner(
            `${index + 1}번 블록 파라미터가 수정되었습니다.`,
            "ok",
          );

          renderDetail();
        }
      });

      actions.append(saveButton);


      // =========================
      // 최종 블록
      // =========================

      body.append(
        summary,
        parameters,
        actions,
      );

      block.append(
        header,
        body,
      );

      dom.detailNodes.append(block);
    });
  }

  function updateSkillNodeParameters(skillNode, container) {
    const inputs = container.querySelectorAll(
      "[data-parameter-name]",
    );

    try {
      inputs.forEach((input) => {
        const name = input.dataset.parameterName;
        const original = skillNode.arguments[name];

        let value;

        // 숫자
        if (typeof original === "number") {
          value = Number(input.value);

          if (!Number.isFinite(value)) {
            throw new Error(`${name}은 숫자여야 합니다.`);
          }
        }

        // boolean
        else if (typeof original === "boolean") {
          value = input.value === "true";
        }

        // 배열
        else if (Array.isArray(original)) {
          value = JSON.parse(input.value);

          if (!Array.isArray(value)) {
            throw new Error(`${name}은 배열이어야 합니다.`);
          }
        }

        // object
        else if (
          original !== null &&
          typeof original === "object"
        ) {
          value = JSON.parse(input.value);

          if (
            value === null ||
            typeof value !== "object" ||
            Array.isArray(value)
          ) {
            throw new Error(`${name}은 객체여야 합니다.`);
          }
        }

        // string
        else {
          value = input.value;
        }

        skillNode.arguments[name] = value;
      });

      return true;

    } catch (error) {
      setBanner(
        `파라미터 수정 실패: ${errorText(error)}`,
        "danger",
      );

      return false;
    }
  }

  function createNumberControl(name, schema, value) {
    const wrapper = create("div", {
      className: "parameter-control",
    });

    const header = create("div", {
      className: "parameter-control-header",
    });

    const label = create("span", {
      text: name,
    });

    const valueLabel = create("span", {
      className: "parameter-value",
    });

    const input = document.createElement("input");

    input.type = "range";
    input.className = "parameter-slider";

    input.min = schema.minimum ?? 0;
    input.max = schema.maximum ?? 1;
    input.step = schema.type === "integer" ? 1 : 0.01;

    input.value =
      value ??
      schema.default ??
      ((Number(input.min) + Number(input.max)) / 2);

    valueLabel.textContent = input.value;

    input.addEventListener("input", () => {
      valueLabel.textContent = input.value;
    });

    header.append(label, valueLabel);

    wrapper.append(header, input);

    return wrapper;
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
      dom.cameraStatus.textContent = `RealSense RGB + 정렬 Depth 스트리밍 중 · 프레임 ${camera.frameNumber ?? "대기"}`;
      dom.cameraStatus.style.color = "var(--ok)";
    } else if (camera.state === "error") {
      dom.cameraStatus.textContent = `RealSense 오류: ${camera.lastError || "장치 또는 Python 바인딩을 확인하세요."}`;
      dom.cameraStatus.style.color = "var(--danger)";
    } else {
      dom.cameraStatus.textContent = "RealSense 정지됨 · 사용자가 시작할 때만 카메라가 열립니다.";
      dom.cameraStatus.style.color = "";
    }
    renderCalibration();
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
      dom.calibrationStatus.textContent = `하드웨어 gate 닫힘 · ${failedGates.join(", ")}`;
      dom.calibrationStatus.style.color = "var(--danger)";
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

  function renderGeometryMarkers() {
    dom.recordedPointOverlay.replaceChildren();
    const teaching = state.geometryTeaching;
    if (!teaching.mode || !dom.recordedRgbFrame.naturalWidth) return;
    const image = dom.recordedRgbFrame;
    const scale = Math.min(
      image.clientWidth / image.naturalWidth,
      image.clientHeight / image.naturalHeight,
    );
    const offsetX = (image.clientWidth - image.naturalWidth * scale) / 2;
    const offsetY = (image.clientHeight - image.naturalHeight * scale) / 2;
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
      dom.recordedPointOverlay.replaceChildren();
      return;
    }
    dom.geometryTeachingTitle.textContent = calibrationMode
      ? "camera → surface TF 보정"
      : "두 fingertip TCP 경로 티칭";
    dom.geometryTeachingInstruction.textContent = calibrationMode
      ? "한 프레임의 같은 평면에서 원점(O), +X 방향점(X), +Y 방향점(Y)을 순서대로 클릭하세요. 세 점은 각각 3 cm 이상 떨어뜨리세요."
      : "현재 프레임에서 집게로 사용할 두 fingertip 끝 A와 B를 클릭하고 ‘현재 프레임 샘플 추가’를 누르세요. 서로 다른 프레임에서 최소 4개를 저장합니다.";
    dom.surfaceAnchorLabel.hidden = !calibrationMode;
    dom.addPathSample.hidden = !pathMode;
    dom.addPathSample.disabled = teaching.busy || teaching.points.length !== 2;
    dom.saveGeometryEvidence.disabled = teaching.busy || (
      calibrationMode ? teaching.points.length !== 3 : teaching.samples.length < 4
    );
    dom.resetGeometryPoints.disabled = teaching.busy || teaching.points.length === 0;
    dom.cancelGeometryTeaching.disabled = teaching.busy;
    dom.geometryPointSummary.textContent = calibrationMode
      ? `선택한 점 ${teaching.points.length}/3 · 기준 프레임 ${state.recordingReview.frameIndex}`
      : `현재 fingertip ${teaching.points.length}/2 · 저장된 경로 샘플 ${teaching.samples.length}/4 이상`;
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
        ? `GPT LIVE 준비됨 · ${capabilities.model} · RGB-D 최대 ${capabilities.maximum_keyframes}쌍 · 두 손가락 TCP 프록시 · 실패 시 PDF 재전송/ZIP 보관`
        : "OPENAI_MODE=live이지만 서버에 OPENAI_API_KEY가 없습니다."
      : `현재 MOCK 분석 모드 · 실제 GPT 전송은 OPENAI_MODE=live에서만 수행됩니다. · ${capabilities.model}`;
    dom.openaiCapability.style.color = ready ? "" : "var(--danger)";
    const configuredMaximum = Number(capabilities.maximum_keyframes || 300);
    dom.draftKeyframeCount.max = String(Math.min(300, configuredMaximum));
    if (Number(dom.draftKeyframeCount.value) > configuredMaximum) {
      dom.draftKeyframeCount.value = String(configuredMaximum);
    }
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
      dom.draftStatus.textContent = "대표 RGB-D 쌍과 두 손가락 TCP 프록시 프롬프트를 준비하는 중…";
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

  async function createManualSkill(event) {
    event.preventDefault();

    if (!dom.manualSkillForm.reportValidity()) return;

    state.manualSkill.loading = true;

    try {
      const result = await api.createSkill({
        name: dom.manualSkillName.value.trim(),
        intent: dom.manualSkillId.value.trim(),
        variant: "default",
        description: dom.manualSkillDescription.value.trim(),
        semantic_version: "0.1.0",
      });

      state.manualSkill.nodes = [];

      dom.manualSkillForm.reset();

      await loadRegistry(
        `${result.name || result.intent || "스킬"} 생성 완료`,
      );

      showPage("skills");
    } catch (error) {
      setBanner(
        `스킬 생성 실패: ${errorText(error)}`,
        "danger",
      );
    } finally {
      state.manualSkill.loading = false;
    }

    renderManualNodes();
  }

  async function loadPrimitiveCatalog() {
    try {
      const result = await api.getPrimitiveCatalog();
      state.primitiveCatalog = result.primitives || [];
    } catch (error) {
      console.error("Primitive catalog loading failed:", error);
      state.primitiveCatalog = [];
    }
  }

  function getPrimitiveMetadata(operation) {
    const primitive = state.primitiveCatalog.find(
      (primitive) => primitive.operation_name === operation
    );

    return primitive;
  }

  function resolveParameterSchema(schema, definitions = {}) {
    if (!schema) {
      return {};
    }

    // 직접적인 $ref
    if (
      schema.$ref &&
      schema.$ref.startsWith("#/$defs/")
    ) {
      const name = schema.$ref.replace(
        "#/$defs/",
        "",
      );

      const resolved = definitions[name];

      if (resolved) {
        return resolveParameterSchema(
          resolved,
          definitions,
        );
      }
    }

    // anyOf
    if (Array.isArray(schema.anyOf)) {
      const candidate = schema.anyOf.find(
        (item) =>
          item &&
          (
            item.type ||
            item.$ref ||
            item.properties
          ),
      );

      if (candidate) {
        return resolveParameterSchema(
          candidate,
          definitions,
        );
      }
    }

    // oneOf
    if (Array.isArray(schema.oneOf)) {
      const candidate = schema.oneOf.find(
        (item) =>
          item &&
          (
            item.type ||
            item.$ref ||
            item.properties
          ),
      );

      if (candidate) {
        return resolveParameterSchema(
          candidate,
          definitions,
        );
      }
    }

    return {
      ...schema,
      __definitions: definitions,
    };
  }

  function createParameterEditor(name, schema, value) {
    const row = create("div", {
      className: "parameter-row",
    });

    const label = create("div", {
      className: "parameter-label",
      text: name,
    });

    const control = create("div", {
      className: "parameter-control",
    });

    row.append(label, control);

    // =========================================================
    // schema 안전 처리
    // =========================================================

    schema = schema || {};

    // =========================================================
    // enum
    // =========================================================

    if (schema.enum) {
      const input = document.createElement("select");

      input.className = "parameter-input";

      schema.enum.forEach((option) => {
        const item = document.createElement("option");

        item.value = option;
        item.textContent = option;

        input.append(item);
      });

      input.value =
        value ??
        schema.default ??
        schema.enum[0];

      input.dataset.parameterName = name;

      control.append(input);

      return row;
    }

    // =========================================================
    // boolean
    // =========================================================

    if (schema.type === "boolean") {
      const toggleWrap = create("label", {
        className: "parameter-toggle",
      });

      const input = document.createElement("input");

      input.type = "checkbox";

      input.checked =
        value ??
        schema.default ??
        false;

      input.dataset.parameterName = name;

      const slider = create("span", {
        className: "parameter-toggle-slider",
      });

      const text = create("span", {
        className: "parameter-toggle-text",
        text: input.checked ? "ON" : "OFF",
      });

      input.addEventListener("change", () => {
        text.textContent =
          input.checked ? "ON" : "OFF";
      });

      toggleWrap.append(
        input,
        slider,
        text,
      );

      control.append(toggleWrap);

      return row;
    }

    // =========================================================
    // number / integer
    // =========================================================

    if (
      schema.type === "number" ||
      schema.type === "integer"
    ) {
      const input = document.createElement("input");

      input.type = "number";
      input.className = "parameter-input";

      if (schema.minimum !== undefined) {
        input.min = schema.minimum;
      }

      if (schema.maximum !== undefined) {
        input.max = schema.maximum;
      }

      input.step =
        schema.type === "integer"
          ? "1"
          : "0.01";

      if (value !== undefined) {
        input.value = value;
      } else if (schema.default !== undefined) {
        input.value = schema.default;
      }

      input.dataset.parameterName = name;

      control.append(input);

      return row;
    }

    // =========================================================
    // object
    // =========================================================

    if (schema.type === "object") {
      const objectEditor = createObjectEditor(
        schema,
        value ?? {},
      );

      const hidden = document.createElement("textarea");

      hidden.className =
        "parameter-input parameter-object parameter-json-hidden";

      hidden.dataset.parameterName = name;
      hidden.hidden = true;

      hidden.value = JSON.stringify(
        value ?? {},
      );

      control.append(
        objectEditor,
        hidden,
      );

      const sync = () => {
        const result =
          objectEditor.__getValue
            ? objectEditor.__getValue()
            : value ?? {};

        hidden.value = JSON.stringify(result);

        hidden.dispatchEvent(
          new Event("change", {
            bubbles: true,
          }),
        );
      };

      objectEditor
        .querySelectorAll("input, select, textarea")
        .forEach((editorInput) => {
          editorInput.addEventListener(
            "change",
            sync,
          );

          editorInput.addEventListener(
            "input",
            sync,
          );
        });

      return row;
    }

    // =========================================================
    // array
    // =========================================================

    if (schema.type === "array") {
      const input = document.createElement("textarea");

      input.className =
        "parameter-input parameter-object";

      input.rows = 4;

      input.value =
        value !== undefined
          ? JSON.stringify(value, null, 2)
          : "[]";

      input.dataset.parameterName = name;

      control.append(input);

      return row;
    }

    // =========================================================
    // string
    // =========================================================

    const input = document.createElement("input");

    input.type = "text";
    input.className = "parameter-input";

    if (value !== undefined) {
      input.value = value;
    } else if (schema.default !== undefined) {
      input.value = schema.default;
    }

    input.dataset.parameterName = name;

    control.append(input);

    return row;
  }

  function createObjectEditor(schema, value = {}) {
    const wrapper = create("div", {
      className: "parameter-object-editor",
    });

    const definitions = schema.__definitions || {};

    function resolve(s) {
      if (!s) return {};

      if (s.$ref && s.$ref.startsWith("#/$defs/")) {
        const name = s.$ref.replace("#/$defs/", "");
        return resolve(definitions[name]);
      }

      if (Array.isArray(s.anyOf)) {
        const candidate = s.anyOf.find(
          (v) => v.type || v.$ref || v.properties,
        );
        if (candidate) {
          return resolve(candidate);
        }
      }

      if (Array.isArray(s.oneOf)) {
        const candidate = s.oneOf.find(
          (v) => v.type || v.$ref || v.properties,
        );
        if (candidate) {
          return resolve(candidate);
        }
      }

      return s;
    }

    function build(parent, currentSchema, currentValue) {
      currentSchema = resolve(currentSchema);

      Object.entries(currentSchema.properties || {}).forEach(
        ([key, raw]) => {
          const fieldSchema = resolve(raw);
          const fieldValue = currentValue?.[key];

          // ============================
          // object
          // ============================

          if (fieldSchema.type === "object") {
            const group = create("div", {
              className: "parameter-object-group",
            });

            group.dataset.objectKey = key;

            group.__schema = fieldSchema;

            const title = create("div", {
              className: "parameter-object-title",
              text: key,
            });

            const body = create("div", {
              className: "parameter-object-body",
            });

            build(body, fieldSchema, fieldValue || {});

            group.append(title, body);

            parent.append(group);

            return;
          }

          // ============================
          // primitive
          // ============================

          const row = create("div", {
            className: "parameter-object-field",
          });

          const label = create("label", {
            className: "parameter-object-field-label",
            text: key,
          });

          let input;

          if (fieldSchema.enum) {
            input = document.createElement("select");

            fieldSchema.enum.forEach((option) => {
              const o = document.createElement("option");
              o.value = option;
              o.textContent = option;
              input.append(o);
            });

            input.value =
              fieldValue ??
              fieldSchema.default ??
              fieldSchema.enum[0];
          }

          else if (
            fieldSchema.type === "number" ||
            fieldSchema.type === "integer"
          ) {
            input = document.createElement("input");
            input.type = "number";

            input.step =
              fieldSchema.type === "integer"
                ? "1"
                : "0.001";

            input.value =
              fieldValue ??
              fieldSchema.default ??
              0;
          }

          else if (
            fieldSchema.type === "boolean"
          ) {
            input = document.createElement("input");
            input.type = "checkbox";
            input.checked =
              fieldValue ??
              fieldSchema.default ??
              false;
          }

          else {
            input = document.createElement("input");
            input.type = "text";
            input.value =
              fieldValue ??
              fieldSchema.default ??
              "";
          }

          input.className = "parameter-input";
          input.dataset.objectKey = key;

          input.__schema = fieldSchema;

          row.append(label, input);

          parent.append(row);
        },
      );
    }

    build(wrapper, schema, value);

    function collect(container) {
      const result = {};

      Array.from(container.children).forEach((child) => {

        if (
          child.classList.contains(
            "parameter-object-group",
          )
        ) {
          const key = child.dataset.objectKey;

          const body =
            child.querySelector(
              ".parameter-object-body",
            );

          result[key] = collect(body);

          return;
        }

        if (
          child.classList.contains(
            "parameter-object-field",
          )
        ) {
          const input =
            child.querySelector(
              "[data-object-key]",
            );

          const key =
            input.dataset.objectKey;

          const schema =
            input.__schema || {};

          if (
            input.type === "checkbox"
          ) {
            result[key] =
              input.checked;
          }

          else if (
            schema.type === "number"
          ) {
            result[key] =
              Number(input.value);
          }

          else if (
            schema.type === "integer"
          ) {
            result[key] =
              parseInt(
                input.value,
                10,
              );
          }

          else {
            result[key] =
              input.value;
          }
        }

      });

      return result;
    }

    wrapper.__getValue = () => collect(wrapper);

    return wrapper;
  }

  function createSkillBlock(node) {
    const block = create("div", {
      className: "skill-block",
    });

    block.append(
      create("div", {
        className: "skill-block-title",
        text: node.operation,
      })
    );

    Object.entries(node.parameters || {}).forEach(([key, value]) => {
      block.append(
        create("div", {
          className: "skill-block-item",
          text: `${key} : ${formatParameterValue(value)}`,
        })
      );
    });

    return block;
  }

  let saveTimer = null;
  function scheduleNodeSave(skill, skillNode, name, value) {
    skillNode.arguments[name] = value;

    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      try {
        await api.updateSkillNode(
          skill.id,
          skillNode.node_id,
          {
            [name]: value,
          },
          skill.version,
        );
      } catch (error) {
        console.error(error);
      }
    }, 400);
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
          ? "GPT 분석 완료 · 직접 RGB-D 입력이 거부되어 PDF로 재전송됨 · ZIP 원본 보관 완료"
          : `GPT 분석 완료 · ${result.openai.model} · RGB-D ${result.transport?.keyframe_pair_count || result.keyframe_indices.length}쌍 전송됨`
        : "Mock 분석 완료 · OPENAI_MODE=live에서 같은 버튼을 누르면 GPT에 전송됩니다.";
    } catch (error) {
      dom.draftStatus.textContent = `스킬 초안 생성 실패: ${errorText(error)}`;
      dom.draftStatus.style.color = "var(--danger)";
    } finally {
      review.loading = false;
    }
    renderRecordingReview();
  }

  function renderAll() {
    renderConnection();
    renderRegistry();
    if (state.page === "detail") renderDetail();
    if (state.page === "monitor") renderMonitor();
    if (state.page === "create") renderRecordingReview();
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
      const [, registry, draftCatalog] = await Promise.all([
        api.health(),
        api.listSkills(),
        api.listSkillDrafts(),
      ]);
      state.skills = (registry.skills || []).map(normalizeSkill);
      state.drafts = (draftCatalog.drafts || []).map(normalizeDraft);
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
    state.geometryTeaching = {
      mode,
      draftId: draft.draftId,
      points: [],
      samples: [],
      busy: false,
    };
    state.recordingReview.selectedId = draft.sourceRecordingId;
    state.recordingReview.loaded = false;
    state.recordingReview.frameIndex = 0;
    state.recordingReview.draft = null;
    stopRecordingPlayback();
    showPage("create");
    setBanner(
      mode === "surface_calibration"
        ? "녹화 RGB에서 표면 원점, +X, +Y를 지정하세요."
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
    if (teaching.mode === "manual_tcp_path" && teaching.samples.length < 4) return;
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
        await loadRegistry("camera → surface TF 보정 완료 · 실제 로봇 TF는 별도 검증 필요");
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
      "GPT의 작업대 영역은 힌트로만 사용하고, 원본 aligned Depth와 카메라 내부 파라미터로 평면을 계산합니다. 결과는 Candidate/Mock 전용이며 화면에서 평면이 맞는지 확인해야 합니다. 계속할까요?",
    );
    if (!confirmed) return;
    setBanner("원본 Depth에서 작업대 평면과 camera → surface TF 계산 중…");
    dom.autoSurfaceCalibration.disabled = true;
    try {
      const result = await api.autoCalibrateDraftSurface(draft.draftId, {
        frame_index: null,
        surface_anchor_id: "teaching_surface",
        operator_confirmed: true,
      });
      const inlier = Number(result.diagnostics?.inlier_ratio || 0);
      await loadRegistry(
        `Depth 평면 TF 생성 완료 · inlier ${(inlier * 100).toFixed(1)}% · 실제 로봇 TF 검증은 별도 필요`,
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
      "GPT가 모든 대표 프레임에 표시한 두 fingertip 좌표를 원본 aligned Depth로 3D 복원합니다. 결과는 Candidate/Mock 전용이며 실제 로봇 궤적으로 승인되지 않습니다. 계속할까요?",
    );
    if (!confirmed) return;
    setBanner("GPT fingertip audit + 로컬 aligned Depth로 TCP 경로 복원 중…");
    dom.autoTcpPath.disabled = true;
    try {
      const result = await api.createDraftTcpTrajectory(draft.draftId, {
        method: "openai_rgbd",
        annotations: [],
        operator_confirmed: true,
      });
      await loadRegistry(`자동 TCP 경로 ${result.quality?.sample_count || 0}개 생성 완료`);
    } catch (error) {
      setBanner(`GPT TCP 경로 적용 실패: ${errorText(error)} · 녹화를 다시 분석하거나 수동 경로 티칭을 사용하세요.`, "danger");
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

  async function saveNodeParameter(skill, skillNode, name, value) {
    skillNode.arguments = skillNode.arguments || {};
    skillNode.arguments[name] = value;

    const nodeId = skillNode.nodeId;

    if (!nodeId) {
      console.error("nodeId를 찾을 수 없습니다.", skillNode);
      setBanner("노드 ID를 찾을 수 없습니다.", "danger");
      return;
    }

    try {
      const updated = await api.updateSkillNode(
        skill.id,
        nodeId,
        {
          [name]: value,
        },
        skill.version,
      );

      setBanner(
        `${name} 파라미터가 저장되었습니다.`,
        "ok",
      );
    } catch (error) {
      console.error("노드 파라미터 저장 실패", error);

      setBanner(
        `파라미터 저장 실패: ${errorText(error)}`,
        "danger",
      );
    }
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
  window.addEventListener("resize", renderGeometryMarkers);

  dom.manualSkillForm.addEventListener(
    "submit",
    createManualSkill,
  );
  renderAll();
  Promise.all([
    loadPrimitiveCatalog(),
    loadRegistry(),
  ]).then(() => {
    initBlockly();
    return Promise.all([
      refreshCameraStatus(),
      refreshHandeyeCalibrationStatus(),
    ]);
  }).catch(console.error);
  window.setInterval(() => {
    if (state.camera.state === "streaming" || state.camera.recordingId) {
      refreshCameraStatus();
    }
    if (["starting", "moving_to_reference", "running", "aborting"]
      .includes(state.calibration.session?.status)) {
      refreshHandeyeCalibrationStatus();
    }
  }, 1000);

  function registerPrimitiveBlock(primitive) {
    const blockType = primitive.operation_name.replace(/\./g, "_");
    Blockly.Blocks[blockType] = {
      init: function () {
        this.appendDummyInput()
          .appendField(
            primitive.display_name || primitive.operation_name
          );
          
        const schema =
          primitive.typed_parameter_schema ||
          primitive.parameter_schema ||
          {};

        const properties = schema.properties || {};
        Object.entries(properties).forEach(([name, info]) => {
          this.appendDummyInput()
            .appendField(name)
            .appendField(
              new Blockly.FieldTextInput(""),
              name,
            );
        });

        this.setPreviousStatement(true);
        this.setNextStatement(true);
        this.setColour(210);
        this.setTooltip(
          primitive.description || ""
        );
      }
    };
  }

  let workspace = null;

  function initBlockly() {
    const div = document.getElementById("blocklyDiv");
    if (!div) return;

    state.primitiveCatalog.forEach(registerPrimitiveBlock);
    workspace = Blockly.inject(div, {
        toolbox: buildBlocklyToolbox(),
        media: "vendor/blockly/media/"
    });
  }

  function buildBlocklyToolbox() {
    return {
      kind: "categoryToolbox",
      contents: [
        {
          kind: "category",
          name: "Primitive",
          colour: "#4C97FF",
          contents: state.primitiveCatalog.map((p) => ({
            kind: "block",
            type: p.operation_name.replace(/\./g, "_"),
          })),
        },
      ],
    };
  }
})();
