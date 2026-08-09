"use strict";

(() => {
  const DEFAULT_TIMEOUT_MS = 15000;

  class DittobotApiError extends Error {
    constructor(message, { status = 0, detail = null } = {}) {
      super(message);
      this.name = "DittobotApiError";
      this.status = status;
      this.detail = detail;
    }
  }

  function configuredBaseUrl() {
    const params = new URLSearchParams(window.location.search);
    const queryBase = params.get("api");
    if (queryBase) return queryBase;

    const meta = document.querySelector('meta[name="dittobot-api-base"]');
    const metaBase = meta?.getAttribute("content")?.trim();
    if (metaBase) return metaBase;

    if (window.location.protocol === "http:" || window.location.protocol === "https:") {
      return window.location.origin;
    }
    return "http://127.0.0.1:8000";
  }

  function normalizeBaseUrl(value) {
    const url = new URL(value, window.location.href);
    if (!/^https?:$/.test(url.protocol)) {
      throw new DittobotApiError("API 주소는 http 또는 https만 사용할 수 있습니다.");
    }
    return url.toString().replace(/\/$/, "");
  }

  class DittobotApiClient {
    constructor({ baseUrl = configuredBaseUrl(), timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
      this.baseUrl = normalizeBaseUrl(baseUrl);
      this.timeoutMs = timeoutMs;
    }

    async request(path, { method = "GET", body = undefined, timeoutMs = this.timeoutMs } = {}) {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await window.fetch(`${this.baseUrl}${path}`, {
          method,
          headers: body === undefined ? { Accept: "application/json" } : {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: controller.signal,
          credentials: "same-origin",
        });
        const contentType = response.headers.get("content-type") || "";
        const payload = contentType.includes("application/json")
          ? await response.json()
          : await response.text();
        if (!response.ok) {
          const detail = payload && typeof payload === "object" ? payload.detail : payload;
          throw new DittobotApiError(
            typeof detail === "string" ? detail : `API 요청 실패 (${response.status})`,
            { status: response.status, detail },
          );
        }
        return payload;
      } catch (error) {
        if (error?.name === "AbortError") {
          throw new DittobotApiError(`API 응답 제한 시간 ${timeoutMs}ms를 초과했습니다.`);
        }
        if (error instanceof DittobotApiError) throw error;
        throw new DittobotApiError("FastAPI 서버에 연결할 수 없습니다.", { detail: error });
      } finally {
        window.clearTimeout(timeout);
      }
    }

    health() {
      return this.request("/health");
    }

    listSkills() {
      return this.request("/skills");
    }

    deleteSkill(skillId) {
      return this.request(`/skills/${encodeURIComponent(skillId)}`, {
        method: "DELETE",
      });
    }

    deactivateSkill(skillId) {
      return this.request(`/skills/${encodeURIComponent(skillId)}/deactivate`, {
        method: "POST",
      });
    }

    skillEditorCatalog() {
      return this.request("/skills/editor/catalog");
    }

    previewSkillBlocks(payload) {
      return this.request("/skills/editor/preview", {
        method: "POST",
        body: payload,
      });
    }

    createSkillBlockCandidate(payload) {
      return this.request("/skills/editor/candidates", {
        method: "POST",
        body: { ...payload, acknowledge_mock_only: true },
        timeoutMs: 120000,
      });
    }

    createSkillParameterCandidate(skillId, version, payload) {
      return this.request(
        `/skills/${encodeURIComponent(skillId)}/versions/${encodeURIComponent(version)}/parameter-candidates`,
        {
          method: "POST",
          body: { ...payload, acknowledge_mock_only: true },
          timeoutMs: 120000,
        },
      );
    }

    listSkillDrafts() {
      return this.request("/skills/drafts");
    }

    getSkillDraft(draftId) {
      return this.request(`/skills/drafts/${encodeURIComponent(draftId)}`);
    }

    calibrateDraftSurface(draftId, payload) {
      return this.request(`/skills/drafts/${encodeURIComponent(draftId)}/surface-calibration`, {
        method: "POST",
        body: payload,
        timeoutMs: 60000,
      });
    }

    autoCalibrateDraftSurface(draftId, payload) {
      return this.request(`/skills/drafts/${encodeURIComponent(draftId)}/surface-calibration/auto`, {
        method: "POST",
        body: payload,
        timeoutMs: 60000,
      });
    }

    createDraftTcpTrajectory(draftId, payload) {
      return this.request(`/skills/drafts/${encodeURIComponent(draftId)}/tcp-trajectory`, {
        method: "POST",
        body: payload,
        timeoutMs: 360000,
      });
    }

    registerDraftCandidate(draftId) {
      return this.request(`/skills/drafts/${encodeURIComponent(draftId)}/candidate`, {
        method: "POST",
        body: { acknowledge_mock_only: true },
        timeoutMs: 120000,
      });
    }

    getSkill(skillId, version = null) {
      const suffix = version ? `?version=${encodeURIComponent(version)}` : "";
      return this.request(`/skills/${encodeURIComponent(skillId)}${suffix}`);
    }

    validateSkill(skillId, version) {
      return this.request(`/skills/${encodeURIComponent(skillId)}/validate`, {
        method: "POST",
        body: { version, mode: "mock" },
      });
    }

    activateSkill(skillId, version) {
      return this.request(`/skills/${encodeURIComponent(skillId)}/activate`, {
        method: "POST",
        body: { version },
      });
    }

    captureScene() {
      return this.request("/scenes/capture", {
        method: "POST",
        body: { mode: "mock" },
      });
    }

    cameraStatus() {
      return this.request("/camera/status");
    }

    handeyeCalibrationStatus() {
      return this.request("/calibration/hand-eye/status");
    }

    startHandeyeCalibration(operatorId = "ui_operator") {
      return this.request("/calibration/hand-eye/start", {
        method: "POST",
        body: {
          operator_id: operatorId,
          operator_confirmed: true,
          board_secured: true,
          workspace_cleared: true,
          estop_ready: true,
        },
      });
    }

    abortHandeyeCalibration(reason = "operator_request") {
      return this.request("/calibration/hand-eye/abort", {
        method: "POST",
        body: { reason },
      });
    }

    jogStatus() {
      return this.request("/jog/status");
    }

    enableJog(payload) {
      return this.request("/jog/enable", {
        method: "POST",
        body: payload,
      });
    }

    moveJogJoint(jointIndex, deltaDeg) {
      return this.request("/jog/joints", {
        method: "POST",
        body: { joint_index: jointIndex, delta_deg: deltaDeg },
      });
    }

    moveJogJoints(targetJointPositionsDeg) {
      return this.request("/jog/movej", {
        method: "POST",
        body: { target_joint_positions_deg: targetJointPositionsDeg },
        timeoutMs: 120000,
      });
    }

    stopJog(reason = "operator_request") {
      return this.request("/jog/stop", {
        method: "POST",
        body: { reason },
      });
    }

    arucoExperimentStatus() {
      return this.request("/aruco-experiment/status");
    }

    enableArucoExperiment(payload) {
      return this.request("/aruco-experiment/enable", {
        method: "POST",
        body: payload,
        timeoutMs: 30000,
      });
    }

    moveArucoReference() {
      return this.request("/aruco-experiment/move-reference", {
        method: "POST",
        body: {},
        timeoutMs: 120000,
      });
    }

    moveArucoPlaneZTest() {
      return this.request("/aruco-experiment/move-plane-z-test", {
        method: "POST",
        body: {},
        timeoutMs: 60000,
      });
    }

    stopArucoExperiment(reason = "operator_request") {
      return this.request("/aruco-experiment/stop", {
        method: "POST",
        body: { reason },
      });
    }

    importLegacyHandeyeNpy(operatorId = "ui_operator") {
      return this.request("/calibration/hand-eye/import-legacy-npy", {
        method: "POST",
        body: {
          operator_id: operatorId,
          operator_confirmed: true,
          acknowledge_candidate_only: true,
        },
      });
    }

    startCameraPreview() {
      return this.request("/camera/preview/start", { method: "POST", body: {} });
    }

    stopCameraPreview() {
      return this.request("/camera/preview/stop", { method: "POST", body: {} });
    }

    cameraStreamUrl(kind) {
      if (!new Set(["rgb", "depth"]).has(kind)) {
        throw new DittobotApiError("카메라 스트림은 rgb 또는 depth여야 합니다.");
      }
      return `${this.baseUrl}/camera/streams/${kind}.mjpg?ts=${Date.now()}`;
    }

    startCameraRecording(maximumDurationS) {
      return this.request("/camera/recordings", {
        method: "POST",
        body: { maximum_duration_s: maximumDurationS },
      });
    }

    stopCameraRecording(recordingId) {
      return this.request(`/camera/recordings/${encodeURIComponent(recordingId)}/stop`, {
        method: "POST",
        body: {},
      });
    }

    getCameraRecording(recordingId) {
      return this.request(`/camera/recordings/${encodeURIComponent(recordingId)}`);
    }

    listCameraRecordings() {
      return this.request("/camera/recordings");
    }

    recordingFrameUrl(recordingId, frameIndex, kind) {
      if (!new Set(["rgb", "depth"]).has(kind)) {
        throw new DittobotApiError("녹화 프레임은 rgb 또는 depth여야 합니다.");
      }
      const safeIndex = Number(frameIndex);
      if (!Number.isInteger(safeIndex) || safeIndex < 0) {
        throw new DittobotApiError("녹화 프레임 번호가 올바르지 않습니다.");
      }
      return `${this.baseUrl}/camera/recordings/${encodeURIComponent(recordingId)}/frames/${safeIndex}/${kind}.jpg`;
    }

    recordingSkillDraftCapabilities() {
      return this.request("/skills/draft-from-recording/capabilities");
    }

    createRecordingSkillDraft({ recordingId, nameHint, operatorInstruction, keyframeCount }) {
      return this.request("/skills/draft-from-recording", {
        method: "POST",
        body: {
          recording_id: recordingId,
          name_hint: nameHint,
          operator_instruction: operatorInstruction,
          keyframe_count: keyframeCount,
        },
        timeoutMs: 360000,
      });
    }

    bindRuntime(skillId, version, sceneId) {
      return this.request("/runtime/bind", {
        method: "POST",
        body: {
          skill_id: skillId,
          version,
          scene_id: sceneId,
          entity_hints: {},
        },
      });
    }

    preflightRuntime(skillId, version, sceneId, bindings) {
      return this.request("/runtime/preflight", {
        method: "POST",
        body: {
          skill_id: skillId,
          version,
          scene_id: sceneId,
          bindings,
          mode: "mock",
        },
      });
    }

    executeRuntime({ skillId, version, sceneId, bindings, runId }) {
      return this.request("/runtime/execute", {
        method: "POST",
        body: {
          skill_id: skillId,
          version,
          scene_id: sceneId,
          bindings,
          mode: "mock",
          text: `UI에서 ${skillId} 실행`,
          run_id: runId,
        },
      });
    }

    abortRuntime(runId, reason = "operator_request") {
      return this.request("/runtime/abort", {
        method: "POST",
        body: { run_id: runId, reason },
      });
    }
  }

  Object.assign(window, { DittobotApiClient, DittobotApiError });
})();
