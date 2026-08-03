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

    async request(path, { method = "GET", body = undefined } = {}) {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), this.timeoutMs);
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
          throw new DittobotApiError(`API 응답 제한 시간 ${this.timeoutMs}ms를 초과했습니다.`);
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
