"use strict";
// adapter.js: Agent Client Protocol (ACP) bridge for the unreal-agent runner.
//
// Zed (ACP client) <-> this adapter <-> unreal-agent-runner (batch process) with
// a local responses-proxy in front of an OpenAI-compatible chat/completions
// gateway.
//
// Lifecycle:
//   initialize            -> capability handshake
//   session/new           -> allocates a session id (reused as the runner's
//                            persisted session id, so follow-up prompts in the
//                            same ACP session continue the same history)
//   session/prompt        -> spawns the runner for one request, translating its
//                            session-item JSONL stream into session/update
//                            notifications; replies with a stopReason
//   session/cancel        -> SIGINT to the runner, prompt ends as "cancelled"
//   session/set_config_option -> model / thinking level per session
//
// Token usage is attached as _meta to every agent_message_chunk (per LLM turn)
// and to the final prompt result (task totals), so a tee/proxy log of the
// traffic carries the full picture: agent name, task text, per-turn tokens and
// task completion time.
//
// Zero dependencies: Node.js >= 20 stdlib only.

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const readline = require("node:readline");
const crypto = require("node:crypto");
const { startProxy } = require("./proxy.js");

const PROTOCOL_VERSION = 1;
const AGENT_INFO = {
  name: "unreal-agent-acp",
  title: "Unreal Agent (ACP bridge)",
  version: "0.1.0",
};

const DEFAULT_MODELS = [
  "glm53-flash",
  "qwen38-flash-next",
  "qwen38-27b",
  "ultra-coder",
  "qwen35-397b-a17b",
  "qwen35-122b-a10b",
  "qwen35-35b-a3b",
];
const THINKING_LEVELS = ["low", "medium", "high", "xhigh", "max"];
const DEFAULT_MODEL = process.env.UA_DEFAULT_MODEL || DEFAULT_MODELS[0];
const DEFAULT_THINKING = process.env.UA_DEFAULT_THINKING || "high";
const SESSION_DIRECTORY =
  process.env.UA_SESSION_DIRECTORY ||
  path.join(process.env.XDG_STATE_HOME || path.join(os.homedir(), ".local", "state"), "unreal-agent", "sessions");

function log(...parts) {
  process.stderr.write("[unreal-acp] " + parts.join(" ") + "\n");
}

function findRunner() {
  const candidates = [
    process.env.UA_RUNNER,
    path.join(os.homedir(), "go", "bin", "unreal-agent-runner"),
    "/opt/homebrew/bin/unreal-agent-runner",
    "/usr/local/bin/unreal-agent-runner",
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      fs.accessSync(candidate, fs.constants.X_OK);
      return candidate;
    } catch {
      // try next
    }
  }
  return "unreal-agent-runner";
}

// ---------- JSON-RPC over ndjson stdio ----------

class Rpc {
  constructor() {
    this.handlers = new Map();
    const rl = readline.createInterface({ input: process.stdin, terminal: false });
    rl.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let message;
      try {
        message = JSON.parse(trimmed);
      } catch (err) {
        log("ignoring non-JSON input line:", err.message);
        return;
      }
      this.dispatch(message);
    });
    rl.on("close", () => this.shutdown());
  }

  dispatch(message) {
    const { id, method, params } = message;
    if (method === undefined) return; // responses to our own requests: none sent
    const handler = this.handlers.get(method);
    if (!handler) {
      if (id !== undefined) this.send({ jsonrpc: "2.0", id, error: { code: -32601, message: "Method not found: " + method } });
      return;
    }
    Promise.resolve()
      .then(() => handler(params, id))
      .then((result) => {
        if (id !== undefined) this.send({ jsonrpc: "2.0", id, result: result === undefined ? {} : result });
      })
      .catch((err) => {
        log("handler error for", method, ":", err.message);
        if (id !== undefined) {
          this.send({
            jsonrpc: "2.0",
            id,
            error: { code: err.code && Number.isInteger(err.code) ? err.code : -32000, message: err.message },
          });
        }
      });
  }

  send(message) {
    process.stdout.write(JSON.stringify(message) + "\n");
  }

  notify(method, params) {
    this.send({ jsonrpc: "2.0", method, params });
  }

  on(method, handler) {
    this.handlers.set(method, handler);
  }

  shutdown() {
    for (const session of sessions.values()) session.cancel();
    process.exit(0);
  }
}

// ---------- session state ----------

const sessions = new Map(); // sessionId -> {cwd, model, thinking, child, cancelled, pendingPromptId}

// ---------- ACP mapping helpers ----------

function mapUsage(usage) {
  if (!usage) return null;
  return {
    input_tokens: usage.InputTokens ?? 0,
    cached_input_tokens: usage.CachedInputTokens ?? 0,
    output_tokens: usage.OutputTokens ?? 0,
    reasoning_tokens: usage.ReasoningTokens ?? 0,
  };
}

function addUsage(total, usage) {
  if (!usage) return;
  total.input_tokens += usage.input_tokens ?? 0;
  total.cached_input_tokens += usage.cached_input_tokens ?? 0;
  total.output_tokens += usage.output_tokens ?? 0;
  total.reasoning_tokens += usage.reasoning_tokens ?? 0;
}

function toolKind(name) {
  switch ((name || "").toLowerCase()) {
    case "bash":
      return "execute";
    case "viewimage":
    case "view_image":
      return "read";
    default:
      return "other";
  }
}

function toolTitle(name, argumentsJson) {
  try {
    const parsed = JSON.parse(argumentsJson || "{}");
    if ((name || "").toLowerCase() === "bash" && typeof parsed.command === "string") return parsed.command;
    if (typeof parsed.path === "string") return name + " " + parsed.path;
  } catch {
    // fall through to default title
  }
  return name;
}

function stopFromResponse(stop) {
  switch (stop) {
    case "max_output_tokens":
      return "max_tokens";
    case "refused":
      return "refused";
    default:
      return "end_turn";
  }
}

// ---------- prompt execution ----------

async function runPrompt(rpc, session, params, promptId) {
  const runner = findRunner();
  const request = {
    prompt: params.prompt
      .filter((block) => block && block.type === "text" && typeof block.text === "string")
      .map((block) => block.text)
      .join("\n"),
    session_id: session.id,
    model: session.model,
    thinking_level: session.thinking,
  };
  if (!request.prompt.trim()) {
    throw Object.assign(new Error("prompt contains no text blocks"), { code: -32602 });
  }

  const totals = { input_tokens: 0, cached_input_tokens: 0, output_tokens: 0, reasoning_tokens: 0 };
  let lastStop = null;
  let errorMessage = null;
  session.cancelled = false;

  await new Promise((resolve) => {
    const child = spawn(
      runner,
      ["-workspace", session.cwd],
      {
        cwd: session.cwd,
        env: {
          ...process.env,
          UNREAL_HARNESS_LLM_PROVIDER: "openai",
          UNREAL_HARNESS_LLM_BASE_URL: "http://127.0.0.1:" + session.proxyPort + "/v1",
          UNREAL_HARNESS_LLM_API_KEY: "local-proxy",
          UNREAL_HARNESS_LLM_MODEL: session.model,
          UNREAL_HARNESS_LLM_MAX_ATTEMPTS: "5",
        },
        stdio: ["pipe", "pipe", "pipe"],
      },
    );
    session.child = child;
    log("runner started pid=" + child.pid, "session=" + session.id, "model=" + session.model);

    child.stdin.write(JSON.stringify(request) + "\n");
    child.stdin.end();

    const rl = readline.createInterface({ input: child.stdout, terminal: false });
    rl.on("line", (line) => {
      if (!line.trim()) return;
      let item;
      try {
        item = JSON.parse(line);
      } catch {
        return;
      }
      // Runner session items: {"Sequence":n,"RecordedAt":...,"Kind":"...","Data":{...}}
      // (field names PascalCase; kind values lowercase). Errors arrive as
      // {"Type":"error","Message":"..."}.
      if (item.Type === "error") {
        errorMessage = item.Message;
        return;
      }
      if (item.Kind !== "model_response" && item.Kind !== "tool_call_status") return;

      if (item.Kind === "model_response") {
        const response = item.Data?.Response || {};
        lastStop = response.Stop || lastStop;
        const turnUsage = mapUsage(response.Usage);
        addUsage(totals, turnUsage);
        for (const output of response.Output || []) {
          if (output.Type === "message") {
            rpc.notify("session/update", {
              sessionId: session.id,
              update: {
                sessionUpdate: "agent_message_chunk",
                content: { type: "text", text: output.Data?.Text ?? "" },
                _meta: { unreal: { model: session.model, turn: item.Data?.TurnID, usage: turnUsage } },
              },
            });
          } else if (output.Type === "tool_call") {
            const call = output.Data || {};
            rpc.notify("session/update", {
              sessionId: session.id,
              update: {
                sessionUpdate: "tool_call",
                toolCallId: call.CallID || call.Name,
                title: toolTitle(call.Name, call.Arguments),
                kind: toolKind(call.Name),
                status: "pending",
                rawInput: call.Arguments || "{}",
              },
            });
          } else if (output.Type === "reasoning") {
            const summary = (output.Data?.Summary || []).join("\n");
            if (summary) {
              rpc.notify("session/update", {
                sessionId: session.id,
                update: {
                  sessionUpdate: "agent_thought_chunk",
                  content: { type: "text", text: summary },
                },
              });
            }
          }
        }
        return;
      }

      // tool_call_status Data: {TurnID, CallID, Status: {Error?, WaitingFor?}, Operations?}
      const status = item.Data?.Status || {};
      const acpStatus = status.Error ? "failed" : (status.WaitingFor?.length ? "in_progress" : "completed");
      const update = {
        sessionUpdate: "tool_call_update",
        toolCallId: item.Data?.CallID,
        status: acpStatus,
      };
      if (status.Error) {
        update.content = [{ type: "content", content: { type: "text", text: "error: " + status.Error } }];
      }
      rpc.notify("session/update", { sessionId: session.id, update });
    });

    let stderrTail = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderrTail = (stderrTail + chunk).slice(-4000);
    });

    child.on("error", (err) => {
      errorMessage = "failed to start runner: " + err.message;
      resolve();
    });
    child.on("close", (code) => {
      log("runner exited code=" + code, "session=" + session.id);
      if (code !== 0 && !session.cancelled && !errorMessage) {
        errorMessage = (stderrTail.split("\n").filter(Boolean).pop() || "runner exited with code " + code).trim();
      }
      resolve();
    });
  });

  if (session.cancelled) {
    return {
      _meta: { unreal: { stop: "cancelled", usage: totals } },
      stopReason: "cancelled",
    };
  }
  if (errorMessage && lastStop === null) {
    throw Object.assign(new Error(errorMessage), { code: -32000 });
  }
  if (errorMessage) {
    // runner finished with an error event but had produced model output; surface it
    log("runner error: " + errorMessage);
  }
  return {
    _meta: { unreal: { stop: lastStop || "complete", usage: totals } },
    stopReason: lastStop ? stopFromResponse(lastStop) : "end_turn",
  };
}

// ---------- server bootstrap ----------

async function main() {
  const proxy = await startProxy({
    baseURL: process.env.UA_BASE_URL || "https://api.openai.com/v1",
    apiKey: process.env.UA_API_KEY || "",
  });

  const rpc = new Rpc();

  rpc.on("initialize", () => ({
    protocolVersion: PROTOCOL_VERSION,
    agentInfo: AGENT_INFO,
    authMethods: [],
    agentCapabilities: {
      loadSession: false,
      promptCapabilities: { image: false, audio: false, embeddedContext: false },
    },
  }));

  rpc.on("initialized", () => {});

  rpc.on("session/new", (params) => {
    const id = crypto.randomUUID();
    sessions.set(id, {
      id,
      cwd: params?.cwd || process.cwd(),
      model: DEFAULT_MODEL,
      thinking: DEFAULT_THINKING,
      proxyPort: proxy.port,
      child: null,
      cancelled: false,
    });
    return {
      sessionId: id,
      configOptions: [
        {
          type: "select",
          id: "model",
          category: "model",
          name: "Model",
          description: "Model served through the responses-proxy",
          currentValue: DEFAULT_MODEL,
          options: DEFAULT_MODELS.map((model) => ({ value: model, name: model, description: null })),
        },
        {
          type: "select",
          id: "thinking_level",
          category: "thought_level",
          name: "Thinking",
          description: "Reasoning effort passed to the runner",
          currentValue: DEFAULT_THINKING,
          options: THINKING_LEVELS.map((level) => ({ value: level, name: "Thinking: " + level, description: null })),
        },
      ],
    };
  });

  rpc.on("session/set_config_option", (params) => {
    const session = sessions.get(params?.sessionId);
    if (!session) throw new Error("unknown session: " + params?.sessionId);
    if (params.configId === "model" && typeof params.value === "string") session.model = params.value;
    if (params.configId === "thinking_level" && typeof params.value === "string") session.thinking = params.value;
    return {};
  });

  rpc.on("session/load", () => {
    throw Object.assign(new Error("loadSession is not supported by unreal-agent-acp"), { code: -32601 });
  });

  rpc.on("session/prompt", async (params, id) => {
    const session = sessions.get(params?.sessionId);
    if (!session) throw Object.assign(new Error("unknown session: " + params?.sessionId), { code: -32602 });
    session.pendingPromptId = id;
    return runPrompt(rpc, session, params, id);
  });

  rpc.on("session/cancel", (params) => {
    const session = sessions.get(params?.sessionId);
    if (!session) return {};
    session.cancelled = true;
    if (session.child) {
      try {
        session.child.kill("SIGINT");
      } catch {
        // already gone
      }
    }
    return {};
  });

  log("ready: runner=" + findRunner(), "sessions=" + SESSION_DIRECTORY);
}

main().catch((err) => {
  log("fatal:", err.message);
  process.exit(1);
});
