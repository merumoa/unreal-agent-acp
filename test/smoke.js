#!/usr/bin/env node
"use strict";
// smoke.js: drives the adapter (optionally through acp-tee.js) with a scripted
// ACP client session and checks the observed traffic.
//
// Env:
//   UA_API_KEY / UA_BASE_URL - upstream chat/completions gateway (required)
//   UA_MODEL                 - model to select for the session (recommended;
//                              defaults to the adapter's first UA_MODELS entry)
//   UA_TEE                   - optional path to acp-tee.js; when set, the adapter
//                              is launched through the tee and the log is checked

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const readline = require("node:readline");

const ADAPTER = path.join(__dirname, "..", "src", "adapter.js");
const TEE = process.env.UA_TEE || "";

function fail(message) {
  process.stderr.write("SMOKE FAIL: " + message + "\n");
  process.exit(1);
}

if (!process.env.UA_API_KEY || !process.env.UA_BASE_URL) {
  fail("UA_API_KEY and UA_BASE_URL must be set");
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "unreal-acp-smoke-"));
const teeLog = path.join(tmp, "smoke.ndjson");

let command = "node";
let args = [ADAPTER];
if (TEE) {
  command = "node";
  args = [TEE, "--log", teeLog, "--tag", "unreal", "--", "node", ADAPTER];
}

const child = spawn(command, args, { env: process.env, stdio: ["pipe", "pipe", "pipe"] });
child.stderr.setEncoding("utf8");
child.stderr.on("data", (chunk) => process.stderr.write("[adapter] " + chunk));

const seen = {
  toolCall: false,
  toolCallUpdate: false,
  messageChunk: false,
  usage: false,
  completedOutput: false,
  replayUserChunk: false,
};
let updateCount = 0;
let currentSession = null;
let nextId = 1;
const pending = new Map(); // request id -> handler name

function send(message) {
  child.stdin.write(JSON.stringify(message) + "\n");
}

function request(method, params, handler) {
  const id = nextId++;
  pending.set(id, handler);
  send({ jsonrpc: "2.0", id, method, params });
  return id;
}

const timer = setTimeout(() => fail("timeout waiting for prompt to finish"), 180000);

function finish() {
  clearTimeout(timer);
  if (TEE) {
    const lines = fs.readFileSync(teeLog, "utf8").trim().split("\n").filter(Boolean);
    if (!lines.some((l) => l.includes('"tag":"unreal"') && l.includes('"dir":"z2a"') && l.includes("initialize"))) {
      fail("tee log misses z2a initialize");
    }
    if (!lines.some((l) => l.includes("agent_message_chunk") && l.includes('"usage"'))) {
      fail("tee log misses usage meta on message chunk");
    }
    if (!lines.some((l) => l.includes('"stopReason"'))) {
      fail("tee log misses prompt result with stopReason");
    }
    process.stderr.write("tee log: " + lines.length + " records in " + teeLog + "\n");
  }
  process.stderr.write(
    "SMOKE OK: session=" +
      currentSession +
      " updates=" +
      updateCount +
      " toolCall=" +
      seen.toolCall +
      " toolCallUpdate=" +
      seen.toolCallUpdate +
      " messageChunk=" +
      seen.messageChunk +
      " usage=" +
      seen.usage +
      " resume=ok" +
      "\n",
  );
  child.kill("SIGTERM");
  process.exit(0);
}

const rl = readline.createInterface({ input: child.stdout, terminal: false });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let message;
  try {
    message = JSON.parse(trimmed);
  } catch {
    return;
  }

  if (message.id !== undefined && message.method === undefined) {
    if (message.error) {
      fail("JSON-RPC error for " + (pending.get(message.id) || message.id) + ": " + JSON.stringify(message.error));
    }
    const handler = pending.get(message.id);
    if (!handler) return;
    pending.delete(message.id);
    if (handler === "initialize") {
      if (message.result.protocolVersion !== 1) fail("unexpected protocolVersion");
      send({ jsonrpc: "2.0", method: "initialized", params: {} });
      request("session/new", { cwd: process.cwd(), mcpServers: [] }, "sessionNew");
    } else if (handler === "sessionNew") {
      currentSession = message.result.sessionId;
      process.stderr.write("session: " + currentSession + "\n");
      if (process.env.UA_MODEL) {
        request(
          "session/set_config_option",
          { sessionId: currentSession, configId: "model", value: process.env.UA_MODEL },
          "setModel",
        );
      } else {
        request(
          "session/set_config_option",
          { sessionId: currentSession, configId: "thinking_level", value: "low" },
          "setConfig",
        );
      }
    } else if (handler === "setModel") {
      request(
        "session/set_config_option",
        { sessionId: currentSession, configId: "thinking_level", value: "low" },
        "setConfig",
      );
    } else if (handler === "setConfig") {
      request(
        "session/prompt",
        {
          sessionId: currentSession,
          prompt: [
            {
              type: "text",
              text:
                "Run the bash command `echo unreal-smoke-ok` with the bash tool, then reply with the exact command output and nothing else.",
            },
          ],
        },
        "prompt",
      );
    } else if (handler === "prompt") {
      if (!message.result.stopReason) fail("prompt result without stopReason");
      if (!seen.usage) fail("no usage meta observed");
      if (!seen.messageChunk) fail("no agent_message_chunk observed");
      if (!seen.toolCall || !seen.toolCallUpdate) fail("tool call flow not observed");
      if (!seen.completedOutput) fail("no completed tool_call_update with output content");
      request("session/load", { sessionId: currentSession, cwd: process.cwd(), mcpServers: [] }, "load");
    } else if (handler === "load") {
      if (!Array.isArray(message.result.configOptions) || !message.result.configOptions.length) {
        fail("session/load returned no configOptions");
      }
      if (!seen.replayUserChunk) fail("session/load replayed no user_message_chunk");
      request(
        "session/prompt",
        {
          sessionId: currentSession,
          prompt: [{ type: "text", text: "Reply with exactly: resume-ok" }],
        },
        "prompt2",
      );
    } else if (handler === "prompt2") {
      if (message.result.stopReason !== "end_turn") fail("resume prompt ended with " + message.result.stopReason);
      finish();
    }
    return;
  }

  if (message.method === "session/update") {
    const update = message.params && message.params.update;
    if (!update) return;
    updateCount++;
    if (update.sessionUpdate === "agent_message_chunk") {
      seen.messageChunk = true;
      const usage = update._meta?.unreal?.usage;
      if (usage && (usage.input_tokens || 0) + (usage.output_tokens || 0) > 0) seen.usage = true;
    }
    if (update.sessionUpdate === "tool_call") seen.toolCall = true;
    if (update.sessionUpdate === "user_message_chunk") seen.replayUserChunk = true;
    if (update.sessionUpdate === "tool_call_update") {
      seen.toolCallUpdate = true;
      if (update.status === "failed" && update.content) {
        fail("tool call failed: " + JSON.stringify(update.content));
      }
      if (update.status === "completed" && Array.isArray(update.content) && update.content.length) {
        seen.completedOutput = true;
      }
    }
  }
});

child.on("exit", (code) => {
  if (code !== 0 && code !== null) fail("adapter exited with code " + code);
});

request(
  "initialize",
  { protocolVersion: 1, clientCapabilities: { fs: { readTextFile: false, writeTextFile: false } } },
  "initialize",
);
