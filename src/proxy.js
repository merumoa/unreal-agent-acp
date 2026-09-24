"use strict";
// responses-proxy.js: local OpenAI Responses API -> chat/completions translator.
//
// The unreal-agent-runner (github.com/unreallabsai/unreal-agent) speaks the
// OpenAI *Responses* API only. Many OpenAI-compatible gateways (vLLM,
// one-api-style aggregators, corporate LLM gateways) expose /v1/chat/completions
// but not /v1/responses. This proxy accepts Responses API requests from the
// runner, translates them to chat/completions, and translates the streaming
// reply back into the minimal Responses SSE event set the runner consumes
// (a single `response.completed` / `response.failed` / `response.incomplete`
// event carrying the full output + usage).
//
// Zero dependencies: Node.js >= 20 stdlib only.

const http = require("node:http");
const https = require("node:https");
const crypto = require("node:crypto");

function log(...parts) {
  process.stderr.write("[responses-proxy] " + parts.join(" ") + "\n");
}

// Kill an upstream call that has been silent for this long instead of hanging
// the prompt forever (the runner has its own retry policy around non-2xx).
const UPSTREAM_TIMEOUT_MS = 10 * 60 * 1000;

// ---------- request translation: Responses input -> chat messages ----------

function contentText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part.text === "string") return part.text;
        if (part && part.type === "input_image")
          return "[image: " + (part.image_url || "embedded") + "]";
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

// Translates the `input` array of a Responses API request into chat messages.
// Reasoning items are dropped: a chat backend never produced them, so there is
// nothing to replay.
function inputToChatMessages(input) {
  const messages = [];
  let pendingCalls = null; // tool_calls accumulated for one assistant message
  const flushCalls = () => {
    if (pendingCalls && pendingCalls.length) {
      messages.push({ role: "assistant", content: null, tool_calls: pendingCalls });
    }
    pendingCalls = null;
  };
  for (const item of Array.isArray(input) ? input : []) {
    if (!item || typeof item !== "object") continue;
    // EasyInputMessage (user/system turns) serializes without a "type" field
    const itemType = item.type || (item.role ? "message" : "");
    switch (itemType) {
      case "message": {
        flushCalls();
        messages.push({ role: item.role || "user", content: contentText(item.content) });
        break;
      }
      case "function_call": {
        if (!pendingCalls) pendingCalls = [];
        pendingCalls.push({
          id: item.call_id || item.id || "call_" + pendingCalls.length,
          type: "function",
          function: { name: item.name || "", arguments: item.arguments || "{}" },
        });
        break;
      }
      case "function_call_output": {
        flushCalls();
        messages.push({
          role: "tool",
          tool_call_id: item.call_id || "",
          content: contentText(item.output) || "(empty output)",
        });
        break;
      }
      case "reasoning":
      default:
        // unknown item kinds are skipped rather than failing the whole request
        break;
    }
  }
  flushCalls();
  return messages;
}

function toolsToChat(tools) {
  const out = [];
  for (const tool of Array.isArray(tools) ? tools : []) {
    if (tool && tool.type === "function") {
      out.push({
        type: "function",
        function: {
          name: tool.name,
          description: tool.description || "",
          parameters: tool.parameters || { type: "object", properties: {} },
        },
      });
    }
    // hosted tools (web_search etc.) have no chat/completions equivalent: skipped
  }
  return out;
}

function responsesToChatRequest(body) {
  const chat = {
    model: body.model,
    stream: true,
    messages: inputToChatMessages(body.input),
  };
  if (body.max_output_tokens != null) chat.max_tokens = body.max_output_tokens;
  const chatTools = toolsToChat(body.tools);
  if (chatTools.length) {
    chat.tools = chatTools;
    chat.tool_choice = "auto";
  }
  if (body.stream_options === null || body.stream_options === undefined || typeof body.stream_options !== "object") {
    chat.stream_options = { include_usage: true };
  }
  return chat;
}

// ---------- response translation: chat -> Responses ----------

function emptyUsage() {
  return {
    input_tokens: 0,
    output_tokens: 0,
    input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
    output_tokens_details: { reasoning_tokens: 0 },
  };
}

function mapChatUsage(usage) {
  const mapped = emptyUsage();
  if (!usage || typeof usage !== "object") return mapped;
  mapped.input_tokens = usage.prompt_tokens ?? 0;
  mapped.output_tokens = usage.completion_tokens ?? 0;
  if (usage.prompt_tokens_details && typeof usage.prompt_tokens_details === "object") {
    mapped.input_tokens_details.cached_tokens = usage.prompt_tokens_details.cached_tokens ?? 0;
  }
  if (usage.completion_tokens_details && typeof usage.completion_tokens_details === "object") {
    mapped.output_tokens_details.reasoning_tokens = usage.completion_tokens_details.reasoning_tokens ?? 0;
  }
  return mapped;
}

// Some chat templates leak reasoning into the answer text: classic "think"
// XML tags, or GLM-style models writing a literal "Thinking (...): ..." label
// the start of the message content. Clean the answer up: think-tag bodies and
// the leading Thinking label move into the reasoning summary, the rest of the
// text (a plan, notes) stays as the message.
function extractInlineThinking(state) {
  let text = state.messageText;
  if (!text) return;
  text = text.replace(/<think>([\s\S]*?)<\/think>/gi, (_, inner) => {
    state.reasoningText += (state.reasoningText ? "\n" : "") + inner.trim();
    return "";
  });
  const open = text.search(/<think>/i);
  if (open >= 0) {
    // unterminated think block (truncated stream): the tail is reasoning
    state.reasoningText += (state.reasoningText ? "\n" : "") + text.slice(open + 7).trim();
    text = text.slice(0, open);
  }
  text = text.replace(/^\s*\**\s*Thinking\s*\**\s*(\([^)]{0,80}\))?\s*[:：]\s*/i, "");
  state.messageText = text.trim();
}

// Builds the final Responses API `output` array from an accumulated chat reply.
function chatToResponseOutput(state) {
  extractInlineThinking(state);
  const output = [];
  if (state.reasoningText) {
    // surfaced as a reasoning item with summary only; Raw is not replayable and
    // is intentionally omitted
    output.push({
      type: "reasoning",
      id: "rs_proxy_" + state.index,
      summary: [{ type: "summary_text", text: state.reasoningText }],
    });
  }
  if (state.messageText || (!state.toolCalls.length && !state.reasoningText)) {
    output.push({
      type: "message",
      id: "msg_proxy_" + state.index,
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: state.messageText, annotations: [] }],
    });
  }
  for (const call of state.toolCalls) {
    output.push({
      type: "function_call",
      id: call.id,
      call_id: call.id,
      name: call.name,
      arguments: call.arguments || "{}",
      status: "completed",
    });
  }
  return output;
}

function stopStatus(finishReason) {
  switch (finishReason) {
    case "length":
      return { status: "incomplete", incomplete_details: { reason: "max_output_tokens" } };
    case "content_filter":
      return { status: "incomplete", incomplete_details: { reason: "content_filter" } };
    default:
      return { status: "completed", incomplete_details: null };
  }
}

function sseEvent(payload) {
  return "data: " + JSON.stringify(payload) + "\n\n";
}

function terminalEvent(id, state, finishReason) {
  const { status, incomplete_details } = stopStatus(finishReason);
  const response = {
    id: id || "resp_proxy_" + state.index,
    status,
    output: chatToResponseOutput(state),
    usage: mapChatUsage(state.usage),
    incomplete_details,
    error: null,
  };
  const type =
    status === "completed"
      ? "response.completed"
      : status === "incomplete"
        ? "response.incomplete"
        : "response.failed";
  return sseEvent({ type, response });
}

// ---------- upstream chat/completions call ----------

function upstreamClient(url) {
  return url.protocol === "https:" ? https : http;
}

function upstreamRequest(url, apiKey) {
  const headers = { "Content-Type": "application/json" };
  if (apiKey) headers.Authorization = "Bearer " + apiKey;
  return {
    hostname: url.hostname,
    port: url.port || (url.protocol === "https:" ? 443 : 80),
    path: url.pathname.replace(/\/$/, "") + "/chat/completions",
    method: "POST",
    headers,
  };
}

function handleStreamingChat(res, url, apiKey, chatBody) {
  const payload = Buffer.from(JSON.stringify(chatBody));
  const options = upstreamRequest(url, apiKey);
  options.headers["Content-Length"] = payload.length;
  options.timeout = UPSTREAM_TIMEOUT_MS;
  const state = {
    index: Math.floor(Math.random() * 1e9),
    messageText: "",
    reasoningText: "",
    toolCalls: [],
    usage: null,
    chatId: null,
    finishReason: null,
    failed: null,
  };
  const client = upstreamClient(url);
  const upstream = client.request(options, (up) => {
    if (up.statusCode !== 200) {
      // pass provider errors through: the runner classifies non-2xx bodies itself
      const chunks = [];
      up.on("data", (chunk) => chunks.push(chunk));
      up.on("end", () => {
        res.writeHead(up.statusCode, { "Content-Type": up.headers["content-type"] || "application/json" });
        res.end(Buffer.concat(chunks));
      });
      return;
    }
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    let buffer = "";
    up.setEncoding("utf8");
    up.on("data", (chunk) => {
      buffer += chunk;
      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        let event;
        try {
          event = JSON.parse(data);
        } catch {
          continue;
        }
        if (event.error) {
          state.failed = event.error.message || JSON.stringify(event.error);
          continue;
        }
        if (event.id && !state.chatId) state.chatId = event.id;
        if (event.usage) state.usage = event.usage;
        const choice = event.choices && event.choices[0];
        if (!choice) continue;
        const delta = choice.delta || {};
        if (typeof delta.content === "string") state.messageText += delta.content;
        // reasoning stream naming differs across gateways
        if (typeof delta.reasoning_content === "string") state.reasoningText += delta.reasoning_content;
        if (typeof delta.reasoning === "string") state.reasoningText += delta.reasoning;
        for (const call of delta.tool_calls || []) {
          const slot = state.toolCalls[call.index] || (state.toolCalls[call.index] = { arguments: "" });
          if (call.id) slot.id = call.id;
          if (call.function) {
            if (call.function.name) slot.name = (slot.name || "") + call.function.name;
            if (call.function.arguments) slot.arguments += call.function.arguments;
          }
        }
        if (choice.finish_reason) state.finishReason = choice.finish_reason;
      }
    });
    up.on("end", () => {
      if (state.failed) {
        res.write(
          sseEvent({
            type: "response.failed",
            response: {
              id: state.chatId || "resp_proxy_" + state.index,
              status: "failed",
              output: [],
              usage: mapChatUsage(state.usage),
              error: { code: "upstream_error", message: state.failed },
            },
          }),
        );
      } else {
        res.write(terminalEvent(state.chatId, state, state.finishReason));
      }
      res.write("data: [DONE]\n\n");
      res.end();
    });
    up.on("error", (err) => {
      log("upstream stream error:", err.message);
      res.end();
    });
  });
  upstream.on("error", (err) => {
    log("upstream connect error:", err.message);
    if (!res.headersSent) {
      res.writeHead(502, { "Content-Type": "application/json" });
    }
    res.end(JSON.stringify({ error: { message: "upstream connect failed: " + err.message, type: "proxy_error" } }));
  });
  upstream.end(payload);
}

// Non-streaming fallback (the runner always sets stream:true; kept for manual
// testing with curl).
function handlePlainChat(res, url, apiKey, body) {
  const chatBody = responsesToChatRequest(body);
  chatBody.stream = false;
  delete chatBody.stream_options;
  const payload = Buffer.from(JSON.stringify(chatBody));
  const options = upstreamRequest(url, apiKey);
  options.headers["Content-Length"] = payload.length;
  options.timeout = UPSTREAM_TIMEOUT_MS;
  const upstream = upstreamClient(url).request(options, (up) => {
    const chunks = [];
    up.on("data", (chunk) => chunks.push(chunk));
    up.on("end", () => {
      res.writeHead(up.statusCode, { "Content-Type": up.headers["content-type"] || "application/json" });
      res.end(Buffer.concat(chunks));
    });
  });
  upstream.on("error", (err) => {
    res.writeHead(502, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: { message: err.message } }));
  });
  upstream.end(payload);
}

// ---------- server ----------

function startProxy(config) {
  const opts = config || {};
  const baseRaw = opts.baseURL || "https://api.openai.com/v1";
  const baseURL = new URL(baseRaw);
  const apiKey = opts.apiKey || "";
  // Random per-launch bearer token: only callers that got it (the adapter,
  // passed to the runner via env) may use the proxy, so unrelated local
  // processes cannot ride the user's API key.
  const token = opts.token || crypto.randomBytes(16).toString("hex");

  const server = http.createServer((req, res) => {
    if ((req.headers.authorization || "") !== "Bearer " + token) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: "unauthorized proxy call" } }));
      return;
    }
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      let parsed;
      try {
        parsed = JSON.parse(body || "{}");
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { message: "invalid JSON body: " + err.message } }));
        return;
      }
      if (!parsed.model) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { message: "model is required" } }));
        return;
      }
      if (!apiKey) {
        log("warning: no API key configured (UA_API_KEY)");
      }
      if (parsed.stream) {
        handleStreamingChat(res, baseURL, apiKey, responsesToChatRequest(parsed));
      } else {
        handlePlainChat(res, baseURL, apiKey, parsed);
      }
    });
  });

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      log("listening on 127.0.0.1:" + address.port + " -> " + baseRaw);
      resolve({ port: address.port, token, close: () => server.close() });
    });
  });
}

module.exports = { startProxy };
