# unreal-agent-acp

[Agent Client Protocol](https://agentclientprotocol.com) (ACP) bridge for the
[unreal-agent](https://github.com/unreallabsai/unreal-agent) runner, so the
harness can be used from Zed (or any other ACP client) against plain
OpenAI-compatible `chat/completions` gateways.

```
Zed (ACP) ── ndjson JSON-RPC ── adapter.js ── spawns per prompt ──> unreal-agent-runner
                                               │
                                               └── responses-proxy.js (127.0.0.1:<ephemeral>)
                                                     Responses API  ⇄  /chat/completions
```

The upstream runner only speaks the OpenAI **Responses API**. Many
OpenAI-compatible deployments - vLLM, aggregators, corporate LLM gateways -
expose `/v1/chat/completions` but not `/v1/responses`. This project adds two
pieces:

- `src/proxy.js` - a local, zero-dependency translator. It accepts the
  Responses API request the runner produces, rewrites it as a streaming
  `chat/completions` request, and translates the reply back into the minimal
  Responses SSE event the runner consumes (one terminal
  `response.completed` / `response.incomplete` / `response.failed` event with
  the full `output` and `usage`).
- `src/adapter.js` - the ACP agent. It implements `initialize`, `session/new`,
  `session/prompt`, `session/cancel` and `session/set_config_option` over
  newline-delimited JSON-RPC and drives one `unreal-agent-runner` process per
  prompt, translating the runner's persisted session-item JSONL stream into
  `session/update` notifications (`agent_message_chunk`, `agent_thought_chunk`,
  `tool_call`, `tool_call_update`).

Sessions are durable: the ACP session id is reused as the runner's persisted
session id, so consecutive prompts in the same session continue the same
history (`~/.local/state/unreal-agent/sessions`).

### Display in the client

The runner is a batch process: a turn's model output is only available when
the whole LLM response is persisted, so the adapter replays it in a
client-friendly way:

- reasoning summaries are emitted as paced `agent_thought_chunk` notifications
  (small chunks within a ~2s budget), giving a live-looking Thinking block;
- the assistant answer is emitted as paced `agent_message_chunk` text;
- bash tool cards keep their command as the title across updates and get the
  command output (stdout/stderr/exit code, extracted from the runner's shell
  operation state) attached to the completed `tool_call_update`;
- per-LLM-turn token usage rides in `_meta.unreal.usage` of the message
  chunk, task totals in `_meta.unreal.usage` of the prompt result.

## Requirements

- Node.js >= 20 (no npm dependencies)
- Go 1.27+ to build the upstream runner:
  ```sh
  go install github.com/unreallabsai/unreal-agent/cmd/unreal-agent-runner@latest
  ```
  The adapter looks for the binary in `$UA_RUNNER`, `~/go/bin`, `/opt/homebrew/bin`,
  `/usr/local/bin`, then `$PATH`.
- An OpenAI-compatible chat/completions endpoint and API key.

## Configuration (environment)

| Variable | Default | Purpose |
| --- | --- | --- |
| `UA_BASE_URL` | `https://api.openai.com/v1` | chat/completions gateway base URL |
| `UA_API_KEY` | - | API key forwarded as `Authorization: Bearer ...` |
| `UA_RUNNER` | discovery order above | path to `unreal-agent-runner` |
| `UA_DEFAULT_MODEL` | `glm53-flash` | model for new sessions |
| `UA_DEFAULT_THINKING` | `high` | thinking level for new sessions (`low`..`max`) |
| `UA_SESSION_DIRECTORY` | `~/.local/state/unreal-agent/sessions` | runner session store |

The model list offered by `session/new` mirrors common OpenAI-compatible
gateways; edit `DEFAULT_MODELS` in `src/adapter.js` to match yours.

## Zed setup

Add to `~/.config/zed/settings.json` (see `zed-settings.example.json`):

```json
{
  "agent_servers": {
    "Unreal Agent": {
      "type": "custom",
      "command": "/bin/zsh",
      "args": [
        "-c",
        "source ~/.zshrc >/dev/null 2>&1; export UA_API_KEY='...'; export UA_BASE_URL='https://your-gateway/v1'; exec node /path/to/unreal-agent-acp/src/adapter.js"
      ]
    }
  }
}
```

The `zsh -c` wrapper is where the API key is injected; it never appears in
adapter logs or in the ACP traffic.

### Logging the traffic (tee)

`adapter.js` is a normal stdio ACP agent, so it can sit behind any logging
tee. With a tee such as the one used by existing harness integrations:

```
node acp-tee.js --log ~/.local/state/harnessmeter/unreal-acp.ndjson --tag unreal -- node .../src/adapter.js
```

The ndjson log then contains everything worth metering per task:

- `initialize` result - agent name/version;
- `session/prompt` params - the task text;
- `session/update` `_meta.unreal.usage` - per-LLM-turn token usage
  (`input_tokens`, `cached_input_tokens`, `output_tokens`, `reasoning_tokens`);
- final `session/prompt` result - `stopReason` plus `_meta.unreal.usage` with
  task totals and the completion timestamp from the tee.

## Smoke test

```sh
UA_BASE_URL=https://your-gateway/v1 UA_API_KEY=... npm run smoke
```

The script drives a full session (initialize, session/new, a prompt that
forces a bash tool call), asserts the update flow and usage metadata, and
verifies the tee log when `UA_TEE` points to `acp-tee.js`.

## Limitations

- Prompts are text-only (the runner's request schema is text-only).
- Tool results carrying images are flattened to a text placeholder by the
  proxy; hosted tools (`web_search`) are skipped.
- `session/load` is not implemented (fresh history per Zed session window).
- The chat backend's reasoning stream (e.g. `reasoning_content`) is surfaced
  as a reasoning summary item, not replayed as provider-native reasoning.
- Thinking/answer "streaming" is replay pacing over batch turn output (the
  runner has no partial-message streaming; `include_partial_messages` is
  accepted but ignored upstream).

## License

MIT
