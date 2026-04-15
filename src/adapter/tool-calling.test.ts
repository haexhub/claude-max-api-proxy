/**
 * Tests for OpenAI tool-calling translation:
 *   - openaiToCli / messagesToPrompt: tools -> system prompt
 *   - extractToolCalls: <tool_call> JSON -> OpenAI tool_calls array
 *
 * Run via `pnpm build && pnpm test` (the build produces .js in dist/ which
 * `node --test` then picks up).
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  messagesToPrompt,
  openaiToCli,
  TOOL_CALL_OPEN_TAG,
  TOOL_CALL_CLOSE_TAG,
} from "./openai-to-cli.js";
import { extractToolCalls } from "./cli-to-openai.js";
import type { OpenAITool } from "../types/openai.js";

const sampleTool: OpenAITool = {
  type: "function",
  function: {
    name: "list_indicators",
    description: "List all FWBG indicator plugins",
    parameters: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
};

// ---------------------------------------------------------------------------
// encoder
// ---------------------------------------------------------------------------

test("openaiToCli passes through hasTools flag", () => {
  const cli = openaiToCli({
    model: "claude-sonnet-4",
    messages: [{ role: "user", content: "ping" }],
    tools: [sampleTool],
  });
  assert.equal(cli.hasTools, true);
  assert.match(cli.prompt, /Tool-Use Protocol/);
  assert.match(cli.prompt, /list_indicators/);
  assert.match(cli.prompt, new RegExp(TOOL_CALL_OPEN_TAG));
});

test("openaiToCli without tools leaves prompt clean", () => {
  const cli = openaiToCli({
    model: "claude-sonnet-4",
    messages: [{ role: "user", content: "hi" }],
  });
  assert.equal(cli.hasTools, false);
  assert.doesNotMatch(cli.prompt, /Tool-Use Protocol/);
});

test("messagesToPrompt handles tool role as tool_result block", () => {
  const prompt = messagesToPrompt([
    { role: "user", content: "list indicators" },
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call_abc",
          type: "function",
          function: { name: "list_indicators", arguments: "{}" },
        },
      ],
    },
    { role: "tool", tool_call_id: "call_abc", content: '["rsi","macd"]' },
  ]);
  assert.match(prompt, new RegExp(TOOL_CALL_OPEN_TAG));
  assert.match(prompt, /"name":"list_indicators"/);
  assert.match(prompt, /<tool_result id="call_abc">/);
  assert.match(prompt, /\["rsi","macd"\]/);
});

test("messagesToPrompt respects tool_choice 'required'", () => {
  const prompt = messagesToPrompt(
    [{ role: "user", content: "go" }],
    { tools: [sampleTool], toolChoice: "required" },
  );
  assert.match(prompt, /You MUST call one of the tools/);
});

test("messagesToPrompt respects tool_choice specific function", () => {
  const prompt = messagesToPrompt(
    [{ role: "user", content: "go" }],
    {
      tools: [sampleTool],
      toolChoice: { type: "function", function: { name: "list_indicators" } },
    },
  );
  assert.match(prompt, /You MUST call the tool named "list_indicators"/);
});

// ---------------------------------------------------------------------------
// decoder
// ---------------------------------------------------------------------------

test("extractToolCalls parses a single tool_call block", () => {
  const text = `Let me check.\n${TOOL_CALL_OPEN_TAG}\n{"name":"list_indicators","arguments":{"category":"trend"}}\n${TOOL_CALL_CLOSE_TAG}\n`;
  const { text: rest, toolCalls } = extractToolCalls(text);
  assert.equal(toolCalls.length, 1);
  assert.equal(toolCalls[0].function.name, "list_indicators");
  assert.deepEqual(JSON.parse(toolCalls[0].function.arguments), {
    category: "trend",
  });
  assert.equal(rest, "Let me check.");
});

test("extractToolCalls parses multiple blocks in order", () => {
  const text = `${TOOL_CALL_OPEN_TAG}{"name":"a","arguments":{}}${TOOL_CALL_CLOSE_TAG}\nthinking\n${TOOL_CALL_OPEN_TAG}{"name":"b","arguments":{"x":1}}${TOOL_CALL_CLOSE_TAG}`;
  const { text: rest, toolCalls } = extractToolCalls(text);
  assert.equal(toolCalls.length, 2);
  assert.equal(toolCalls[0].function.name, "a");
  assert.equal(toolCalls[1].function.name, "b");
  assert.match(rest, /thinking/);
});

test("extractToolCalls normalizes missing arguments", () => {
  const text = `${TOOL_CALL_OPEN_TAG}{"name":"ping"}${TOOL_CALL_CLOSE_TAG}`;
  const { toolCalls } = extractToolCalls(text);
  assert.equal(toolCalls.length, 1);
  assert.equal(toolCalls[0].function.arguments, "{}");
});

test("extractToolCalls preserves custom id if given", () => {
  const text = `${TOOL_CALL_OPEN_TAG}{"id":"call_xyz","name":"ping","arguments":{}}${TOOL_CALL_CLOSE_TAG}`;
  const { toolCalls } = extractToolCalls(text);
  assert.equal(toolCalls[0].id, "call_xyz");
});

test("extractToolCalls generates id when missing", () => {
  const text = `${TOOL_CALL_OPEN_TAG}{"name":"ping","arguments":{}}${TOOL_CALL_CLOSE_TAG}`;
  const { toolCalls } = extractToolCalls(text);
  assert.match(toolCalls[0].id, /^call_[a-z0-9]+_\d+$/);
});

test("extractToolCalls drops malformed JSON blocks silently", () => {
  const text = `before ${TOOL_CALL_OPEN_TAG}{not valid json}${TOOL_CALL_CLOSE_TAG} after`;
  const { text: rest, toolCalls } = extractToolCalls(text);
  assert.equal(toolCalls.length, 0);
  assert.match(rest, /before/);
  assert.match(rest, /after/);
});

test("extractToolCalls skips blocks without a name", () => {
  const text = `${TOOL_CALL_OPEN_TAG}{"arguments":{}}${TOOL_CALL_CLOSE_TAG}`;
  const { toolCalls } = extractToolCalls(text);
  assert.equal(toolCalls.length, 0);
});

test("extractToolCalls handles unterminated block gracefully", () => {
  const text = `before ${TOOL_CALL_OPEN_TAG}{"name":"x"} with no close tag ever`;
  const { text: rest, toolCalls } = extractToolCalls(text);
  assert.equal(toolCalls.length, 0);
  assert.match(rest, /before/);
  // Unterminated block kept as-is so caller can surface it
  assert.match(rest, new RegExp(TOOL_CALL_OPEN_TAG));
});

test("extractToolCalls on plain text returns it unchanged", () => {
  const { text, toolCalls } = extractToolCalls("Just a regular answer.");
  assert.equal(toolCalls.length, 0);
  assert.equal(text, "Just a regular answer.");
});
