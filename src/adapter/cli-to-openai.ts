/**
 * Converts Claude CLI output to OpenAI-compatible response format.
 *
 * Tool-calling extraction:
 *   When the request had `tools`, the openai-to-cli encoder instructs
 *   Claude to emit tool calls wrapped in <tool_call>…</tool_call> tags
 *   (see {@link ../adapter/openai-to-cli.ts}). This decoder looks for those
 *   tags in the response text, parses the inner JSON, and returns the
 *   result as OpenAI-format `tool_calls` so clients (Hermes, Clawdbot,
 *   anything speaking OpenAI) can handle them natively.
 *
 *   If parsing fails (malformed JSON, missing fields), we fall through to
 *   plain-text content so the client still gets something useful.
 */

import type { ClaudeCliAssistant, ClaudeCliResult } from "../types/claude-cli.js";
import type {
  OpenAIChatResponse,
  OpenAIChatChunk,
  OpenAIToolCall,
  OpenAIFinishReason,
} from "../types/openai.js";
import {
  TOOL_CALL_OPEN_TAG,
  TOOL_CALL_CLOSE_TAG,
} from "./openai-to-cli.js";

/**
 * Extract text content from Claude CLI assistant message
 */
export function extractTextContent(message: ClaudeCliAssistant): string {
  return message.message.content
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("");
}

// ---------------------------------------------------------------------------
// Tool-call extraction from Claude's text output
// ---------------------------------------------------------------------------

export interface ExtractedToolCalls {
  /** Text with tool_call blocks stripped. May be empty. */
  text: string;
  /** Tool calls parsed out, in order. Empty array when none present. */
  toolCalls: OpenAIToolCall[];
}

let toolCallCounter = 0;
function nextToolCallId(): string {
  toolCallCounter = (toolCallCounter + 1) % 1_000_000;
  return `call_${Date.now().toString(36)}_${toolCallCounter}`;
}

/**
 * Find all <tool_call>…</tool_call> blocks in `text`, parse each as JSON,
 * and return OpenAI-format tool_calls plus the remaining text with the
 * blocks stripped.
 *
 * Robust against:
 *   - Extra whitespace / newlines inside the tags
 *   - Missing `id` (we generate one)
 *   - Missing or non-object `arguments` (normalized to empty object)
 *   - Malformed JSON (skipped silently, text kept as-is with tags removed)
 */
export function extractToolCalls(text: string): ExtractedToolCalls {
  const calls: OpenAIToolCall[] = [];
  const open = TOOL_CALL_OPEN_TAG;
  const close = TOOL_CALL_CLOSE_TAG;

  const out: string[] = [];
  let cursor = 0;
  while (cursor < text.length) {
    const start = text.indexOf(open, cursor);
    if (start === -1) {
      out.push(text.slice(cursor));
      break;
    }
    // Keep any text before the block
    out.push(text.slice(cursor, start));
    const end = text.indexOf(close, start + open.length);
    if (end === -1) {
      // Unterminated block: keep raw text, stop scanning
      out.push(text.slice(start));
      break;
    }
    const payload = text.slice(start + open.length, end).trim();
    cursor = end + close.length;

    try {
      const parsed = JSON.parse(payload);
      if (!parsed || typeof parsed !== "object") continue;
      const name = typeof parsed.name === "string" ? parsed.name : null;
      if (!name) continue;
      const argsObj =
        parsed.arguments && typeof parsed.arguments === "object"
          ? parsed.arguments
          : {};
      const id =
        typeof parsed.id === "string" && parsed.id.length > 0
          ? parsed.id
          : nextToolCallId();
      calls.push({
        id,
        type: "function",
        function: {
          name,
          arguments: JSON.stringify(argsObj),
        },
      });
    } catch {
      // Malformed JSON inside block — drop silently; the surrounding text
      // is preserved via `out` so the client still sees the assistant's
      // prose (minus the broken block).
      continue;
    }
  }

  return {
    text: out.join("").trim(),
    toolCalls: calls,
  };
}

// ---------------------------------------------------------------------------
// Streaming chunk conversion
// ---------------------------------------------------------------------------

/**
 * Convert Claude CLI assistant message to OpenAI streaming chunk.
 *
 * When `extractTools` is true, the chunk text is inspected for tool_call
 * blocks; if found, an OpenAI tool_calls delta is emitted and the block
 * is stripped from the text content.
 */
export function cliToOpenaiChunk(
  message: ClaudeCliAssistant,
  requestId: string,
  isFirst: boolean = false,
  extractTools: boolean = false,
): OpenAIChatChunk {
  const rawText = extractTextContent(message);
  const { text, toolCalls } = extractTools
    ? extractToolCalls(rawText)
    : { text: rawText, toolCalls: [] as OpenAIToolCall[] };

  const finishReason: OpenAIFinishReason = message.message.stop_reason
    ? toolCalls.length > 0
      ? "tool_calls"
      : "stop"
    : null;

  return {
    id: `chatcmpl-${requestId}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: normalizeModelName(message.message.model),
    choices: [
      {
        index: 0,
        delta: {
          role: isFirst ? "assistant" : undefined,
          content: text ? text : undefined,
          tool_calls:
            toolCalls.length > 0
              ? toolCalls.map((c, index) => ({
                  index,
                  id: c.id,
                  type: "function",
                  function: {
                    name: c.function.name,
                    arguments: c.function.arguments,
                  },
                }))
              : undefined,
        },
        finish_reason: finishReason,
      },
    ],
  };
}

/**
 * Create a final "done" chunk for streaming
 */
export function createDoneChunk(
  requestId: string,
  model: string,
  finishReason: OpenAIFinishReason = "stop",
): OpenAIChatChunk {
  return {
    id: `chatcmpl-${requestId}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: normalizeModelName(model),
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: finishReason,
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Non-streaming response
// ---------------------------------------------------------------------------

/**
 * Convert Claude CLI result to OpenAI non-streaming response.
 *
 * When `extractTools` is true, the result text is scanned for tool_call
 * blocks and returned in `message.tool_calls` with content stripped; the
 * finish_reason becomes "tool_calls" to signal the client.
 */
export function cliResultToOpenai(
  result: ClaudeCliResult,
  requestId: string,
  extractTools: boolean = false,
): OpenAIChatResponse {
  const modelName = result.modelUsage
    ? Object.keys(result.modelUsage)[0]
    : "claude-sonnet-4";

  const { text, toolCalls } = extractTools
    ? extractToolCalls(result.result)
    : { text: result.result, toolCalls: [] as OpenAIToolCall[] };

  return {
    id: `chatcmpl-${requestId}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: normalizeModelName(modelName),
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: toolCalls.length > 0 ? (text || null) : text,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        },
        finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop",
      },
    ],
    usage: {
      prompt_tokens: result.usage?.input_tokens || 0,
      completion_tokens: result.usage?.output_tokens || 0,
      total_tokens:
        (result.usage?.input_tokens || 0) + (result.usage?.output_tokens || 0),
    },
  };
}

/**
 * Normalize Claude model names to a consistent format
 * e.g., "claude-sonnet-4-5-20250929" -> "claude-sonnet-4"
 */
function normalizeModelName(model: string): string {
  if (model.includes("opus")) return "claude-opus-4";
  if (model.includes("sonnet")) return "claude-sonnet-4";
  if (model.includes("haiku")) return "claude-haiku-4";
  return model;
}
