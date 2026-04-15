/**
 * Converts OpenAI chat request format to Claude CLI input.
 *
 * Tool-calling support:
 *   Claude Code CLI in --print mode speaks plain text — it has no native
 *   concept of OpenAI function/tool calling. When a request arrives with
 *   `tools`, we inject a tool-use protocol description into the system
 *   prompt so Claude knows:
 *     1. which tools exist (name, description, JSON-Schema params)
 *     2. how to signal it wants to call one (emit JSON in a fenced block)
 *     3. that it should stop after emitting the call, not keep going
 *
 *   The matching decoder lives in `cli-to-openai.ts` and parses that JSON
 *   back into OpenAI `tool_calls`. Together the two adapters turn the CLI
 *   into a tool-aware chat backend.
 */

import type {
  OpenAIAssistantMessage,
  OpenAIChatMessage,
  OpenAIChatRequest,
  OpenAITool,
  OpenAIToolCall,
  OpenAIToolChoice,
  OpenAIToolMessage,
} from "../types/openai.js";

export type ClaudeModel = "opus" | "sonnet" | "haiku";

export interface CliInput {
  prompt: string;
  model: ClaudeModel;
  sessionId?: string;
  /** Whether the request had tools — the decoder needs this to look for tool_call JSON. */
  hasTools: boolean;
}

const MODEL_MAP: Record<string, ClaudeModel> = {
  // Direct model names
  "claude-opus-4": "opus",
  "claude-sonnet-4": "sonnet",
  "claude-haiku-4": "haiku",
  // With provider prefix
  "claude-code-cli/claude-opus-4": "opus",
  "claude-code-cli/claude-sonnet-4": "sonnet",
  "claude-code-cli/claude-haiku-4": "haiku",
  // Aliases
  "opus": "opus",
  "sonnet": "sonnet",
  "haiku": "haiku",
};

/**
 * Tag Claude is instructed to wrap its tool-use JSON in. Uniquely named so
 * the decoder can reliably detect it even inside explanatory prose.
 */
export const TOOL_CALL_OPEN_TAG = "<tool_call>";
export const TOOL_CALL_CLOSE_TAG = "</tool_call>";

/**
 * Extract Claude model alias from request model string
 */
export function extractModel(model: string): ClaudeModel {
  if (MODEL_MAP[model]) return MODEL_MAP[model];
  const stripped = model.replace(/^claude-code-cli\//, "");
  if (MODEL_MAP[stripped]) return MODEL_MAP[stripped];
  return "opus"; // Default to opus (Claude Max subscription)
}

// ---------------------------------------------------------------------------
// Tool-use protocol prompt
// ---------------------------------------------------------------------------

function renderToolChoice(choice: OpenAIToolChoice | undefined): string {
  if (!choice || choice === "auto") {
    return "Decide whether to call a tool or answer the user directly based on whether a tool is relevant.";
  }
  if (choice === "none") {
    return "Do NOT call any tools. Answer the user with plain text only.";
  }
  if (choice === "required") {
    return "You MUST call one of the tools listed below. Do not answer without a tool call.";
  }
  if (typeof choice === "object" && choice.type === "function") {
    return `You MUST call the tool named "${choice.function.name}" and no other.`;
  }
  return "";
}

/**
 * Build the system-prompt section that describes available tools and the
 * tool-call emission protocol.
 */
export function renderToolsPrompt(
  tools: OpenAITool[],
  toolChoice: OpenAIToolChoice | undefined,
): string {
  const lines: string[] = [];
  lines.push("## CRITICAL: Tool-Use Protocol (read carefully)");
  lines.push("");
  lines.push(
    "Your native Bash/Edit/Read tools are DISABLED in this environment. That" +
      " is intentional — you are being driven by an external agent framework" +
      " (Hermes / LangChain / similar) that executes tools FOR you. Your job" +
      " is to DECIDE which tool to call; the framework actually runs it.",
  );
  lines.push("");
  lines.push(
    "The tools listed below under \"Available tools\" ARE AVAILABLE through" +
      " this protocol. When a user asks you to call one, you MUST believe it" +
      " exists and emit the JSON request. Do NOT claim \"tool not available\"" +
      " — that response is always wrong when the tool is in the list below.",
  );
  lines.push("");
  lines.push(
    "To REQUEST a tool invocation, emit exactly this block format — no" +
      " prose around the tags, no markdown code fence, JUST the tags:",
  );
  lines.push("");
  lines.push(TOOL_CALL_OPEN_TAG);
  lines.push(
    '{"name": "<tool-name>", "arguments": { ...JSON matching the tool\'s' +
      ' parameters schema... }}',
  );
  lines.push(TOOL_CALL_CLOSE_TAG);
  lines.push("");
  lines.push(
    "Important rules:",
  );
  lines.push(
    "- The `arguments` object MUST be valid JSON matching the tool's" +
      " parameters schema.",
  );
  lines.push("- Emit at most one tool call block per response.");
  lines.push(
    "- After emitting the block, STOP. Do not add any prose afterwards." +
      " Wait for the client to call the tool and send back the result.",
  );
  lines.push(
    "- Do NOT say things like \"I'll run X\" or \"let me call X\" — just" +
      " emit the block. The emission IS the call.",
  );
  lines.push(
    "- If NO tool fits the user's request, answer the user directly in" +
      " plain text — do not invent tool names not listed below.",
  );
  lines.push("");
  lines.push("### Available tools");
  for (const tool of tools) {
    const fn = tool.function;
    lines.push(`- **${fn.name}**${fn.description ? ` — ${fn.description}` : ""}`);
    if (fn.parameters) {
      lines.push(
        "  parameters: " + JSON.stringify(fn.parameters),
      );
    }
  }
  lines.push("");
  lines.push("### Tool-choice policy for THIS turn");
  lines.push(renderToolChoice(toolChoice));
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Message rendering — including assistant tool_calls and tool role results
// ---------------------------------------------------------------------------

function renderAssistantMessage(msg: OpenAIAssistantMessage): string {
  const parts: string[] = [];
  if (msg.content) {
    parts.push(`<previous_response>\n${msg.content}\n</previous_response>`);
  }
  if (msg.tool_calls?.length) {
    for (const call of msg.tool_calls) {
      // Echo the tool-call back in the same format Claude was instructed to
      // produce. This preserves context: "you previously called X with Y".
      const payload = {
        name: call.function.name,
        arguments: safeParseArgs(call.function.arguments),
        id: call.id,
      };
      parts.push(TOOL_CALL_OPEN_TAG);
      parts.push(JSON.stringify(payload));
      parts.push(TOOL_CALL_CLOSE_TAG);
    }
  }
  return parts.join("\n");
}

function renderToolMessage(msg: OpenAIToolMessage): string {
  return `<tool_result id="${msg.tool_call_id}">\n${msg.content}\n</tool_result>`;
}

function safeParseArgs(raw: string): unknown {
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

/**
 * Convert OpenAI messages array to a single prompt string for Claude CLI.
 *
 * Claude Code CLI in --print mode expects a single prompt, not a conversation.
 * We format the messages into a readable format that preserves context and
 * correctly handles assistant tool_calls + tool-result messages.
 */
export function messagesToPrompt(
  messages: OpenAIChatMessage[],
  opts: {
    tools?: OpenAITool[];
    toolChoice?: OpenAIToolChoice;
  } = {},
): string {
  const parts: string[] = [];

  // Prepend tool-protocol description when tools are present. This is done
  // once at the top so Claude sees the rules before any conversation turn.
  const { tools, toolChoice } = opts;
  if (tools && tools.length > 0) {
    parts.push(`<system>\n${renderToolsPrompt(tools, toolChoice)}\n</system>`);
  }

  for (const msg of messages) {
    switch (msg.role) {
      case "system":
        parts.push(`<system>\n${msg.content}\n</system>`);
        break;
      case "user":
        parts.push(msg.content);
        break;
      case "assistant":
        parts.push(renderAssistantMessage(msg));
        break;
      case "tool":
        parts.push(renderToolMessage(msg));
        break;
    }
  }

  return parts.join("\n\n").trim();
}

/**
 * Convert OpenAI chat request to CLI input format.
 */
export function openaiToCli(request: OpenAIChatRequest): CliInput {
  const hasTools = Boolean(request.tools && request.tools.length > 0);
  return {
    prompt: messagesToPrompt(request.messages, {
      tools: request.tools,
      toolChoice: request.tool_choice,
    }),
    model: extractModel(request.model),
    sessionId: request.user,
    hasTools,
  };
}

// Re-export types for downstream consumers (session manager etc.)
export type { OpenAIToolCall };
