/**
 * Types for OpenAI-compatible API (chat completions spec, v1)
 * Aligned with OpenAI Platform reference:
 *   https://platform.openai.com/docs/api-reference/chat/create
 *
 * This file adds full typing for the function/tool-calling flow (tools,
 * tool_calls, tool_choice, and the assistant/tool roles), so that clients
 * like Hermes Agent which speak standard OpenAI can drive tool use through
 * the proxy.
 */

// ---------------------------------------------------------------------------
// Tool / function definitions (request -> proxy -> system prompt)
// ---------------------------------------------------------------------------

export interface OpenAIFunctionDefinition {
  name: string;
  description?: string;
  /** JSON Schema describing the function parameters. */
  parameters?: Record<string, unknown>;
}

export interface OpenAITool {
  type: "function";
  function: OpenAIFunctionDefinition;
}

export type OpenAIToolChoice =
  | "none"
  | "auto"
  | "required"
  | { type: "function"; function: { name: string } };

// ---------------------------------------------------------------------------
// Tool calls (response -> OpenAI format back to client)
// ---------------------------------------------------------------------------

export interface OpenAIToolCallFunction {
  name: string;
  /** JSON-encoded arguments string (OpenAI convention). */
  arguments: string;
}

export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: OpenAIToolCallFunction;
}

/** Streaming delta for a tool call: fields can arrive piece-by-piece. */
export interface OpenAIToolCallDelta {
  /** Index of the tool call within the response (OpenAI streams per-index). */
  index: number;
  id?: string;
  type?: "function";
  function?: {
    name?: string;
    arguments?: string;
  };
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

export interface OpenAISystemMessage {
  role: "system";
  content: string;
  name?: string;
}

export interface OpenAIUserMessage {
  role: "user";
  content: string;
  name?: string;
}

export interface OpenAIAssistantMessage {
  role: "assistant";
  /** Content is nullable when the assistant is only making tool calls. */
  content: string | null;
  name?: string;
  tool_calls?: OpenAIToolCall[];
}

/** Tool-result message sent back after the client executed a tool call. */
export interface OpenAIToolMessage {
  role: "tool";
  /** Must match the id of a tool_call made by the assistant. */
  tool_call_id: string;
  content: string;
}

export type OpenAIChatMessage =
  | OpenAISystemMessage
  | OpenAIUserMessage
  | OpenAIAssistantMessage
  | OpenAIToolMessage;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

export interface OpenAIChatRequest {
  model: string;
  messages: OpenAIChatMessage[];
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  /** Used for session mapping. */
  user?: string;

  // Function / tool calling
  tools?: OpenAITool[];
  tool_choice?: OpenAIToolChoice;
  parallel_tool_calls?: boolean;
}

// ---------------------------------------------------------------------------
// Response (non-streaming)
// ---------------------------------------------------------------------------

export type OpenAIFinishReason =
  | "stop"
  | "length"
  | "content_filter"
  | "tool_calls"
  | null;

export interface OpenAIChatResponseChoice {
  index: number;
  message: {
    role: "assistant";
    content: string | null;
    tool_calls?: OpenAIToolCall[];
  };
  finish_reason: OpenAIFinishReason;
}

export interface OpenAIChatResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: OpenAIChatResponseChoice[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// ---------------------------------------------------------------------------
// Streaming (SSE chunks)
// ---------------------------------------------------------------------------

export interface OpenAIChatChunkDelta {
  role?: "assistant";
  content?: string;
  tool_calls?: OpenAIToolCallDelta[];
}

export interface OpenAIChatChunkChoice {
  index: number;
  delta: OpenAIChatChunkDelta;
  finish_reason: OpenAIFinishReason;
}

export interface OpenAIChatChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: OpenAIChatChunkChoice[];
}

// ---------------------------------------------------------------------------
// Models listing / error
// ---------------------------------------------------------------------------

export interface OpenAIModel {
  id: string;
  object: "model";
  owned_by: string;
  created?: number;
}

export interface OpenAIModelList {
  object: "list";
  data: OpenAIModel[];
}

export interface OpenAIError {
  error: {
    message: string;
    type: string;
    code: string | null;
  };
}
