export interface Citation {
  chunk_id: string;
  source_filename: string;
  page_number: number | null;
  section_heading: string | null;
  token_count?: number;
}

/** Lifecycle of an assistant message. */
export type MessageStatus = "thinking" | "streaming" | "done" | "error";

export interface Message {
  id: string;
  role: "user" | "assistant";
  /** For assistant messages this holds the RAW streamed buffer (JSON). */
  content: string;
  citations: Citation[];
  status?: MessageStatus;
  /** True until the first token arrives — drives the "thinking…" indicator. */
  isStreaming?: boolean;
  confidence_score?: number;
  retry_count?: number;
  router_decision?: string;
}

export type SSEEvent =
  | { type: "token"; content: string }
  | {
      type: "metadata";
      source_chunk_ids: string[];
      confidence_score: number;
      retry_count: number;
      router_decision: string;
    }
  | { type: "done" }
  | { type: "error"; message: string };
