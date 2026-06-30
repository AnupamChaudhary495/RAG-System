export interface Citation {
  chunk_id: string;
  source_filename: string;
  page_number: number | null;
  section_heading: string | null;
  token_count?: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
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
