import type { SSEEvent } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Clear a session's server-side conversation history (best-effort). */
export async function clearSession(sessionId: string): Promise<void> {
  try {
    await fetch(`${API_URL}/session/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    });
  } catch {
    // non-fatal — the local view is already reset
  }
}

export async function* streamChat(
  query: string,
  sessionId: string,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const response = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: sessionId }),
    signal,
  });

  if (!response.ok || !response.body) {
    yield { type: "error", message: "Failed to connect to API" };
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6)) as SSEEvent;
        } catch {
          // skip malformed lines
        }
      }
    }
  }
}
