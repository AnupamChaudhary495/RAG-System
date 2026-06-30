"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Citation, Message, SSEEvent } from "@/types";
import { resolveCitations } from "@/lib/citations";
import { streamChat } from "@/lib/stream";
import { MessageBubble } from "./Message";

function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}

interface Props {
  sessionId: string;
}

export function ChatInterface({ sessionId }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(async () => {
    const query = input.trim();
    if (!query || streaming) return;

    setInput("");
    setStreaming(true);

    const userMsg: Message = {
      id: generateId(),
      role: "user",
      content: query,
      citations: [],
    };

    const assistantId = generateId();
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      citations: [],
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    let fullContent = "";
    let chunkIds: string[] = [];

    try {
      for await (const event of streamChat(query, sessionId)) {
        const ev = event as SSEEvent;

        if (ev.type === "token") {
          fullContent += ev.content;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: fullContent } : m,
            ),
          );
        } else if (ev.type === "metadata") {
          chunkIds = ev.source_chunk_ids;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    confidence_score: ev.confidence_score,
                    retry_count: ev.retry_count,
                    router_decision: ev.router_decision,
                  }
                : m,
            ),
          );
        } else if (ev.type === "done") {
          break;
        } else if (ev.type === "error") {
          fullContent += `\n\n_Error: ${ev.message}_`;
          break;
        }
      }
    } catch (err) {
      fullContent += "\n\n_Connection error. Please try again._";
    }

    const citations: Citation[] = await resolveCitations(chunkIds);

    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantId
          ? { ...m, content: fullContent, citations, isStreaming: false }
          : m,
      ),
    );
    setStreaming(false);
  }, [input, sessionId, streaming]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            Ask anything about your documents…
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 dark:border-gray-700 px-4 py-4">
        <div className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
            rows={1}
            disabled={streaming}
            className="flex-1 resize-none rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 px-4 py-3 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 overflow-hidden"
            style={{ maxHeight: "160px" }}
          />
          <button
            onClick={sendMessage}
            disabled={streaming || !input.trim()}
            className="rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white px-4 py-3 text-sm font-medium transition-colors"
          >
            {streaming ? "…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
