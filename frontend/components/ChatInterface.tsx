"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Citation, Message, SSEEvent } from "@/types";
import type { Conversation } from "@/lib/conversations";
import { resolveCitations } from "@/lib/citations";
import { streamChat } from "@/lib/stream";
import { MessageBubble } from "./Message";
import {
  ArrowDownIcon,
  SendIcon,
  SidebarIcon,
  SparkIcon,
  StopIcon,
} from "./icons";

function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}

const EXAMPLES = [
  "What is RAG and how does hybrid retrieval work?",
  "How does reciprocal rank fusion combine results?",
  "Explain the role of the cross-encoder reranker.",
  "What are the phases of the system architecture?",
];

interface Props {
  conversation: Conversation;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  onPersist: (id: string, messages: Message[]) => void;
  onStreamingChange: (streaming: boolean) => void;
}

export function ChatInterface({
  conversation,
  sidebarOpen,
  onToggleSidebar,
  onPersist,
  onStreamingChange,
}: Props) {
  const sessionId = conversation.id;
  const [messages, setMessages] = useState<Message[]>(conversation.messages);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [atBottom, setAtBottom] = useState(true);

  const messagesRef = useRef<Message[]>(conversation.messages);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => onStreamingChange(streaming), [streaming, onStreamingChange]);

  const commit = useCallback(
    (next: Message[]) => {
      messagesRef.current = next;
      onPersist(sessionId, next);
    },
    [onPersist, sessionId],
  );

  const update = useCallback(
    (updater: (prev: Message[]) => Message[]) => {
      setMessages((prev) => {
        const next = updater(prev);
        messagesRef.current = next;
        return next;
      });
    },
    [],
  );

  const patch = useCallback(
    (id: string, next: Partial<Message>) => {
      update((prev) => prev.map((m) => (m.id === id ? { ...m, ...next } : m)));
    },
    [update],
  );

  // Auto-scroll to newest content while the user is anchored to the bottom.
  useEffect(() => {
    if (atBottom) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, atBottom]);

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    setAtBottom(distance < 80);
  }, []);

  const scrollToBottom = () => {
    setAtBottom(true);
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const autoGrow = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  const sendMessage = useCallback(
    async (override?: string) => {
      const query = (override ?? input).trim();
      if (!query || streaming) return;

      setInput("");
      requestAnimationFrame(autoGrow);
      setStreaming(true);
      setAtBottom(true);

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
        status: "thinking",
        isStreaming: true,
      };

      update((prev) => [...prev, userMsg, assistantMsg]);
      commit(messagesRef.current); // persist user turn + title immediately

      const controller = new AbortController();
      abortRef.current = controller;

      let raw = "";
      let chunkIds: string[] = [];
      let sawToken = false;
      let aborted = false;

      try {
        for await (const event of streamChat(query, sessionId, controller.signal)) {
          const ev = event as SSEEvent;
          if (ev.type === "token") {
            raw += ev.content;
            if (!sawToken) {
              sawToken = true;
              patch(assistantId, { status: "streaming" });
            }
            patch(assistantId, { content: raw });
          } else if (ev.type === "metadata") {
            chunkIds = ev.source_chunk_ids;
            patch(assistantId, {
              confidence_score: ev.confidence_score,
              retry_count: ev.retry_count,
              router_decision: ev.router_decision,
            });
          } else if (ev.type === "done") {
            break;
          } else if (ev.type === "error") {
            patch(assistantId, {
              content: `⚠️ ${ev.message}`,
              status: "error",
              isStreaming: false,
            });
            commit(messagesRef.current);
            setStreaming(false);
            abortRef.current = null;
            return;
          }
        }
      } catch (err) {
        if ((err as { name?: string })?.name === "AbortError") {
          aborted = true;
        } else {
          patch(assistantId, {
            content: raw || "⚠️ Connection error. Please try again.",
            status: "error",
            isStreaming: false,
          });
          commit(messagesRef.current);
          setStreaming(false);
          abortRef.current = null;
          return;
        }
      }

      const citations: Citation[] = aborted
        ? []
        : await resolveCitations(chunkIds);

      patch(assistantId, {
        content: raw || (aborted ? "_Stopped._" : ""),
        citations,
        status: "done",
        isStreaming: false,
      });
      commit(messagesRef.current);
      setStreaming(false);
      abortRef.current = null;
    },
    [input, streaming, sessionId, autoGrow, update, patch, commit],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const empty = messages.length === 0;

  return (
    <div className="relative flex h-screen flex-1 flex-col">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-white/8 bg-ink-950/70 backdrop-blur-xl">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <button
              onClick={onToggleSidebar}
              aria-label="Toggle sidebar"
              className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-white/8 hover:text-slate-200"
            >
              <SidebarIcon className="h-5 w-5" />
            </button>
            <h1 className="max-w-[50vw] truncate text-sm font-medium text-slate-200">
              {conversation.title}
            </h1>
          </div>

          <div className="flex items-center gap-3">
            <span className="hidden items-center gap-1.5 text-[11px] text-slate-400 sm:flex">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-pulse-ring rounded-full bg-emerald-400" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
              </span>
              Connected
            </span>
          </div>
        </div>
      </header>

      {/* Messages */}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="scrollbar-slim flex-1 overflow-y-auto"
      >
        <div className="mx-auto max-w-3xl px-4 py-6">
          {empty ? (
            <EmptyState onPick={(q) => sendMessage(q)} disabled={streaming} />
          ) : (
            <div className="space-y-6">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Scroll-to-bottom */}
      {!atBottom && !empty && (
        <button
          onClick={scrollToBottom}
          aria-label="Scroll to bottom"
          className="absolute bottom-28 left-1/2 z-10 -translate-x-1/2 rounded-full border border-white/10 bg-ink-800/90 p-2 text-slate-300 shadow-lg backdrop-blur transition-all hover:bg-ink-700 animate-fade-in"
        >
          <ArrowDownIcon className="h-4 w-4" />
        </button>
      )}

      {/* Composer */}
      <div className="border-t border-white/8 bg-ink-950/70 backdrop-blur-xl">
        <div className="mx-auto max-w-3xl px-4 py-4">
          <div className="flex items-end gap-2 rounded-2xl border border-white/10 bg-ink-800/80 p-2 shadow-lg transition-colors focus-within:border-brand-500/50 focus-within:ring-1 focus-within:ring-brand-500/30">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                autoGrow();
              }}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your documents…"
              rows={1}
              disabled={streaming}
              className="max-h-[200px] flex-1 resize-none bg-transparent px-2 py-2 text-[0.95rem] text-slate-100 placeholder-slate-500 focus:outline-none disabled:opacity-60"
            />
            {streaming ? (
              <button
                onClick={stop}
                aria-label="Stop generating"
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/10 text-slate-200 transition-all hover:bg-white/20"
              >
                <StopIcon className="h-3.5 w-3.5" />
              </button>
            ) : (
              <button
                onClick={() => sendMessage()}
                disabled={!input.trim()}
                aria-label="Send"
                className="shine flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-violet-600 text-white shadow-md shadow-brand-600/30 transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
              >
                <SendIcon className="h-4 w-4" />
              </button>
            )}
          </div>
          <p className="mt-2 px-1 text-center text-[11px] text-slate-600">
            {streaming
              ? "Generating… press Stop to cancel"
              : "Enter to send · Shift+Enter for a new line"}
          </p>
        </div>
      </div>
    </div>
  );
}

function EmptyState({
  onPick,
  disabled,
}: {
  onPick: (q: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex min-h-[62vh] flex-col items-center justify-center text-center">
      <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-500 to-violet-600 text-white shadow-xl shadow-brand-600/30">
        <SparkIcon className="h-7 w-7" />
      </div>
      <h2 className="text-2xl font-semibold text-gradient-animated">
        Ask your documents anything
      </h2>
      <p className="mt-2 max-w-md text-sm text-slate-400">
        Answers are grounded in your knowledge base with hybrid retrieval,
        reranking, and inline source citations.
      </p>

      <div className="mt-8 grid w-full max-w-xl gap-2.5 sm:grid-cols-2">
        {EXAMPLES.map((q, i) => (
          <button
            key={q}
            onClick={() => onPick(q)}
            disabled={disabled}
            style={{ animationDelay: `${i * 60}ms` }}
            onMouseMove={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
              e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
            }}
            className="spotlight group relative animate-fade-in-up overflow-hidden rounded-xl border border-white/8 bg-white/[0.03] p-3.5 text-left text-sm text-slate-300 transition-all hover:-translate-y-0.5 hover:border-brand-500/40 hover:bg-white/[0.06] disabled:opacity-50"
          >
            <span className="relative line-clamp-2">{q}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
