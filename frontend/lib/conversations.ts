import type { Message } from "@/types";

/** A single chat thread. `id` doubles as the backend Redis session id. */
export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = "rag.conversations.v1";
const ACTIVE_KEY = "rag.active-conversation.v1";

export function newId(): string {
  return `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createConversation(): Conversation {
  const now = Date.now();
  return {
    id: newId(),
    title: "New chat",
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

/** Derive a concise thread title from the first user message. */
export function deriveTitle(text: string): string {
  const clean = text.trim().replace(/\s+/g, " ");
  if (clean.length <= 48) return clean;
  return `${clean.slice(0, 48).trimEnd()}…`;
}

/** Strip transient streaming flags before persisting. */
function sanitize(messages: Message[]): Message[] {
  return messages.map((m) => ({
    ...m,
    isStreaming: false,
    status: m.status === "error" ? "error" : "done",
  }));
}

export function loadConversations(): Conversation[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Conversation[];
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    return [];
  }
}

export function saveConversations(conversations: Conversation[]): void {
  if (typeof window === "undefined") return;
  try {
    const serializable = conversations.map((c) => ({
      ...c,
      messages: sanitize(c.messages),
    }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
  } catch {
    // storage full or unavailable — non-fatal
  }
}

export function loadActiveId(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACTIVE_KEY);
}

export function saveActiveId(id: string): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(ACTIVE_KEY, id);
  } catch {
    // non-fatal
  }
}
