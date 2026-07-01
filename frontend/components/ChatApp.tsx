"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Message } from "@/types";
import {
  Conversation,
  createConversation,
  deriveTitle,
  loadActiveId,
  loadConversations,
  saveActiveId,
  saveConversations,
} from "@/lib/conversations";
import { clearSession } from "@/lib/stream";
import { Aurora } from "./Aurora";
import { ChatInterface } from "./ChatInterface";
import { Sidebar, type ConversationSummary } from "./Sidebar";

export function ChatApp() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [ready, setReady] = useState(false);
  const saveTimer = useRef<ReturnType<typeof setTimeout>>();

  // Hydrate from localStorage on mount.
  useEffect(() => {
    const loaded = loadConversations();
    if (loaded.length > 0) {
      const savedActive = loadActiveId();
      const active =
        savedActive && loaded.some((c) => c.id === savedActive)
          ? savedActive
          : loaded[0]!.id;
      setConversations(loaded);
      setActiveId(active);
    } else {
      const fresh = createConversation();
      setConversations([fresh]);
      setActiveId(fresh.id);
    }
    setSidebarOpen(window.innerWidth >= 768);
    setReady(true);
  }, []);

  // Debounced persistence.
  useEffect(() => {
    if (!ready) return;
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => saveConversations(conversations), 350);
    return () => clearTimeout(saveTimer.current);
  }, [conversations, ready]);

  useEffect(() => {
    if (ready && activeId) saveActiveId(activeId);
  }, [activeId, ready]);

  const isMobile = () =>
    typeof window !== "undefined" && window.innerWidth < 768;

  const handleNew = useCallback(() => {
    if (streaming) return;
    // Reuse an existing empty "New chat" instead of stacking blanks.
    const blank = conversations.find(
      (c) => c.messages.length === 0 && c.title === "New chat",
    );
    if (blank) {
      setActiveId(blank.id);
    } else {
      const fresh = createConversation();
      setConversations((prev) => [fresh, ...prev]);
      setActiveId(fresh.id);
    }
    if (isMobile()) setSidebarOpen(false);
  }, [streaming, conversations]);

  const handleSelect = useCallback(
    (id: string) => {
      if (streaming) return;
      setActiveId(id);
      if (isMobile()) setSidebarOpen(false);
    },
    [streaming],
  );

  const handleDelete = useCallback(
    (id: string) => {
      clearSession(id);
      const next = conversations.filter((c) => c.id !== id);
      if (next.length === 0) {
        const fresh = createConversation();
        setConversations([fresh]);
        setActiveId(fresh.id);
      } else {
        setConversations(next);
        if (id === activeId) setActiveId(next[0]!.id);
      }
    },
    [conversations, activeId],
  );

  const handleRename = useCallback((id: string, title: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title } : c)),
    );
  }, []);

  // Called by ChatInterface to sync a conversation's messages.
  const handlePersist = useCallback((id: string, messages: Message[]) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== id) return c;
        const firstUser = messages.find((m) => m.role === "user");
        const title =
          c.title === "New chat" && firstUser
            ? deriveTitle(firstUser.content)
            : c.title;
        return { ...c, messages, title, updatedAt: Date.now() };
      }),
    );
  }, []);

  // Keyboard shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "b") {
        e.preventDefault();
        setSidebarOpen((v) => !v);
      } else if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        handleNew();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleNew]);

  const summaries: ConversationSummary[] = useMemo(
    () =>
      conversations.map((c) => ({
        id: c.id,
        title: c.title,
        updatedAt: c.updatedAt,
      })),
    [conversations],
  );

  const active = conversations.find((c) => c.id === activeId);

  if (!ready || !active) {
    return <div className="h-screen bg-ink-950" />;
  }

  return (
    <>
      <Aurora />
      <div className="relative z-10 flex h-screen overflow-hidden">
        <Sidebar
        conversations={summaries}
        activeId={activeId}
        open={sidebarOpen}
        disabled={streaming}
        onSelect={handleSelect}
        onNew={handleNew}
        onDelete={handleDelete}
        onRename={handleRename}
        onClose={() => setSidebarOpen(false)}
        onToggle={() => setSidebarOpen((v) => !v)}
      />

        <ChatInterface
          key={active.id}
          conversation={active}
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen((v) => !v)}
          onPersist={handlePersist}
          onStreamingChange={setStreaming}
        />
      </div>
    </>
  );
}
