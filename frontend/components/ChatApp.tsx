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
import { DocumentsModal } from "./DocumentsModal";
import { Sidebar, type ConversationSummary } from "./Sidebar";
import { UploadCloudIcon } from "./icons";

export function ChatApp() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [ready, setReady] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[] | null>(null);
  const [dragging, setDragging] = useState(false);
  const dragCount = useRef(0);
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

  // Global drag-and-drop: drop files anywhere to add them to the knowledge base.
  useEffect(() => {
    const hasFiles = (e: DragEvent) =>
      Array.from(e.dataTransfer?.types ?? []).includes("Files");

    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragCount.current += 1;
      setDragging(true);
    };
    const onOver = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const onLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      dragCount.current = Math.max(0, dragCount.current - 1);
      if (dragCount.current === 0) setDragging(false);
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragCount.current = 0;
      setDragging(false);
      const files = Array.from(e.dataTransfer?.files ?? []);
      if (files.length) {
        setPendingFiles(files);
        setDocsOpen(true);
      }
    };

    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragover", onOver);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, []);

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
        onOpenDocs={() => setDocsOpen(true)}
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

      {/* Global drag-and-drop overlay */}
      {dragging && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/80 p-6 backdrop-blur-sm animate-fade-in">
          <div className="pointer-events-none flex flex-col items-center rounded-3xl border-2 border-dashed border-brand-500/60 bg-ink-900/70 px-12 py-14 text-center">
            <UploadCloudIcon className="h-12 w-12 text-brand-400" />
            <p className="mt-4 text-lg font-semibold text-slate-100">
              Drop files to add to your knowledge base
            </p>
            <p className="mt-1 text-sm text-slate-400">
              PDF, Markdown, or text — we&apos;ll handle the rest
            </p>
          </div>
        </div>
      )}

      <DocumentsModal
        open={docsOpen}
        onClose={() => setDocsOpen(false)}
        pendingFiles={pendingFiles}
        onConsumePending={() => setPendingFiles(null)}
      />
    </>
  );
}
