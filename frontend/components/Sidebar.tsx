"use client";

import { memo, useEffect, useRef, useState } from "react";
import {
  ChatIcon,
  EditIcon,
  PlusIcon,
  SidebarIcon,
  SparkIcon,
  TrashIcon,
} from "./icons";
import { InstallButton } from "./InstallButton";

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: number;
}

interface Props {
  conversations: ConversationSummary[];
  activeId: string;
  open: boolean;
  disabled: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onClose: () => void;
  onToggle: () => void;
}

const DAY = 86_400_000;

function bucketOf(ts: number): string {
  const now = new Date();
  const startOfToday = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate(),
  ).getTime();
  if (ts >= startOfToday) return "Today";
  if (ts >= startOfToday - DAY) return "Yesterday";
  if (ts >= startOfToday - 7 * DAY) return "Previous 7 days";
  return "Older";
}

const BUCKET_ORDER = ["Today", "Yesterday", "Previous 7 days", "Older"];

function SidebarInner({
  conversations,
  activeId,
  open,
  disabled,
  onSelect,
  onNew,
  onDelete,
  onRename,
  onClose,
  onToggle,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const editRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId) editRef.current?.select();
  }, [editingId]);

  const groups = BUCKET_ORDER.map((label) => ({
    label,
    items: conversations.filter((c) => bucketOf(c.updatedAt) === label),
  })).filter((g) => g.items.length > 0);

  const startRename = (id: string, current: string) => {
    setEditingId(id);
    setDraft(current);
  };
  const commitRename = () => {
    if (editingId) {
      const t = draft.trim();
      if (t) onRename(editingId, t);
    }
    setEditingId(null);
  };

  return (
    <>
      {open && (
        <div
          onClick={onClose}
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm md:hidden"
          aria-hidden
        />
      )}

      <aside
        className={`z-40 shrink-0 overflow-hidden border-r border-white/8 bg-ink-900/85 backdrop-blur-xl transition-all duration-300 ease-out fixed inset-y-0 left-0 md:static ${
          open
            ? "w-[280px] translate-x-0"
            : "w-[280px] -translate-x-full md:w-0 md:translate-x-0"
        }`}
      >
        <div className="flex h-full w-[280px] flex-col">
          {/* Brand + collapse */}
          <div className="flex items-center justify-between px-3 py-3">
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-violet-600 text-white shadow-md shadow-brand-600/25">
                <SparkIcon className="h-5 w-5" />
              </div>
              <span className="text-sm font-semibold text-slate-100">
                RAG Assistant
              </span>
            </div>
            <button
              onClick={onToggle}
              aria-label="Collapse sidebar"
              className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-white/8 hover:text-slate-200"
            >
              <SidebarIcon className="h-5 w-5" />
            </button>
          </div>

          {/* New chat */}
          <div className="px-3 pb-2">
            <button
              onClick={onNew}
              disabled={disabled}
              className="flex w-full items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2.5 text-sm font-medium text-slate-200 transition-all hover:border-brand-500/40 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <PlusIcon className="h-4 w-4" />
              New chat
            </button>
          </div>

          {/* Conversation list */}
          <nav className="scrollbar-slim flex-1 overflow-y-auto px-2 pb-4">
            {conversations.length === 0 ? (
              <p className="px-3 py-6 text-center text-xs text-slate-600">
                No conversations yet
              </p>
            ) : (
              groups.map((group) => (
                <div key={group.label} className="mb-2">
                  <p className="px-3 py-1.5 text-[11px] font-medium uppercase tracking-wide text-slate-600">
                    {group.label}
                  </p>
                  <ul className="space-y-0.5">
                    {group.items.map((c) => {
                      const active = c.id === activeId;
                      const editing = editingId === c.id;
                      return (
                        <li key={c.id}>
                          <div
                            className={`group relative flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm transition-colors ${
                              active
                                ? "bg-brand-500/15 text-slate-100"
                                : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
                            }`}
                          >
                            <ChatIcon className="h-4 w-4 shrink-0 opacity-70" />

                            {editing ? (
                              <input
                                ref={editRef}
                                value={draft}
                                onChange={(e) => setDraft(e.target.value)}
                                onBlur={commitRename}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter") commitRename();
                                  if (e.key === "Escape") setEditingId(null);
                                }}
                                className="min-w-0 flex-1 rounded border border-brand-500/40 bg-ink-800 px-1.5 py-0.5 text-sm text-slate-100 focus:outline-none"
                              />
                            ) : (
                              <button
                                onClick={() => !disabled && onSelect(c.id)}
                                className="min-w-0 flex-1 truncate text-left"
                                title={c.title}
                              >
                                {c.title}
                              </button>
                            )}

                            {!editing && (
                              <div
                                className={`flex shrink-0 items-center gap-0.5 transition-opacity ${
                                  active
                                    ? "opacity-100"
                                    : "opacity-0 group-hover:opacity-100"
                                }`}
                              >
                                <button
                                  onClick={() => startRename(c.id, c.title)}
                                  aria-label="Rename"
                                  className="rounded p-1 text-slate-400 transition-colors hover:bg-white/10 hover:text-slate-200"
                                >
                                  <EditIcon className="h-3.5 w-3.5" />
                                </button>
                                <button
                                  onClick={() => onDelete(c.id)}
                                  aria-label="Delete"
                                  className="rounded p-1 text-slate-400 transition-colors hover:bg-rose-500/20 hover:text-rose-300"
                                >
                                  <TrashIcon className="h-3.5 w-3.5" />
                                </button>
                              </div>
                            )}
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))
            )}
          </nav>

          <div className="space-y-2.5 border-t border-white/8 px-3 py-3">
            <InstallButton />
            <p className="px-1 text-[11px] text-slate-600">
              Hybrid retrieval · BGE-M3 · llama3.2
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}

export const Sidebar = memo(SidebarInner);
