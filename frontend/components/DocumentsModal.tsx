"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ACCEPTED,
  ACCEPTED_LABEL,
  DocInfo,
  deleteDocument,
  isSupported,
  listDocuments,
  uploadDocuments,
} from "@/lib/documents";
import {
  CloseIcon,
  DocIcon,
  SpinnerIcon,
  TrashIcon,
  UploadCloudIcon,
} from "./icons";

interface Props {
  open: boolean;
  onClose: () => void;
  pendingFiles?: File[] | null;
  onConsumePending?: () => void;
}

type Note = { kind: "ok" | "err"; text: string } | null;

export function DocumentsModal({
  open,
  onClose,
  pendingFiles,
  onConsumePending,
}: Props) {
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [note, setNote] = useState<Note>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setDocs(await listDocuments());
    setLoading(false);
  }, []);

  const handleFiles = useCallback(
    async (files: File[]) => {
      const supported = files.filter((f) => isSupported(f.name));
      const skipped = files.length - supported.length;
      if (supported.length === 0) {
        setNote({
          kind: "err",
          text: `Unsupported file type. Try ${ACCEPTED_LABEL}.`,
        });
        return;
      }
      setUploading(true);
      setNote(null);
      try {
        const res = await uploadDocuments(supported);
        await refresh();
        const addedCount = res.added.length;
        const errCount = res.errors.length + skipped;
        const parts: string[] = [];
        if (addedCount) parts.push(`Added ${addedCount} document${addedCount > 1 ? "s" : ""}`);
        if (errCount) parts.push(`${errCount} skipped`);
        setNote({
          kind: res.errors.length ? "err" : "ok",
          text: parts.join(" · ") || "Nothing to add",
        });
      } catch {
        setNote({ kind: "err", text: "Upload failed. Is the server running?" });
      } finally {
        setUploading(false);
      }
    },
    [refresh],
  );

  // Initial load + auto-upload any files dropped onto the app.
  useEffect(() => {
    if (!open) return;
    refresh();
    if (pendingFiles && pendingFiles.length) {
      handleFiles(pendingFiles);
      onConsumePending?.();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Escape to close.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !uploading) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, uploading, onClose]);

  if (!open) return null;

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length) handleFiles(files);
  };

  const remove = async (filename: string) => {
    setDocs((prev) => prev.filter((d) => d.filename !== filename));
    await deleteDocument(filename);
    refresh();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
    >
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={() => !uploading && onClose()}
      />

      <div className="relative w-full max-w-lg animate-fade-in-up rounded-2xl border border-white/10 bg-ink-850/95 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/8 px-5 py-4">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Knowledge base
            </h2>
            <p className="text-xs text-slate-500">
              Add documents to chat with them
            </p>
          </div>
          <button
            onClick={onClose}
            disabled={uploading}
            aria-label="Close"
            className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-white/8 hover:text-slate-200 disabled:opacity-40"
          >
            <CloseIcon className="h-5 w-5" />
          </button>
        </div>

        <div className="p-5">
          {/* Dropzone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => !uploading && inputRef.current?.click()}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-8 text-center transition-colors ${
              dragOver
                ? "border-brand-500/70 bg-brand-500/10"
                : "border-white/12 bg-white/[0.02] hover:border-brand-500/40 hover:bg-white/[0.04]"
            }`}
          >
            {uploading ? (
              <>
                <SpinnerIcon className="h-8 w-8 animate-spin text-brand-400" />
                <p className="mt-3 text-sm font-medium text-slate-200">
                  Processing… embedding your documents
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  This can take a moment on first run
                </p>
              </>
            ) : (
              <>
                <UploadCloudIcon className="h-8 w-8 text-brand-400" />
                <p className="mt-3 text-sm font-medium text-slate-200">
                  Drag &amp; drop files here
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  or{" "}
                  <span className="text-brand-400 underline">browse your files</span>
                </p>
                <p className="mt-1.5 text-[11px] text-slate-600">
                  {ACCEPTED_LABEL}
                </p>
              </>
            )}
            <input
              ref={inputRef}
              type="file"
              multiple
              accept={ACCEPTED}
              className="hidden"
              onChange={(e) => {
                const files = Array.from(e.target.files ?? []);
                e.target.value = "";
                if (files.length) handleFiles(files);
              }}
            />
          </div>

          {note && (
            <p
              className={`mt-3 rounded-lg px-3 py-2 text-xs ${
                note.kind === "ok"
                  ? "bg-emerald-500/10 text-emerald-300"
                  : "bg-amber-500/10 text-amber-300"
              }`}
            >
              {note.text}
            </p>
          )}

          {/* Document list */}
          <div className="mt-5">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Your documents
              </h3>
              {!loading && (
                <span className="text-xs text-slate-600">{docs.length}</span>
              )}
            </div>

            <div className="scrollbar-slim max-h-56 overflow-y-auto">
              {loading ? (
                <p className="py-6 text-center text-xs text-slate-600">Loading…</p>
              ) : docs.length === 0 ? (
                <p className="py-6 text-center text-xs text-slate-600">
                  No documents yet — add some above.
                </p>
              ) : (
                <ul className="space-y-1">
                  {docs.map((d) => (
                    <li
                      key={d.filename}
                      className="group flex items-center gap-2.5 rounded-lg border border-white/6 bg-white/[0.02] px-3 py-2"
                    >
                      <DocIcon className="h-4 w-4 shrink-0 text-slate-500" />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm text-slate-200">
                          {d.filename}
                        </p>
                        <p className="text-[11px] text-slate-500">
                          {d.chunks} chunk{d.chunks === 1 ? "" : "s"}
                        </p>
                      </div>
                      <button
                        onClick={() => remove(d.filename)}
                        aria-label={`Remove ${d.filename}`}
                        className="rounded p-1 text-slate-500 opacity-0 transition-all hover:bg-rose-500/20 hover:text-rose-300 group-hover:opacity-100"
                      >
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
