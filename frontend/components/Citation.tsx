"use client";

import { useState } from "react";
import type { Citation } from "@/types";
import { ChevronIcon, DocIcon } from "./icons";

interface Props {
  citations: Citation[];
}

export function CitationList({ citations }: Props) {
  const [open, setOpen] = useState(false);

  if (!citations.length) return null;

  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="group inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs font-medium text-slate-400 transition-colors hover:bg-white/5 hover:text-slate-200"
      >
        <DocIcon className="h-3.5 w-3.5" />
        <span>
          {citations.length} source{citations.length > 1 ? "s" : ""}
        </span>
        <ChevronIcon
          className={`h-3.5 w-3.5 transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>

      {open && (
        <ul className="mt-2 grid gap-2 animate-fade-in sm:grid-cols-2">
          {citations.map((c, idx) => (
            <li
              key={c.chunk_id}
              className="flex items-start gap-2.5 rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2.5 transition-colors hover:border-white/15 hover:bg-white/[0.06]"
            >
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-brand-500/15 text-[10px] font-semibold text-brand-400">
                {idx + 1}
              </span>
              <div className="min-w-0">
                <p className="truncate font-mono text-xs text-slate-200">
                  {c.source_filename}
                </p>
                <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-slate-500">
                  {c.page_number != null && <span>page {c.page_number}</span>}
                  {c.section_heading && (
                    <span className="truncate italic">{c.section_heading}</span>
                  )}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
