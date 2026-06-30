"use client";

import { useState } from "react";
import type { Citation } from "@/types";

interface Props {
  citations: Citation[];
}

export function CitationList({ citations }: Props) {
  const [open, setOpen] = useState(false);

  if (!citations.length) return null;

  return (
    <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
      >
        <span>{open ? "▼" : "▶"}</span>
        <span>Sources ({citations.length})</span>
      </button>

      {open && (
        <ul className="mt-1 space-y-1 pl-4 border-l border-gray-200 dark:border-gray-700">
          {citations.map((c) => (
            <li key={c.chunk_id}>
              <span className="font-medium">{c.source_filename}</span>
              {c.page_number != null && (
                <span className="ml-1">p.{c.page_number}</span>
              )}
              {c.section_heading && (
                <span className="ml-1 italic">— {c.section_heading}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
