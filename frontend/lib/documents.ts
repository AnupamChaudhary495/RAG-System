const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface DocInfo {
  filename: string;
  chunks: number;
}

export interface UploadResult {
  added: { filename: string; chunks: number }[];
  errors: { filename: string; error: string }[];
}

/** File extensions the backend can ingest. */
export const ACCEPTED = ".pdf,.md,.markdown,.txt";
const ACCEPTED_EXTS = ["pdf", "md", "markdown", "txt"];

export function isSupported(name: string): boolean {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return ACCEPTED_EXTS.includes(ext);
}

export async function listDocuments(): Promise<DocInfo[]> {
  try {
    const r = await fetch(`${API_URL}/documents`);
    if (!r.ok) return [];
    return (await r.json()) as DocInfo[];
  } catch {
    return [];
  }
}

export async function uploadDocuments(files: File[]): Promise<UploadResult> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const r = await fetch(`${API_URL}/documents`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) {
    throw new Error(`Upload failed (${r.status})`);
  }
  return (await r.json()) as UploadResult;
}

export async function deleteDocument(filename: string): Promise<void> {
  await fetch(`${API_URL}/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  });
}
