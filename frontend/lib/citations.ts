import type { Citation } from "@/types";

export async function resolveCitations(
  chunkIds: string[],
): Promise<Citation[]> {
  if (!chunkIds.length) return [];
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  const ids = chunkIds.join(",");
  try {
    const res = await fetch(`${apiUrl}/chunks?ids=${encodeURIComponent(ids)}`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}
