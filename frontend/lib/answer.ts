/**
 * Progressive extraction of the `answer` field from a streaming JSON payload.
 *
 * The generator returns JSON-mode output like:
 *   { "answer": "…", "source_chunk_ids": [...], "confidence": 0.9 }
 * streamed token-by-token. Rendering the raw buffer would show the JSON
 * scaffolding, so this walks the partial buffer and returns just the current
 * (unescaped) value of the `answer` string — even while it is mid-stream.
 *
 * Falls back to returning the raw text unchanged when the buffer is not JSON
 * (e.g. a generator that emits plain prose), so the UI degrades gracefully.
 */
export function extractAnswer(raw: string): string {
  if (!raw) return "";

  // Plain-text fallback: not a JSON object → show as-is.
  if (!raw.trimStart().startsWith("{")) return raw;

  const keyIdx = raw.indexOf('"answer"');
  if (keyIdx === -1) return ""; // key not streamed yet

  // Advance past the key, the colon, and any whitespace to the opening quote.
  let i = keyIdx + '"answer"'.length;
  while (i < raw.length && raw[i] !== ":") i++;
  i++; // skip ':'
  while (i < raw.length && /\s/.test(raw[i]!)) i++;
  if (raw[i] !== '"') return ""; // value hasn't started
  i++; // enter the string body

  let out = "";
  while (i < raw.length) {
    const ch = raw[i]!;

    if (ch === "\\") {
      const next = raw[i + 1];
      if (next === undefined) break; // incomplete escape at buffer edge

      // Only collapse the escapes that are unambiguous JSON. Crucially we do
      // NOT interpret \n \t \r \b \f as control chars: LLMs routinely emit
      // single-backslash LaTeX (\text, \frac, \theta, \tau, \nabla) inside the
      // JSON string, and mangling those into tabs/newlines breaks KaTeX. The
      // backslash is preserved so the math survives.
      if (next === '"' || next === "\\" || next === "/") {
        out += next;
        i += 2;
        continue;
      }
      if (next === "u") {
        const hex = raw.slice(i + 2, i + 6);
        if (/^[0-9a-fA-F]{4}$/.test(hex)) {
          out += String.fromCharCode(parseInt(hex, 16));
          i += 6;
          continue;
        }
        // Not a real \uXXXX escape (e.g. \underline) → keep the backslash.
      }
      // Preserve the backslash; the following char is handled next iteration.
      out += "\\";
      i += 1;
      continue;
    }

    if (ch === '"') break; // closing quote → answer complete
    out += ch;
    i++;
  }

  return out;
}

/** Confidence tier used to colour-code the confidence badge. */
export type ConfidenceTier = "high" | "medium" | "low";

export function confidenceTier(score: number): ConfidenceTier {
  if (score >= 0.7) return "high";
  if (score >= 0.4) return "medium";
  return "low";
}
