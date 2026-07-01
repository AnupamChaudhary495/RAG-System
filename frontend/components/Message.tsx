import { useState } from "react";
import type { Message } from "@/types";
import { confidenceTier, extractAnswer } from "@/lib/answer";
import { CitationList } from "./Citation";
import { Markdown } from "./Markdown";
import { CheckIcon, CopyIcon, SparkIcon, UserIcon } from "./icons";

interface Props {
  message: Message;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable — ignore
    }
  };

  return (
    <button
      onClick={copy}
      aria-label={copied ? "Copied" : "Copy answer"}
      className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium text-slate-500 transition-colors hover:bg-white/5 hover:text-slate-300"
    >
      {copied ? (
        <>
          <CheckIcon className="h-3 w-3" /> Copied
        </>
      ) : (
        <>
          <CopyIcon className="h-3 w-3" /> Copy
        </>
      )}
    </button>
  );
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1 py-1" aria-label="Assistant is thinking">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 rounded-full bg-slate-400 animate-dot-bounce"
          style={{ animationDelay: `${i * 0.16}s` }}
        />
      ))}
    </div>
  );
}

const TIER_STYLES: Record<string, string> = {
  high: "bg-emerald-500/12 text-emerald-300 ring-emerald-500/25",
  medium: "bg-amber-500/12 text-amber-300 ring-amber-500/25",
  low: "bg-rose-500/12 text-rose-300 ring-rose-500/25",
};

function ConfidenceBadge({ score }: { score: number }) {
  const tier = confidenceTier(score);
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${TIER_STYLES[tier]}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {Math.round(score * 100)}% confidence
    </span>
  );
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex animate-fade-in-up justify-end gap-3">
        <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-gradient-to-br from-brand-500 to-violet-600 px-4 py-2.5 text-[0.95rem] leading-relaxed text-white shadow-lg shadow-brand-600/20">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white/8 text-slate-300 ring-1 ring-white/10">
          <UserIcon className="h-4 w-4" />
        </div>
      </div>
    );
  }

  const answer = extractAnswer(message.content);
  const thinking = message.status === "thinking" || (message.isStreaming && !answer);
  const showCaret = message.isStreaming && !!answer;
  const isError = message.status === "error";
  const showActions = !message.isStreaming && !isError && !!answer;

  return (
    <div className="flex animate-fade-in-up justify-start gap-3">
      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-brand-500 to-violet-600 text-white shadow-md shadow-brand-600/25">
        <SparkIcon className="h-4 w-4" />
      </div>

      <div className="min-w-0 max-w-[85%] flex-1">
        <div className="rounded-2xl rounded-tl-sm border border-white/8 bg-ink-800/70 px-4 py-3 backdrop-blur-sm">
          {thinking ? (
            <ThinkingDots />
          ) : (
            <div className="relative">
              <Markdown>{answer}</Markdown>
              {showCaret && (
                <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 bg-brand-400 animate-blink" />
              )}
            </div>
          )}
        </div>

        {(message.confidence_score != null ||
          message.router_decision) && (
          <div className="mt-2 flex flex-wrap items-center gap-2 px-1">
            {message.confidence_score != null && (
              <ConfidenceBadge score={message.confidence_score} />
            )}
            {message.router_decision && (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-white/5 px-2 py-0.5 text-[11px] font-medium text-slate-400 ring-1 ring-inset ring-white/8">
                {message.router_decision}
              </span>
            )}
            {message.retry_count != null && message.retry_count > 0 && (
              <span className="text-[11px] text-slate-500">
                {message.retry_count} refinement
                {message.retry_count > 1 ? "s" : ""}
              </span>
            )}
            {showActions && <CopyButton text={answer} />}
          </div>
        )}

        {showActions &&
          message.confidence_score == null &&
          !message.router_decision && (
            <div className="mt-2 flex items-center gap-2 px-1">
              <CopyButton text={answer} />
            </div>
          )}

        <CitationList citations={message.citations} />
      </div>
    </div>
  );
}
