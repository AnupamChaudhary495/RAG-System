import type { Message } from "@/types";
import { CitationList } from "./Citation";

interface Props {
  message: Message;
}

function renderMarkdown(text: string): React.ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\n)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code
          key={i}
          className="bg-gray-100 dark:bg-gray-800 px-1 rounded text-sm font-mono"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part === "\n") return <br key={i} />;
    return <span key={i}>{part}</span>;
  });
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-blue-600 text-white"
            : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100"
        }`}
      >
        <div className="text-sm leading-relaxed whitespace-pre-wrap">
          {renderMarkdown(message.content)}
          {message.isStreaming && (
            <span className="inline-block w-2 h-4 bg-current animate-pulse ml-0.5 align-middle" />
          )}
        </div>

        {!isUser && <CitationList citations={message.citations} />}
      </div>
    </div>
  );
}
