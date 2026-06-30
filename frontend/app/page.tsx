"use client";

import { useEffect, useState } from "react";
import { ChatInterface } from "@/components/ChatInterface";

function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return "ssr";
  const key = "rag_session_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = `sess-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    localStorage.setItem(key, id);
  }
  return id;
}

export default function Page() {
  const [sessionId, setSessionId] = useState<string | null>(null);

  useEffect(() => {
    setSessionId(getOrCreateSessionId());
  }, []);

  if (!sessionId) return null;

  return <ChatInterface sessionId={sessionId} />;
}
