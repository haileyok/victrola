import { useEffect, useState, useRef, useCallback } from "react";
import { useParams } from "react-router-dom";
import { api } from "@/lib/api";
import type { Message } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";
import { ToolActivity } from "./ToolActivity";
import { StatusBar } from "./StatusBar";
import { TokenStats } from "./TokenStats";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ChatItem {
  type: "message" | "tool";
  key: string;
  role?: string;
  content?: string;
  toolName?: string;
  code?: string;
  result?: unknown;
  success?: boolean;
  done?: boolean;
}

function parseMessage(raw: Message): { role: string; content: string } {
  try {
    const parsed = JSON.parse(raw.content);
    let content = parsed.content ?? raw.content;
    if (Array.isArray(content)) {
      content = content
        .filter((b: { type: string }) => b.type === "text")
        .map((b: { text: string }) => b.text)
        .join("\n");
    }
    return { role: parsed.role ?? raw.sender, content: String(content) };
  } catch {
    return { role: raw.sender, content: raw.content };
  }
}

export function ChatView() {
  const { id } = useParams<{ id: string }>();
  const [items, setItems] = useState<ChatItem[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [thinking, setThinking] = useState(false);
  const [toolName, setToolName] = useState<string | null>(null);
  const [tokenStats, setTokenStats] = useState({
    ctx: 0,
    tps: null as number | null,
    input: 0,
    output: 0,
    calls: 0,
  });
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, []);

  const loadHistory = useCallback(async () => {
    if (!id) return;
    try {
      const data = await api.listMessages(id, 100);
      const chatItems: ChatItem[] = data.messages.map((m) => {
        const { role, content } = parseMessage(m);
        return { type: "message" as const, key: `msg-${m.id}`, role, content };
      });
      setItems(chatItems);
    } catch (e) {
      console.error("Failed to load messages:", e);
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    scrollToBottom();
  }, [items, scrollToBottom]);

  const handleSend = async () => {
    if (!id || !input.trim()) return;
    const userText = input.trim();
    setInput("");
    setThinking(true);

    // add user message to UI immediately
    setItems((prev) => [
      ...prev,
      { type: "message", key: `user-${Date.now()}`, role: "user", content: userText },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await api.chat(
        id,
        userText,
        undefined,
        (event, data) => {
          switch (event) {
            case "llm_start":
              setThinking(true);
              setToolName(null);
              break;
            case "llm_done": {
              setThinking(false);
              setToolName(null);
              const usage = data.usage as Record<string, number> | undefined;
              if (usage) {
                setTokenStats((prev) => ({
                  ctx: prev.ctx,
                  tps: prev.tps,
                  input: (prev.input + (usage.input_tokens ?? 0)),
                  output: (prev.output + (usage.output_tokens ?? 0)),
                  calls: prev.calls + 1,
                }));
              }
              break;
            }
            case "tool_start": {
              const name = (data.tool as string) || "unknown";
              const code = (data.code as string) || "";
              setToolName(name);
              setThinking(false);
              setItems((prev) => [
                ...prev,
                {
                  type: "tool",
                  key: `tool-${Date.now()}`,
                  toolName: name,
                  code,
                  done: false,
                },
              ]);
              break;
            }
            case "tool_done": {
              setToolName(null);
              const result = data.result;
              const success = data.success !== false;
              setItems((prev) => {
                const next = [...prev];
                // update last tool activity
                for (let i = next.length - 1; i >= 0; i--) {
                  if (next[i].type === "tool" && next[i].done === false) {
                    next[i] = { ...next[i], result, success, done: true };
                    // auto-expand on error
                    break;
                  }
                }
                return next;
              });
              break;
            }
            case "response": {
              const text = (data.text as string) || "";
              setThinking(false);
              setToolName(null);
              if (text) {
                setItems((prev) => [
                  ...prev,
                  { type: "message", key: `resp-${Date.now()}`, role: "assistant", content: text },
                ]);
              }
              break;
            }
            case "error": {
              const msg = (data.message as string) || "Unknown error";
              setThinking(false);
              setToolName(null);
              setItems((prev) => [
                ...prev,
                { type: "message", key: `err-${Date.now()}`, role: "system", content: msg },
              ]);
              break;
            }
            case "done":
              break;
          }
        },
        controller.signal,
      );
    } catch (e) {
      if (e instanceof Error && e.name !== "AbortError") {
        setItems((prev) => [
          ...prev,
          { type: "message", key: `err-${Date.now()}`, role: "system", content: `Chat error: ${e.message}` },
        ]);
      }
    } finally {
      setThinking(false);
      setToolName(null);
      abortRef.current = null;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isResponding = thinking || toolName !== null;

  return (
    <div className="flex h-full flex-col">
      {/* Chat log */}
      <ScrollArea className="flex-1">
        <div ref={scrollRef} className="flex flex-col gap-1 py-2">
          {loading ? (
            <div className="px-4 py-8 text-center text-muted-foreground">Loading…</div>
          ) : items.length === 0 ? (
            <div className="px-4 py-8 text-center text-muted-foreground">
              Send a message to start chatting.
            </div>
          ) : (
            items.map((item) =>
              item.type === "message" ? (
                <MessageBubble
                  key={item.key}
                  role={item.role || "user"}
                  content={item.content || ""}
                />
              ) : (
                <ToolActivity
                  key={item.key}
                  toolName={item.toolName || "unknown"}
                  code={item.code}
                  result={item.result}
                  success={item.success}
                  done={item.done || false}
                />
              ),
            )
          )}
        </div>
      </ScrollArea>

      {/* Status bar */}
      <StatusBar thinking={thinking} toolName={toolName} />

      {/* Token stats */}
      <TokenStats
        ctx={tokenStats.ctx}
        tps={tokenStats.tps}
        input={tokenStats.input}
        output={tokenStats.output}
        calls={tokenStats.calls}
      />

      {/* Input */}
      <div className="flex items-center gap-2 border-t border-border px-4 py-3">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message…"
          disabled={isResponding}
        />
        <Button onClick={handleSend} disabled={isResponding || !input.trim()}>
          Send
        </Button>
      </div>
    </div>
  );
}
