import { useEffect, useState, useCallback } from "react";
import { RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type { SystemPrompt as SystemPromptType } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

export function SystemPromptView() {
  const [prompt, setPrompt] = useState<SystemPromptType | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      setPrompt(await api.getSystemPrompt());
    } catch (e) {
      console.error("Failed to load system prompt:", e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleRefresh = () => {
    setRefreshing(true);
    load();
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border px-6 py-3">
        <div>
          <h2 className="text-lg font-semibold">System Prompt</h2>
          {prompt && (
            <div className="text-xs text-muted-foreground">
              {prompt.char_count.toLocaleString()} chars · ~{prompt.token_estimate.toLocaleString()} tokens
            </div>
          )}
        </div>
        <Button size="sm" variant="outline" onClick={handleRefresh} disabled={refreshing}>
          <RefreshCw className={`mr-1 h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-6">
          {loading ? (
            <div className="text-muted-foreground">Loading…</div>
          ) : prompt ? (
            <pre className="whitespace-pre-wrap break-words font-mono text-sm">
              {prompt.text}
            </pre>
          ) : (
            <div className="text-muted-foreground">Failed to load system prompt.</div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
