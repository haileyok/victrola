import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Check, X, Trash2 } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { ToolSummary } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

export function ToolsView() {
  const navigate = useNavigate();
  const [tools, setTools] = useState<ToolSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      setTools(await api.listTools());
    } catch (e) {
      console.error("Failed to load tools:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleApprove = async (e: React.MouseEvent, name: string) => {
    e.stopPropagation();
    try {
      await api.approveTool(name);
      await refresh();
    } catch (e) {
      if (e instanceof ApiError && e.status === 400) {
        navigate(`/tools/${name}`);
      } else {
        console.error("Failed to approve:", e);
      }
    }
  };

  const handleRevoke = async (e: React.MouseEvent, name: string) => {
    e.stopPropagation();
    try {
      await api.revokeTool(name);
      await refresh();
    } catch (e) {
      console.error("Failed to revoke:", e);
    }
  };

  const handleDelete = async (e: React.MouseEvent, name: string) => {
    e.stopPropagation();
    if (!confirm(`Delete tool "${name}"?`)) return;
    try {
      await api.deleteTool(name);
      await refresh();
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="px-6 py-3">
        <h2 className="text-lg font-semibold">Custom Tools</h2>
      </div>
      <ScrollArea className="flex-1">
        <div className="px-3 pb-4">
          {loading ? (
            <div className="px-3 py-8 text-center text-muted-foreground">Loading…</div>
          ) : tools.length === 0 ? (
            <div className="px-3 py-8 text-center text-muted-foreground">
              No custom tools. Create one via the agent.
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              {tools.map((t) => (
                <div
                  key={t.name}
                  onClick={() => navigate(`/tools/${t.name}`)}
                  className="group flex cursor-pointer items-center gap-3 rounded-md px-3 py-2.5 hover:bg-accent"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">{t.name}</span>
                      {t.approved ? (
                        <Badge className="bg-green-600 text-white">approved</Badge>
                      ) : (
                        <Badge variant="secondary" className="bg-yellow-600 text-white">pending</Badge>
                      )}
                      {t.requires_net && <Badge variant="outline">net</Badge>}
                    </div>
                    <div className="truncate text-xs text-muted-foreground">{t.description}</div>
                  </div>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
                    {!t.approved && (
                      <Button variant="ghost" size="icon" className="h-7 w-7" onClick={(e) => handleApprove(e, t.name)}>
                        <Check className="h-3.5 w-3.5 text-green-500" />
                      </Button>
                    )}
                    {t.approved && (
                      <Button variant="ghost" size="icon" className="h-7 w-7" onClick={(e) => handleRevoke(e, t.name)}>
                        <X className="h-3.5 w-3.5 text-yellow-500" />
                      </Button>
                    )}
                    <Button variant="ghost" size="icon" className="h-7 w-7" onClick={(e) => handleDelete(e, t.name)}>
                      <Trash2 className="h-3.5 w-3.5 text-red-500" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
