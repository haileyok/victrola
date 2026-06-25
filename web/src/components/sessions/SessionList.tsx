import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Plus, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import type { Session, Status } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

export function SessionList() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [status, setStatus] = useState<Status | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const [s, st] = await Promise.all([api.listSessions(), api.getStatus()]);
      setSessions(s.sessions);
      setStatus(st);
    } catch (e) {
      console.error("Failed to load:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleCreate = async () => {
    try {
      const session = await api.createSession();
      navigate(`/sessions/${session.rkey}`);
    } catch (e) {
      console.error("Failed to create session:", e);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await api.deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.rkey !== id));
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Status banner */}
      {status && (
        <div className="flex flex-wrap items-center gap-4 border-b border-border px-6 py-3 text-sm">
          <span><strong>Model:</strong> {status.model}</span>
          <span><strong>Discord:</strong> {status.discord ? "on" : "off"}</span>
          <span><strong>Schedules:</strong> {status.schedules}</span>
          <span><strong>Secrets:</strong> {status.secrets}</span>
          <span>
            <strong>Custom tools:</strong> {status.custom_tools_approved} approved,{" "}
            {status.custom_tools_pending} pending
          </span>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3">
        <h2 className="text-lg font-semibold">Sessions</h2>
        <Button size="sm" onClick={handleCreate}>
          <Plus className="mr-1 h-4 w-4" /> New
        </Button>
      </div>

      {/* Session list */}
      <ScrollArea className="flex-1">
        <div className="px-3 pb-4">
          {loading ? (
            <div className="px-3 py-8 text-center text-muted-foreground">Loading…</div>
          ) : sessions.length === 0 ? (
            <div className="px-3 py-8 text-center text-muted-foreground">
              No sessions yet. Click "New" to create one.
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              {sessions.map((s) => (
                <div
                  key={s.rkey}
                  onClick={() => navigate(`/sessions/${s.rkey}`)}
                  className="group flex cursor-pointer items-center justify-between rounded-md px-3 py-2.5 hover:bg-accent"
                >
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium">
                      {s.title || "(untitled)"}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {s.rkey} · {s.createdAt.slice(0, 19)}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 opacity-0 group-hover:opacity-100"
                    onClick={(e) => handleDelete(e, s.rkey)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
