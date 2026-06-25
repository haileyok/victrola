import { useEffect, useState, useCallback } from "react";
import { Plus, Trash2, Power, PowerOff } from "lucide-react";
import { api } from "@/lib/api";
import type { Schedule } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

export function SchedulesView() {
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [name, setName] = useState("");
  const [schedule, setSchedule] = useState("");
  const [prompt, setPrompt] = useState("");
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      setSchedules(await api.listSchedules());
    } catch (e) {
      console.error("Failed to load schedules:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleCreate = async () => {
    setError("");
    if (!name || !schedule || !prompt) {
      setError("All fields are required");
      return;
    }
    try {
      await api.createSchedule(name, schedule, prompt);
      setName("");
      setSchedule("");
      setPrompt("");
      setShowAdd(false);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create schedule");
    }
  };

  const handleToggle = async (s: Schedule) => {
    try {
      if (s.enabled) {
        await api.disableSchedule(s.name);
      } else {
        await api.enableSchedule(s.name);
      }
      await refresh();
    } catch (e) {
      console.error("Failed to toggle schedule:", e);
    }
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete schedule "${name}"?`)) return;
    try {
      await api.deleteSchedule(name);
      await refresh();
    } catch (e) {
      console.error("Failed to delete schedule:", e);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-6 py-3">
        <h2 className="text-lg font-semibold">Schedules</h2>
        <Button size="sm" onClick={() => setShowAdd(true)}>
          <Plus className="mr-1 h-4 w-4" /> New
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="px-3 pb-4">
          {loading ? (
            <div className="px-3 py-8 text-center text-muted-foreground">Loading…</div>
          ) : schedules.length === 0 ? (
            <div className="px-3 py-8 text-center text-muted-foreground">
              No scheduled tasks. Click "New" to create one.
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              {schedules.map((s) => (
                <div
                  key={s.name}
                  className="group flex items-center gap-3 rounded-md px-3 py-2.5 hover:bg-accent"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">{s.name}</span>
                      {s.enabled ? (
                        <Badge className="bg-green-600 text-white">enabled</Badge>
                      ) : (
                        <Badge variant="secondary" className="bg-red-600 text-white">disabled</Badge>
                      )}
                    </div>
                    <div className="truncate text-xs text-muted-foreground">
                      {s.schedule} · {s.prompt.slice(0, 60)}
                    </div>
                    {s.next_run && (
                      <div className="text-xs text-muted-foreground">
                        next: {new Date(s.next_run).toLocaleString()}
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => handleToggle(s)}
                    >
                      {s.enabled ? (
                        <PowerOff className="h-3.5 w-3.5 text-yellow-500" />
                      ) : (
                        <Power className="h-3.5 w-3.5 text-green-500" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => handleDelete(s.name)}
                    >
                      <Trash2 className="h-3.5 w-3.5 text-red-500" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>

      <Dialog open={showAdd} onOpenChange={setShowAdd}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New Schedule</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            {error && <div className="text-sm text-red-500">{error}</div>}
            <div>
              <label className="text-sm font-medium">Name</label>
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="my_task" />
            </div>
            <div>
              <label className="text-sm font-medium">Schedule expression</label>
              <Input
                value={schedule}
                onChange={(e) => setSchedule(e.target.value)}
                placeholder='e.g. "30m", "daily@9:00", "weekly@monday"'
              />
            </div>
            <div>
              <label className="text-sm font-medium">Prompt</label>
              <Textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Instruction for the agent…"
              />
            </div>
            <Button onClick={handleCreate}>Create</Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
