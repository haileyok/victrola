import { useEffect, useState, useCallback } from "react";
import { Plus, Trash2, Power, PowerOff, FlaskConical, Check, X } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { Schedule } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export function SchedulesView() {
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [name, setName] = useState("");
  const [schedule, setSchedule] = useState("");
  const [prompt, setPrompt] = useState("");
  const [conditionCode, setConditionCode] = useState("");
  const [requiresNet, setRequiresNet] = useState(false);
  const [secretsInput, setSecretsInput] = useState("");
  const [error, setError] = useState("");
  const [testResult, setTestResult] = useState<Record<string, unknown> | null>(null);
  const [testingName, setTestingName] = useState<string | null>(null);
  const [approveError, setApproveError] = useState<string | null>(null);

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
      setError("Name, schedule, and prompt are required");
      return;
    }
    try {
      const secrets = secretsInput
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      await api.createSchedule(
        name,
        schedule,
        prompt,
        conditionCode || undefined,
        requiresNet || undefined,
        secrets.length > 0 ? secrets : undefined,
      );
      setName("");
      setSchedule("");
      setPrompt("");
      setConditionCode("");
      setRequiresNet(false);
      setSecretsInput("");
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

  const handleApprove = async (s: Schedule) => {
    setApproveError(null);
    try {
      await api.approveSchedule(s.name);
      await refresh();
    } catch (e) {
      if (e instanceof ApiError && e.status === 400) {
        const detail = e.detail as { missing_secrets?: string[]; message?: string };
        if (detail?.missing_secrets) {
          setApproveError(
            `Missing secrets: ${detail.missing_secrets.join(", ")}`,
          );
        } else {
          setApproveError(detail?.message || "Approval failed");
        }
      } else {
        setApproveError("Failed to approve schedule");
      }
    }
  };

  const handleRevoke = async (s: Schedule) => {
    try {
      await api.revokeSchedule(s.name);
      await refresh();
    } catch (e) {
      console.error("Failed to revoke schedule:", e);
    }
  };

  const handleTest = async (s: Schedule) => {
    setTestingName(s.name);
    setTestResult(null);
    try {
      const result = await api.testSchedule(s.name);
      setTestResult(result);
    } catch (e) {
      setTestResult({ error: e instanceof Error ? e.message : "Test failed" });
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
      {approveError && (
        <div className="px-6 pb-2 text-sm text-red-500">{approveError}</div>
      )}
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
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium">{s.name}</span>
                      {s.enabled ? (
                        <Badge className="bg-green-600 text-white">enabled</Badge>
                      ) : (
                        <Badge variant="secondary" className="bg-red-600 text-white">disabled</Badge>
                      )}
                      {s.condition_code && (
                        <>
                          <Badge variant="outline" className="border-blue-500 text-blue-500">
                            trigger
                          </Badge>
                          {s.approved ? (
                            <Badge className="bg-green-600 text-white">approved</Badge>
                          ) : (
                            <Badge variant="secondary" className="bg-yellow-600 text-white">
                              pending
                            </Badge>
                          )}
                          {s.requires_net && (
                            <Badge variant="outline" className="border-orange-500 text-orange-500">
                              net
                            </Badge>
                          )}
                        </>
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
                    {s.condition_code && (
                      <>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          title="Test condition"
                          onClick={() => handleTest(s)}
                        >
                          <FlaskConical className="h-3.5 w-3.5 text-blue-500" />
                        </Button>
                        {s.approved ? (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7"
                            title="Revoke approval"
                            onClick={() => handleRevoke(s)}
                          >
                            <X className="h-3.5 w-3.5 text-yellow-500" />
                          </Button>
                        ) : (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7"
                            title="Approve condition"
                            onClick={() => handleApprove(s)}
                          >
                            <Check className="h-3.5 w-3.5 text-green-500" />
                          </Button>
                        )}
                      </>
                    )}
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
        <DialogContent className="max-h-[85vh] overflow-y-auto">
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
            <div className="border-t pt-3">
              <div className="text-sm font-medium mb-1">
                Condition script (optional)
              </div>
              <p className="text-xs text-muted-foreground mb-2">
                TypeScript that runs on schedule before waking the agent. Call
                <code className="mx-1 px-1 bg-muted rounded">output(&#123; wake: true &#125;)</code>
                to wake, or
                <code className="mx-1 px-1 bg-muted rounded">output(&#123; wake: false &#125;)</code>
                to skip. Requires operator approval.
              </p>
              <Textarea
                value={conditionCode}
                onChange={(e) => setConditionCode(e.target.value)}
                placeholder="// Optional: TypeScript condition script"
                className="font-mono text-xs"
                rows={5}
              />
              <div className="flex items-center gap-4 mt-2">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={requiresNet}
                    onChange={(e) => setRequiresNet(e.target.checked)}
                  />
                  Requires network
                </label>
              </div>
              <div className="mt-2">
                <label className="text-sm font-medium">Secrets (comma-separated)</label>
                <Input
                  value={secretsInput}
                  onChange={(e) => setSecretsInput(e.target.value)}
                  placeholder="API_KEY, CALENDAR_URL"
                />
              </div>
            </div>
            <Button onClick={handleCreate}>Create</Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        open={testResult !== null}
        onOpenChange={(o) => !o && setTestResult(null)}
      >
        <DialogContent className="max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Condition test result{testingName ? `: ${testingName}` : ""}</DialogTitle>
          </DialogHeader>
          <pre className="overflow-x-auto rounded-md bg-zinc-900 p-3 text-xs text-zinc-100">
            <code>{JSON.stringify(testResult, null, 2)}</code>
          </pre>
        </DialogContent>
      </Dialog>
    </div>
  );
}
