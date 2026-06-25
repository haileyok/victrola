import { useEffect, useState, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, Check, X, Trash2, FlaskConical } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { ToolDetail as ToolDetailType } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";

export function ToolDetail() {
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const [tool, setTool] = useState<ToolDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [missingSecrets, setMissingSecrets] = useState<string[] | null>(null);
  const [secretName, setSecretName] = useState("");
  const [secretValue, setSecretValue] = useState("");

  const refresh = useCallback(async () => {
    if (!name) return;
    try {
      setTool(await api.getTool(name));
    } catch (e) {
      console.error("Failed to load tool:", e);
    } finally {
      setLoading(false);
    }
  }, [name]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleApprove = async () => {
    if (!name) return;
    try {
      await api.approveTool(name);
      await refresh();
    } catch (e) {
      // check for missing secrets in structured error detail
      if (e instanceof ApiError && e.status === 400) {
        const detail = e.detail as { missing_secrets?: string[] } | undefined;
        if (detail?.missing_secrets) {
          setMissingSecrets(detail.missing_secrets);
          setSecretName(detail.missing_secrets[0] || "");
          return;
        }
      }
      console.error("Failed to approve:", e);
    }
  };

  const handleSaveSecret = async () => {
    if (!secretName || !secretValue) return;
    try {
      await api.setSecret(secretName, secretValue);
      setSecretValue("");
      // advance to next missing secret or retry approve
      if (missingSecrets && missingSecrets.length > 1) {
        const next = missingSecrets.slice(1);
        setMissingSecrets(next);
        setSecretName(next[0] || "");
      } else {
        setMissingSecrets(null);
        // retry approval now that secrets are set
        try {
          await api.approveTool(name!);
          await refresh();
        } catch (e) {
          console.error("Approval still failed:", e);
        }
      }
    } catch (e) {
      console.error("Failed to save secret:", e);
    }
  };

  const handleRevoke = async () => {
    if (!name) return;
    try {
      await api.revokeTool(name);
      await refresh();
    } catch (e) {
      console.error("Failed to revoke:", e);
    }
  };

  const handleDelete = async () => {
    if (!name || !confirm(`Delete tool "${name}"?`)) return;
    try {
      await api.deleteTool(name);
      navigate("/tools");
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  const [testResult, setTestResult] = useState<Record<string, unknown> | null>(null);
  const [testing, setTesting] = useState(false);

  const handleTest = async () => {
    if (!name) return;
    setTesting(true);
    setTestResult(null);
    try {
      const result = await api.testTool(name);
      setTestResult(result);
    } catch (e) {
      setTestResult({ error: e instanceof Error ? e.message : "Test failed" });
    } finally {
      setTesting(false);
    }
  };

  if (loading) return <div className="p-6 text-muted-foreground">Loading…</div>;
  if (!tool) return <div className="p-6 text-muted-foreground">Tool not found.</div>;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-border px-6 py-3">
        <Button variant="ghost" size="icon" onClick={() => navigate("/tools")}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <h2 className="text-lg font-semibold">{tool.name}</h2>
        {tool.approved ? (
          <Badge className="bg-green-600 text-white">approved</Badge>
        ) : (
          <Badge variant="secondary" className="bg-yellow-600 text-white">pending</Badge>
        )}
        <div className="ml-auto flex gap-2">
          {!tool.approved && (
            <Button size="sm" onClick={handleApprove}>
              <Check className="mr-1 h-4 w-4" /> Approve
            </Button>
          )}
          {tool.approved && (
            <Button size="sm" variant="outline" onClick={handleRevoke}>
              <X className="mr-1 h-4 w-4" /> Revoke
            </Button>
          )}
          <Button size="sm" variant="outline" onClick={handleTest} disabled={testing}>
            <FlaskConical className="mr-1 h-4 w-4" /> Test
          </Button>
          <Button size="sm" variant="destructive" onClick={handleDelete}>
            <Trash2 className="mr-1 h-4 w-4" /> Delete
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="mb-4">
          <div className="text-xs font-medium text-muted-foreground">Description</div>
          <div className="text-sm">{tool.description}</div>
        </div>

        <div className="mb-4">
          <div className="text-xs font-medium text-muted-foreground">Requires Network</div>
          <div className="text-sm">{tool.requires_net ? "Yes — tool can make network calls" : "No (sandboxed)"}</div>
        </div>

        {tool.secrets.length > 0 && (
          <div className="mb-4">
            <div className="mb-1 text-xs font-medium text-muted-foreground">Secrets</div>
            <div className="flex flex-col gap-1">
              {tool.secrets.map((s) => (
                <div key={s.name} className="flex items-center gap-2 text-sm">
                  <code className="rounded bg-muted px-1.5 py-0.5">{s.name}</code>
                  {s.status === "set" ? (
                    <Badge className="bg-green-600 text-white">set</Badge>
                  ) : (
                    <Badge variant="secondary" className="bg-yellow-600 text-white">missing</Badge>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="mb-4">
          <div className="mb-1 text-xs font-medium text-muted-foreground">Parameters</div>
          <pre className="overflow-x-auto rounded-md bg-zinc-900 p-3 text-xs text-zinc-100">
            <code>{JSON.stringify(tool.parameters, null, 2)}</code>
          </pre>
        </div>

        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Code</div>
          <pre className="overflow-x-auto rounded-md bg-zinc-900 p-3 text-xs text-zinc-100">
            <code>{tool.code}</code>
          </pre>
        </div>

        {testResult !== null && (
          <div>
            <div className="mb-1 text-xs font-medium text-muted-foreground">Test Result</div>
            <pre className="overflow-x-auto rounded-md bg-zinc-900 p-3 text-xs text-zinc-100">
              <code>{JSON.stringify(testResult, null, 2)}</code>
            </pre>
          </div>
        )}
      </div>

      {/* Missing secret dialog */}
      <Dialog open={missingSecrets !== null && missingSecrets.length > 0} onOpenChange={(o) => !o && setMissingSecrets(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Set missing secret</DialogTitle>
            <DialogDescription>
              This tool requires secrets that aren't configured yet. Set them to complete approval.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            <div>
              <label className="text-sm font-medium">Secret name</label>
              <Input value={secretName} onChange={(e) => setSecretName(e.target.value)} disabled />
            </div>
            <div>
              <label className="text-sm font-medium">Secret value</label>
              <Input
                type="password"
                value={secretValue}
                onChange={(e) => setSecretValue(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSaveSecret()}
                placeholder="Enter value…"
              />
            </div>
            <Button onClick={handleSaveSecret} disabled={!secretValue}>
              Save Secret
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
