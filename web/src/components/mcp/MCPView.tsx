import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Plus, Trash2 } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { MCPServerSummary } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";

export function MCPView() {
  const navigate = useNavigate();
  const [servers, setServers] = useState<MCPServerSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);

  // form state
  const [name, setName] = useState("");
  const [transport, setTransport] = useState<"sse" | "stdio">("sse");
  const [url, setUrl] = useState("");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [authType, setAuthType] = useState<"none" | "bearer" | "oauth">("none");
  const [authTokenSecret, setAuthTokenSecret] = useState("");
  const [envSecrets, setEnvSecrets] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      setServers(await api.listMCPServers());
    } catch (e) {
      console.error("Failed to load MCP servers:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleCreate = async () => {
    setCreating(true);
    setError("");
    try {
      await api.createMCPServer({
        name,
        transport,
        url: transport === "sse" ? url || undefined : undefined,
        command: transport === "stdio" ? command || undefined : undefined,
        args: transport === "stdio" && args ? args.split(/\s+/) : [],
        auth_type: authType,
        auth_token_secret: (authType === "bearer" && authTokenSecret) ? authTokenSecret : undefined,
        env_secrets: transport === "stdio" && envSecrets ? envSecrets.split(",").map((s) => s.trim()).filter(Boolean) : [],
        enabled: true,
      });
      setShowAdd(false);
      setName("");
      setUrl("");
      setCommand("");
      setArgs("");
      setAuthType("none");
      setAuthTokenSecret("");
      setEnvSecrets("");
      await refresh();
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.message);
      } else {
        setError("Failed to create server");
      }
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, name: string) => {
    e.stopPropagation();
    if (!confirm(`Delete MCP server "${name}"?`)) return;
    try {
      await api.deleteMCPServer(name);
      await refresh();
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 px-6 py-3">
        <h2 className="text-lg font-semibold">MCP Servers</h2>
        <Button size="sm" variant="outline" onClick={() => setShowAdd(true)}>
          <Plus className="mr-1 h-4 w-4" /> Add Server
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="px-3 pb-4">
          {loading ? (
            <div className="px-3 py-8 text-center text-muted-foreground">Loading…</div>
          ) : servers.length === 0 ? (
            <div className="px-3 py-8 text-center text-muted-foreground">
              No MCP servers configured. Click "Add Server" to connect to an external MCP server.
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              {servers.map((s) => (
                <div
                  key={s.name}
                  onClick={() => navigate(`/mcp/${s.name}`)}
                  className="group flex cursor-pointer items-center gap-3 rounded-md px-3 py-2.5 hover:bg-accent"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">{s.name}</span>
                      <Badge variant="outline">{s.transport}</Badge>
                      {s.connected ? (
                        <Badge className="bg-green-600 text-white">connected</Badge>
                      ) : (
                        <Badge variant="secondary" className="bg-zinc-500 text-white">disconnected</Badge>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {s.tools_approved}/{s.tools_total} tools approved
                    </div>
                  </div>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
                    <Button variant="ghost" size="icon" className="h-7 w-7" onClick={(e) => handleDelete(e, s.name)}>
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
            <DialogTitle>Add MCP Server</DialogTitle>
            <DialogDescription>
              Connect to an external MCP server. Tools are discovered and individually approved.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            <div>
              <label className="text-sm font-medium">Name</label>
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. fastmail" />
            </div>
            <div>
              <label className="text-sm font-medium">Transport</label>
              <div className="flex gap-2 mt-1">
                <Button
                  size="sm"
                  variant={transport === "sse" ? "default" : "outline"}
                  onClick={() => setTransport("sse")}
                >
                  SSE (HTTP)
                </Button>
                <Button
                  size="sm"
                  variant={transport === "stdio" ? "default" : "outline"}
                  onClick={() => setTransport("stdio")}
                >
                  stdio
                </Button>
              </div>
            </div>
            {transport === "sse" ? (
              <div>
                <label className="text-sm font-medium">URL</label>
                <Input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://api.example.com/mcp" />
              </div>
            ) : (
              <>
                <div>
                  <label className="text-sm font-medium">Command</label>
                  <Input value={command} onChange={(e) => setCommand(e.target.value)} placeholder="e.g. npx" />
                </div>
                <div>
                  <label className="text-sm font-medium">Args (space-separated)</label>
                  <Input value={args} onChange={(e) => setArgs(e.target.value)} placeholder="-y @some/mcp-server" />
                </div>
              </>
            )}
            <div>
              <label className="text-sm font-medium">Authentication</label>
              <div className="flex gap-2 mt-1">
                <Button
                  size="sm"
                  variant={authType === "none" ? "default" : "outline"}
                  onClick={() => setAuthType("none")}
                >
                  None
                </Button>
                <Button
                  size="sm"
                  variant={authType === "bearer" ? "default" : "outline"}
                  onClick={() => setAuthType("bearer")}
                  disabled={transport === "stdio"}
                >
                  Bearer Token
                </Button>
                <Button
                  size="sm"
                  variant={authType === "oauth" ? "default" : "outline"}
                  onClick={() => setAuthType("oauth")}
                  disabled={transport === "stdio"}
                >
                  OAuth
                </Button>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {authType === "oauth" && "Connects via OAuth consent screen. You'll authorize via browser on first connect."}
                {authType === "bearer" && "Uses a static API token from the Secrets store."}
                {authType === "none" && "No authentication."}
              </p>
            </div>
            {authType === "bearer" && (
              <div>
                <label className="text-sm font-medium">Auth Token Secret Name</label>
                <Input
                  value={authTokenSecret}
                  onChange={(e) => setAuthTokenSecret(e.target.value)}
                  placeholder="e.g. FASTMAIL_API_TOKEN"
                />
                <p className="text-xs text-muted-foreground mt-1">Resolved from the Secrets store at connection time.</p>
              </div>
            )}
            {transport === "stdio" && (
              <div>
                <label className="text-sm font-medium">Env Secret Names (comma-separated, optional)</label>
                <Input
                  value={envSecrets}
                  onChange={(e) => setEnvSecrets(e.target.value)}
                  placeholder="API_KEY, DATABASE_URL"
                />
              </div>
            )}
            {error && <div className="text-sm text-red-500">{error}</div>}
            <Button onClick={handleCreate} disabled={creating || !name}>
              {creating ? "Creating…" : "Create Server"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
