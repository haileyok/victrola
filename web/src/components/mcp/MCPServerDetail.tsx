import { useEffect, useState, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Check,
  X,
  Trash2,
  Plug,
  PlugZap,
  RefreshCw,
} from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { MCPServerDetail as MCPServerDetailType } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

export function MCPServerDetail() {
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const [server, setServer] = useState<MCPServerDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    if (!name) return;
    try {
      setServer(await api.getMCPServer(name));
    } catch (e) {
      console.error("Failed to load MCP server:", e);
    } finally {
      setLoading(false);
    }
  }, [name]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleConnect = async () => {
    if (!name) return;
    setBusy(true);
    setError("");
    try {
      setServer(await api.connectMCPServer(name));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to connect");
    } finally {
      setBusy(false);
    }
  };

  const handleDisconnect = async () => {
    if (!name) return;
    setBusy(true);
    setError("");
    try {
      setServer(await api.disconnectMCPServer(name));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to disconnect");
    } finally {
      setBusy(false);
    }
  };

  const handleRefresh = async () => {
    if (!name) return;
    setBusy(true);
    setError("");
    try {
      setServer(await api.refreshMCPServer(name));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to refresh tools");
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    if (!name || !confirm(`Delete MCP server "${name}"?`)) return;
    try {
      await api.deleteMCPServer(name);
      navigate("/mcp");
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  const handleApproveTool = async (toolName: string) => {
    if (!name) return;
    try {
      await api.approveMCPTool(name, toolName);
      await refresh();
    } catch (e) {
      console.error("Failed to approve:", e);
    }
  };

  const handleRevokeTool = async (toolName: string) => {
    if (!name) return;
    try {
      await api.revokeMCPTool(name, toolName);
      await refresh();
    } catch (e) {
      console.error("Failed to revoke:", e);
    }
  };

  const handleDeauthorize = async () => {
    if (!name || !confirm("Clear OAuth tokens? You'll need to re-authorize via the consent screen.")) return;
    try {
      await api.deauthorizeMCPOAuth(name);
      await refresh();
    } catch (e) {
      console.error("Failed to deauthorize:", e);
    }
  };

  if (loading) return <div className="p-6 text-muted-foreground">Loading…</div>;
  if (!server) return <div className="p-6 text-muted-foreground">MCP server not found.</div>;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-border px-6 py-3">
        <Button variant="ghost" size="icon" onClick={() => navigate("/mcp")}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <h2 className="text-lg font-semibold">{server.name}</h2>
        <Badge variant="outline">{server.transport}</Badge>
        {server.connected ? (
          <Badge className="bg-green-600 text-white">connected</Badge>
        ) : (
          <Badge variant="secondary" className="bg-zinc-500 text-white">disconnected</Badge>
        )}
        <div className="ml-auto flex gap-2">
          {!server.connected ? (
            <Button size="sm" onClick={handleConnect} disabled={busy}>
              <Plug className="mr-1 h-4 w-4" /> Connect
            </Button>
          ) : (
            <Button size="sm" variant="outline" onClick={handleDisconnect} disabled={busy}>
              <PlugZap className="mr-1 h-4 w-4" /> Disconnect
            </Button>
          )}
          <Button size="sm" variant="outline" onClick={handleRefresh} disabled={busy || !server.connected}>
            <RefreshCw className="mr-1 h-4 w-4" /> Refresh
          </Button>
          <Button size="sm" variant="destructive" onClick={handleDelete} disabled={busy}>
            <Trash2 className="mr-1 h-4 w-4" /> Delete
          </Button>
        </div>
      </div>

      {error && (
        <div className="px-6 py-2 text-sm text-red-500 bg-red-50 dark:bg-red-950/20">
          {error}
        </div>
      )}

      <ScrollArea className="flex-1">
        <div className="p-6">
          {/* Config */}
          <div className="mb-6 space-y-2">
            <div className="text-xs font-medium text-muted-foreground">Configuration</div>
            {server.transport === "sse" ? (
              <div className="text-sm">
                <span className="text-muted-foreground">URL: </span>
                <code className="rounded bg-muted px-1.5 py-0.5">{server.url || "—"}</code>
              </div>
            ) : (
              <>
                <div className="text-sm">
                  <span className="text-muted-foreground">Command: </span>
                  <code className="rounded bg-muted px-1.5 py-0.5">{server.command || "—"}</code>
                </div>
                {server.args.length > 0 && (
                  <div className="text-sm">
                    <span className="text-muted-foreground">Args: </span>
                    <code className="rounded bg-muted px-1.5 py-0.5">{server.args.join(" ")}</code>
                  </div>
                )}
              </>
            )}
            {server.auth_token_secret && (
              <div className="flex items-center gap-2 text-sm">
                <span className="text-muted-foreground">Auth token:</span>
                <code className="rounded bg-muted px-1.5 py-0.5">{server.auth_token_secret}</code>
                {server.auth_token_status === "set" ? (
                  <Badge className="bg-green-600 text-white">set</Badge>
                ) : (
                  <Badge variant="secondary" className="bg-yellow-600 text-white">missing</Badge>
                )}
              </div>
            )}
            {server.auth_type === "oauth" && (
              <div className="flex items-center gap-2 text-sm">
                <span className="text-muted-foreground">OAuth:</span>
                {server.oauth_status === "authorized" ? (
                  <>
                    <Badge className="bg-green-600 text-white">authorized</Badge>
                    <Button size="sm" variant="outline" onClick={handleDeauthorize}>
                      Deauthorize
                    </Button>
                  </>
                ) : server.oauth_status === "not_authorized" ? (
                  <>
                    <Badge variant="secondary" className="bg-yellow-600 text-white">not authorized</Badge>
                    <span className="text-xs text-muted-foreground">
                      Click "Connect" to start the OAuth consent flow.
                    </span>
                  </>
                ) : (
                  <Badge variant="outline">{server.oauth_status}</Badge>
                )}
              </div>
            )}
            {server.env_secrets.length > 0 && (
              <div className="flex flex-col gap-1">
                {server.env_secrets.map((s) => (
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
            )}
          </div>

          {/* Tools */}
          <div>
            <div className="mb-2 text-xs font-medium text-muted-foreground">
              Tools ({server.tools.filter((t) => t.approved).length}/{server.tools.length} approved)
            </div>
            {!server.connected && server.tools.length === 0 ? (
              <div className="text-sm text-muted-foreground">
                Connect to this server to discover available tools.
              </div>
            ) : server.tools.length === 0 ? (
              <div className="text-sm text-muted-foreground">
                No tools discovered. Click "Refresh" to re-discover.
              </div>
            ) : (
              <div className="flex flex-col gap-1">
                {server.tools.map((tool) => (
                  <div
                    key={tool.name}
                    className="group flex items-center gap-3 rounded-md px-3 py-2 hover:bg-accent"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <code className="text-sm font-medium">{tool.name}</code>
                        {tool.approved ? (
                          <Badge className="bg-green-600 text-white">approved</Badge>
                        ) : (
                          <Badge variant="secondary" className="bg-yellow-600 text-white">pending</Badge>
                        )}
                      </div>
                      <div className="truncate text-xs text-muted-foreground">{tool.description}</div>
                    </div>
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
                      {!tool.approved && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => handleApproveTool(tool.name)}
                        >
                          <Check className="h-3.5 w-3.5 text-green-500" />
                        </Button>
                      )}
                      {tool.approved && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => handleRevokeTool(tool.name)}
                        >
                          <X className="h-3.5 w-3.5 text-yellow-500" />
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
