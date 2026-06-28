import { useEffect, useState, useCallback } from "react";
import {
  Folder,
  File as FileIcon,
  Trash2,
  ChevronRight,
  FolderPlus,
  HardDrive,
} from "lucide-react";
import { api } from "@/lib/api";
import type { WorkspaceEntry, WorkspaceFile } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

function formatSize(bytes: number | null): string {
  if (bytes === null) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatRelative(iso: string): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    return d.toLocaleDateString();
  } catch {
    return "";
  }
}

export function WorkspaceView() {
  const [currentPath, setCurrentPath] = useState("");
  const [entries, setEntries] = useState<WorkspaceEntry[]>([]);
  const [totalSize, setTotalSize] = useState(0);
  const [maxSize, setMaxSize] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [fileContent, setFileContent] = useState<WorkspaceFile | null>(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [showNewDir, setShowNewDir] = useState(false);
  const [newDirName, setNewDirName] = useState("");

  const refresh = useCallback(async (path: string) => {
    setError("");
    try {
      const result = await api.listWorkspace(path);
      setEntries(result.entries);
      setTotalSize(result.total_size_bytes);
      setMaxSize(result.max_size_bytes);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load workspace");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    setFileContent(null);
    refresh(currentPath);
  }, [currentPath, refresh]);

  const navigateTo = (path: string) => {
    setCurrentPath(path);
  };

  const navigateUp = () => {
    const parts = currentPath.split("/").filter(Boolean);
    parts.pop();
    setCurrentPath(parts.join("/"));
  };

  const breadcrumbs = currentPath
    ? currentPath.split("/").filter(Boolean)
    : [];

  const handleClick = async (entry: WorkspaceEntry) => {
    const fullPath = currentPath
      ? `${currentPath}/${entry.name}`
      : entry.name;

    if (entry.type === "directory") {
      navigateTo(fullPath);
      return;
    }

    // File — load content
    setFileLoading(true);
    setFileContent(null);
    try {
      const result = await api.readWorkspaceFile(fullPath);
      setFileContent(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to read file");
    } finally {
      setFileLoading(false);
    }
  };

  const handleDelete = async (entry: WorkspaceEntry) => {
    const fullPath = currentPath
      ? `${currentPath}/${entry.name}`
      : entry.name;
    if (!confirm(`Delete "${entry.name}"?${entry.type === "directory" ? " This will remove the directory and all its contents." : ""}`)) return;
    try {
      await api.deleteWorkspaceFile(fullPath);
      setFileContent(null);
      await refresh(currentPath);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to delete");
    }
  };

  const handleCreateDir = async () => {
    if (!newDirName.trim()) return;
    const fullPath = currentPath
      ? `${currentPath}/${newDirName.trim()}`
      : newDirName.trim();
    try {
      await api.createWorkspaceDir(fullPath);
      setNewDirName("");
      setShowNewDir(false);
      await refresh(currentPath);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create directory");
    }
  };

  const sizePercent = maxSize > 0 ? Math.min(100, (totalSize / maxSize) * 100) : 0;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-6 py-3">
        <h2 className="text-lg font-semibold">Workspace</h2>
        <Button size="sm" onClick={() => setShowNewDir(true)}>
          <FolderPlus className="mr-1 h-4 w-4" /> New Dir
        </Button>
      </div>

      {/* Size indicator */}
      <div className="flex items-center gap-2 px-6 pb-2">
        <HardDrive className="h-4 w-4 text-muted-foreground" />
        <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              sizePercent > 90 ? "bg-red-500" : "bg-blue-500"
            }`}
            style={{ width: `${sizePercent}%` }}
          />
        </div>
        <span className="text-xs text-muted-foreground whitespace-nowrap">
          {formatSize(totalSize)} / {formatSize(maxSize)}
        </span>
      </div>

      {/* Breadcrumbs */}
      <div className="flex items-center gap-1 px-6 pb-2 text-sm">
        <button
          className="text-muted-foreground hover:text-foreground"
          onClick={() => navigateTo("")}
        >
          workspace
        </button>
        {breadcrumbs.map((part, i) => {
          const path = breadcrumbs.slice(0, i + 1).join("/");
          const isLast = i === breadcrumbs.length - 1;
          return (
            <span key={path} className="flex items-center gap-1">
              <ChevronRight className="h-3 w-3 text-muted-foreground" />
              {isLast ? (
                <span className="font-medium">{part}</span>
              ) : (
                <button
                  className="text-muted-foreground hover:text-foreground"
                  onClick={() => navigateTo(path)}
                >
                  {part}
                </button>
              )}
            </span>
          );
        })}
        {currentPath && (
          <button
            className="ml-2 text-xs text-muted-foreground hover:text-foreground"
            onClick={navigateUp}
          >
            ↑ up
          </button>
        )}
      </div>

      {error && (
        <div className="px-6 pb-2 text-sm text-red-500">{error}</div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* File list */}
        <ScrollArea className="flex-1">
          <div className="px-3 pb-4">
            {loading ? (
              <div className="px-3 py-8 text-center text-muted-foreground">Loading…</div>
            ) : entries.length === 0 ? (
              <div className="px-3 py-8 text-center text-muted-foreground">
                Empty directory. The agent hasn't written any files here yet.
              </div>
            ) : (
              <div className="flex flex-col gap-0.5">
                {entries.map((entry) => (
                  <div
                    key={entry.name}
                    className="group flex items-center gap-2 rounded-md px-3 py-2 hover:bg-accent cursor-pointer"
                    onClick={() => handleClick(entry)}
                  >
                    {entry.type === "directory" ? (
                      <Folder className="h-4 w-4 text-blue-500 flex-shrink-0" />
                    ) : (
                      <FileIcon className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                    )}
                    <span className="text-sm truncate flex-1">{entry.name}</span>
                    {entry.size !== null && (
                      <span className="text-xs text-muted-foreground">
                        {formatSize(entry.size)}
                      </span>
                    )}
                    <span className="text-xs text-muted-foreground">
                      {formatRelative(entry.modified)}
                    </span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 opacity-0 group-hover:opacity-100"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(entry);
                      }}
                    >
                      <Trash2 className="h-3.5 w-3.5 text-red-500" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </ScrollArea>

        {/* File content panel */}
        {(fileContent || fileLoading) && (
          <div className="w-1/2 border-l border-border flex flex-col">
            <div className="flex items-center justify-between px-4 py-2 border-b border-border">
              <span className="text-sm font-medium truncate">
                {fileContent?.path ?? "Loading…"}
              </span>
              {fileContent && (
                <span className="text-xs text-muted-foreground ml-2">
                  {formatSize(fileContent.size)}
                </span>
              )}
            </div>
            <ScrollArea className="flex-1">
              <div className="p-4">
                {fileLoading ? (
                  <div className="text-center text-muted-foreground">Loading…</div>
                ) : fileContent ? (
                  <pre className="whitespace-pre-wrap text-sm font-mono">
                    {fileContent.content}
                  </pre>
                ) : null}
              </div>
            </ScrollArea>
          </div>
        )}
      </div>

      {/* New directory dialog */}
      <Dialog open={showNewDir} onOpenChange={setShowNewDir}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Directory</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            {error && <div className="text-sm text-red-500">{error}</div>}
            <div>
              <label className="text-sm font-medium">
                {currentPath ? `in ${currentPath}/` : "in workspace root"}
              </label>
              <Input
                value={newDirName}
                onChange={(e) => setNewDirName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreateDir()}
                placeholder="directory name"
                autoFocus
              />
            </div>
            <Button onClick={handleCreateDir} disabled={!newDirName.trim()}>
              Create
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
