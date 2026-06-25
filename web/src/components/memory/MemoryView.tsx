import { useEffect, useState, useCallback } from "react";
import { Plus, Trash2, Pencil, Search, X, ChevronDown, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import type { MemoryEntry, MemorySearchResult } from "@/lib/types";
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
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";

const TYPE_PILLS = ["self", "operator", "skill", "episodic", "factual"];

function typeColor(type: string): string {
  switch (type) {
    case "self":
      return "bg-purple-600 text-white";
    case "operator":
      return "bg-blue-600 text-white";
    case "skill":
      return "bg-green-600 text-white";
    case "episodic":
      return "bg-orange-600 text-white";
    case "factual":
      return "bg-teal-600 text-white";
    default:
      return "bg-gray-600 text-white";
  }
}

function getTags(entry: MemoryEntry): string[] {
  const tags = entry.metadata?.tags;
  return Array.isArray(tags) ? (tags as string[]) : [];
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

export function MemoryView() {
  const [entries, setEntries] = useState<MemoryEntry[]>([]);
  const [cursor, setCursor] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeType, setActiveType] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<MemorySearchResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [showAdd, setShowAdd] = useState(false);
  const [editingEntry, setEditingEntry] = useState<MemoryEntry | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());
  // dialog form state
  const [formType, setFormType] = useState("episodic");
  const [formScope, setFormScope] = useState("");
  const [formContent, setFormContent] = useState("");
  const [formTags, setFormTags] = useState("");
  const [error, setError] = useState("");
  const [listError, setListError] = useState("");

  const refresh = useCallback(async () => {
    setListError("");
    try {
      const result = await api.listMemory(activeType ?? undefined);
      setEntries(result.entries);
      setCursor(result.cursor);
    } catch (e) {
      setListError(e instanceof Error ? e.message : "Failed to load memory entries");
      console.error("Failed to load memory entries:", e);
    } finally {
      setLoading(false);
    }
  }, [activeType]);

  useEffect(() => {
    setLoading(true);
    setSearchResults(null);
    refresh();
  }, [refresh]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    setError("");
    try {
      const result = await api.searchMemory(searchQuery, activeType ?? undefined);
      setSearchResults(result.results);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setSearching(false);
    }
  };

  const clearSearch = () => {
    setSearchQuery("");
    setSearchResults(null);
    setError("");
  };

  const toggleExpand = (id: number) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const openAdd = () => {
    setFormType("episodic");
    setFormScope("");
    setFormContent("");
    setFormTags("");
    setError("");
    setShowAdd(true);
  };

  const openEdit = (entry: MemoryEntry) => {
    setEditingEntry(entry);
    setFormType(entry.type);
    setFormScope(entry.scope);
    setFormContent(entry.content);
    setFormTags(getTags(entry).join(", "));
    setError("");
    setShowAdd(true);
  };

  const closeDialog = () => {
    setShowAdd(false);
    setEditingEntry(null);
    setError("");
  };

  const handleCreate = async () => {
    setError("");
    if (!formContent) {
      setError("Content is required");
      return;
    }
    let scope = formScope;
    if (formType === "self") {
      scope = "self";
    } else if (formType === "skill" && !scope.startsWith("skill:")) {
      scope = `skill:${scope}`;
    }
    if (!scope && formType !== "self") {
      setError("Scope is required");
      return;
    }
    try {
      const tags = formTags
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      await api.createMemory(formType, scope, formContent, tags.length > 0 ? tags : undefined);
      setShowAdd(false);
      setFormType("episodic");
      setFormScope("");
      setFormContent("");
      setFormTags("");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create entry");
    }
  };

  const handleUpdate = async () => {
    if (!editingEntry) return;
    setError("");
    if (!formContent) {
      setError("Content is required");
      return;
    }
    try {
      const tags = formTags
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      // Only send fields that changed — avoids unnecessary embedding regeneration
      const contentChanged = formContent !== editingEntry.content;
      const oldTags = getTags(editingEntry);
      const tagsChanged = JSON.stringify(tags) !== JSON.stringify(oldTags);
      if (!contentChanged && !tagsChanged) {
        setShowAdd(false);
        setEditingEntry(null);
        return;
      }
      await api.updateMemory(
        editingEntry.id,
        contentChanged ? formContent : undefined,
        tagsChanged ? tags : undefined,
      );
      setShowAdd(false);
      setEditingEntry(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to update entry");
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Delete this memory entry?")) return;
    try {
      await api.deleteMemory(id);
      await refresh();
    } catch (e) {
      console.error("Failed to delete entry:", e);
    }
  };

  const handleLoadMore = async () => {
    if (cursor === null) return;
    try {
      const result = await api.listMemory(activeType ?? undefined, 50, cursor);
      setEntries((prev) => [...prev, ...result.entries]);
      setCursor(result.cursor);
    } catch (e) {
      console.error("Failed to load more:", e);
    }
  };

  const isEditing = editingEntry !== null;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-6 py-3">
        <h2 className="text-lg font-semibold">Memory</h2>
        <Button size="sm" onClick={openAdd}>
          <Plus className="mr-1 h-4 w-4" /> Add
        </Button>
      </div>

      {/* Search bar */}
      <div className="flex items-center gap-2 px-6 pb-2">
        <Input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="Search memory…"
          className="flex-1"
        />
        <Button size="sm" variant="secondary" onClick={handleSearch} disabled={searching}>
          <Search className="mr-1 h-4 w-4" /> Search
        </Button>
        {searchResults && (
          <Button size="sm" variant="ghost" onClick={clearSearch}>
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      {error && searchResults !== null && (
        <div className="px-6 pb-2 text-sm text-red-500">{error}</div>
      )}

      {/* List load error */}
      {listError && !searchResults && (
        <div className="px-6 pb-2 text-sm text-red-500">{listError}</div>
      )}

      {/* Type filter pills */}
      {!searchResults && (
        <div className="flex items-center gap-1 px-6 pb-2 flex-wrap">
          <Button
            size="sm"
            variant={activeType === null ? "default" : "ghost"}
            onClick={() => setActiveType(null)}
          >
            All
          </Button>
          {TYPE_PILLS.map((t) => (
            <Button
              key={t}
              size="sm"
              variant={activeType === t ? "default" : "ghost"}
              onClick={() => setActiveType(t)}
            >
              {t}
            </Button>
          ))}
        </div>
      )}

      <ScrollArea className="flex-1">
        <div className="px-3 pb-4">
          {/* Search results */}
          {searchResults ? (
            searchResults.length === 0 ? (
              <div className="px-3 py-8 text-center text-muted-foreground">
                No results found.
              </div>
            ) : (
              <div className="flex flex-col gap-1 px-3">
                {searchResults.map((r) => (
                  <div
                    key={r.id}
                    className="group rounded-md border border-border p-3"
                  >
                    <div className="flex items-center gap-2 flex-wrap mb-1">
                      <Badge className={typeColor(r.type)}>{r.type}</Badge>
                      <span className="text-xs text-muted-foreground">{r.scope}</span>
                      <Badge variant="outline" className="border-blue-500 text-blue-500">
                        score: {r.score.toFixed(2)}
                      </Badge>
                      <Badge variant="outline" className="border-purple-500 text-purple-500">
                        {r.matched_by}
                      </Badge>
                    </div>
                    <pre className="whitespace-pre-wrap text-sm font-sans">{r.content}</pre>
                  </div>
                ))}
              </div>
            )
          ) : loading ? (
            <div className="px-3 py-8 text-center text-muted-foreground">Loading…</div>
          ) : entries.length === 0 ? (
            <div className="px-3 py-8 text-center text-muted-foreground">
              No memory entries. Click "Add" to create one.
            </div>
          ) : (
            <div className="flex flex-col gap-1 px-3">
              {entries.map((entry) => {
                const expanded = expandedIds.has(entry.id);
                const tags = getTags(entry);
                return (
                  <Collapsible key={entry.id} open={expanded} onOpenChange={() => toggleExpand(entry.id)}>
                    <div className="group flex items-start gap-3 rounded-md px-3 py-2.5 hover:bg-accent">
                      <CollapsibleTrigger asChild>
                        <button className="mt-0.5 flex-shrink-0">
                          {expanded ? (
                            <ChevronDown className="h-4 w-4 text-muted-foreground" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-muted-foreground" />
                          )}
                        </button>
                      </CollapsibleTrigger>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge className={typeColor(entry.type)}>{entry.type}</Badge>
                          <span className="text-xs text-muted-foreground">{entry.scope}</span>
                          {tags.map((tag) => (
                            <Badge key={tag} variant="outline">{tag}</Badge>
                          ))}
                          <span className="text-xs text-muted-foreground ml-auto">
                            {formatRelative(entry.updatedAt)}
                          </span>
                        </div>
                        <CollapsibleContent>
                          <pre className="mt-2 whitespace-pre-wrap text-sm font-sans">
                            {entry.content}
                          </pre>
                          {Object.keys(entry.metadata).length > 0 && (
                            <details className="mt-2">
                              <summary className="text-xs text-muted-foreground cursor-pointer">
                                metadata
                              </summary>
                              <pre className="mt-1 text-xs text-muted-foreground overflow-x-auto">
                                {JSON.stringify(entry.metadata, null, 2)}
                              </pre>
                            </details>
                          )}
                        </CollapsibleContent>
                        {!expanded && (
                          <div className="truncate text-sm text-muted-foreground mt-0.5">
                            {entry.content.slice(0, 100)}
                            {entry.content.length > 100 && "…"}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          title="Edit"
                          onClick={() => openEdit(entry)}
                        >
                          <Pencil className="h-3.5 w-3.5 text-blue-500" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          title="Delete"
                          onClick={() => handleDelete(entry.id)}
                        >
                          <Trash2 className="h-3.5 w-3.5 text-red-500" />
                        </Button>
                      </div>
                    </div>
                  </Collapsible>
                );
              })}
              {cursor !== null && (
                <div className="px-3 py-2 text-center">
                  <Button size="sm" variant="ghost" onClick={handleLoadMore}>
                    Load more
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Add/Edit dialog */}
      <Dialog open={showAdd} onOpenChange={(o) => !o && closeDialog()}>
        <DialogContent className="max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {isEditing ? "Edit Memory Entry" : "New Memory Entry"}
            </DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            {error && <div className="text-sm text-red-500">{error}</div>}
            <div>
              <label className="text-sm font-medium">Type</label>
              <select
                value={formType}
                onChange={(e) => setFormType(e.target.value)}
                disabled={isEditing}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm disabled:opacity-50"
              >
                {TYPE_PILLS.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-sm font-medium">Scope</label>
              <Input
                value={formScope}
                onChange={(e) => setFormScope(e.target.value)}
                disabled={isEditing || formType === "self"}
                placeholder={
                  formType === "self" ? "self"
                  : formType === "skill" ? "skill:my-skill (e.g. deploy)"
                  : formType === "operator" ? "operator"
                  : "topic, session ID, or free-form"
                }
              />
              {!isEditing && formType === "skill" && formScope && !formScope.startsWith("skill:") && (
                <p className="text-xs text-muted-foreground mt-1">
                  Will be saved as <code>skill:{formScope}</code>
                </p>
              )}
            </div>
            <div>
              <label className="text-sm font-medium">Content</label>
              <Textarea
                value={formContent}
                onChange={(e) => setFormContent(e.target.value)}
                placeholder="Memory content…"
                rows={6}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Tags (comma-separated)</label>
              <Input
                value={formTags}
                onChange={(e) => setFormTags(e.target.value)}
                placeholder="deploy, ops, important"
              />
            </div>
            <Button onClick={isEditing ? handleUpdate : handleCreate}>
              {isEditing ? "Save" : "Create"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
