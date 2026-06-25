import { useEffect, useState, useCallback } from "react";
import { Plus, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import type { Secret } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

export function SecretsView() {
  const [secrets, setSecrets] = useState<Secret[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState("");
  const [newValue, setNewValue] = useState("");

  const refresh = useCallback(async () => {
    try {
      setSecrets(await api.listSecrets());
    } catch (e) {
      console.error("Failed to load secrets:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleAdd = async () => {
    if (!newName || !newValue) return;
    try {
      await api.setSecret(newName, newValue);
      setNewName("");
      setNewValue("");
      setShowAdd(false);
      await refresh();
    } catch (e) {
      console.error("Failed to save secret:", e);
    }
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete secret "${name}"?`)) return;
    try {
      await api.deleteSecret(name);
      await refresh();
    } catch (e) {
      console.error("Failed to delete secret:", e);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-6 py-3">
        <h2 className="text-lg font-semibold">Secrets</h2>
        <Button size="sm" onClick={() => setShowAdd(true)}>
          <Plus className="mr-1 h-4 w-4" /> Add
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="px-3 pb-4">
          {loading ? (
            <div className="px-3 py-8 text-center text-muted-foreground">Loading…</div>
          ) : secrets.length === 0 ? (
            <div className="px-3 py-8 text-center text-muted-foreground">
              No secrets configured. Click "Add" to create one.
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              {secrets.map((s) => (
                <div
                  key={s.name}
                  className="group flex items-center justify-between rounded-md px-3 py-2.5 hover:bg-accent"
                >
                  <div className="flex items-center gap-3">
                    <code className="text-sm font-medium">{s.name}</code>
                    <span className="text-sm text-muted-foreground">{s.masked_value}</span>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 opacity-0 group-hover:opacity-100"
                    onClick={() => handleDelete(s.name)}
                  >
                    <Trash2 className="h-3.5 w-3.5 text-red-500" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>

      <Dialog open={showAdd} onOpenChange={setShowAdd}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Secret</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            <div>
              <label className="text-sm font-medium">Name</label>
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="SECRET_NAME"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Value</label>
              <Input
                type="password"
                value={newValue}
                onChange={(e) => setNewValue(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAdd()}
                placeholder="secret value"
              />
            </div>
            <Button onClick={handleAdd} disabled={!newName || !newValue}>
              Save
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
