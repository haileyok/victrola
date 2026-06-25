import { useState } from "react";
import { ChevronDown, ChevronRight, AlertCircle, CheckCircle2 } from "lucide-react";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

interface ToolActivityProps {
  toolName: string;
  code?: string;
  result?: unknown;
  success?: boolean;
  done: boolean;
}

export function ToolActivity({ toolName, code, result, success, done }: ToolActivityProps) {
  const [open, setOpen] = useState(success === false);

  const resultStr = result !== undefined ? JSON.stringify(result, null, 2) : "";

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mx-4 my-1 rounded-lg border border-border">
      <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-1.5 text-sm hover:bg-accent">
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        <span className="font-medium">{toolName}</span>
        {done && (
          success ? (
            <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
          ) : (
            <AlertCircle className="h-3.5 w-3.5 text-red-500" />
          )
        )}
        {!done && (
          <span className="text-xs text-muted-foreground animate-pulse">running…</span>
        )}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="border-t border-border px-3 py-2">
          {code && (
            <div className="mb-2">
              <div className="mb-1 text-xs font-medium text-muted-foreground">Code</div>
              <pre className="overflow-x-auto rounded-md bg-zinc-900 p-2 text-xs text-zinc-100">
                <code>{code}</code>
              </pre>
            </div>
          )}
          {resultStr && (
            <div>
              <div className="mb-1 text-xs font-medium text-muted-foreground">Result</div>
              <pre className={cn(
                "overflow-x-auto rounded-md p-2 text-xs",
                success === false ? "bg-red-950/50 text-red-200" : "bg-zinc-900 text-zinc-100",
              )}>
                <code>{resultStr}</code>
              </pre>
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
