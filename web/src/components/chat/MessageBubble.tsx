import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

export function MessageBubble({ role, content }: { role: string; content: string }) {
  const isUser = role === "user";
  const isSystem = role === "system";

  return (
    <div
      className={cn(
        "flex flex-col gap-1 px-4 py-2",
        isUser && "items-end",
      )}
    >
      <div className="flex items-baseline gap-2">
        <span className="text-xs font-semibold text-muted-foreground">
          {isUser ? "You" : isSystem ? "System" : "Agent"}
        </span>
      </div>
      <div
        className={cn(
          "max-w-[85%] rounded-lg px-4 py-2 text-sm",
          isUser && "bg-primary text-primary-foreground",
          !isUser && !isSystem && "bg-secondary text-secondary-foreground",
          isSystem && "bg-destructive/10 text-destructive-foreground border border-destructive/30",
        )}
      >
        {isUser ? (
          <div className="whitespace-pre-wrap">{content}</div>
        ) : (
          <div className="prose prose-sm prose-invert max-w-none break-words">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
