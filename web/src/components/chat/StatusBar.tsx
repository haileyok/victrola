interface StatusBarProps {
  thinking: boolean;
  toolName: string | null;
}

export function StatusBar({ thinking, toolName }: StatusBarProps) {
  const visible = thinking || toolName !== null;
  if (!visible) return null;

  return (
    <div className="px-4 py-1 text-sm text-muted-foreground">
      {toolName ? (
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-blue-500" />
          Running: {toolName}
        </span>
      ) : (
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-yellow-500" />
          Thinking…
        </span>
      )}
    </div>
  );
}
