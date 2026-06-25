interface TokenStatsProps {
  ctx: number;
  tps: number | null;
  input: number;
  output: number;
  calls: number;
}

export function TokenStats({ ctx, tps, input, output, calls }: TokenStatsProps) {
  if (calls === 0) return null;

  return (
    <div className="flex items-center gap-4 border-t border-border px-4 py-1 text-xs text-muted-foreground">
      <span>ctx: {ctx.toLocaleString()}</span>
      {tps !== null && <span>tps: {tps.toFixed(1)}</span>}
      <span>in: {input.toLocaleString()}</span>
      <span>out: {output.toLocaleString()}</span>
      <span>calls: {calls}</span>
    </div>
  );
}
