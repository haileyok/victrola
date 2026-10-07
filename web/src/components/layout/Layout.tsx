import { useEffect, useRef } from "react";
import { Outlet, Link, useLocation } from "react-router-dom";
import { MessageSquare, Wrench, Key, Clock, FileText, Plug, Brain, FolderOpen } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { to: "/sessions", label: "Sessions", icon: MessageSquare },
  { to: "/tools", label: "Tools", icon: Wrench },
  { to: "/mcp", label: "MCP", icon: Plug },
  { to: "/workspace", label: "Workspace", icon: FolderOpen },
  { to: "/secrets", label: "Secrets", icon: Key },
  { to: "/schedules", label: "Schedules", icon: Clock },
  { to: "/system-prompt", label: "Prompt", icon: FileText },
  { to: "/memory", label: "Memory", icon: Brain },
];

export function Layout() {
  const location = useLocation();
  const mobileNavRef = useRef<HTMLElement>(null);

  // Keep the active chip in view when the mobile nav's scroll strip
  // is narrower than its content (e.g. deep links straight to /memory).
  useEffect(() => {
    const active = mobileNavRef.current?.querySelector<HTMLAnchorElement>('a[data-active="true"]');
    if (active && typeof active.scrollIntoView === "function") {
      active.scrollIntoView({ block: "nearest", inline: "center" });
    }
  }, [location.pathname]);

  return (
    // h-screen keeps the shell full-height on browsers without dvh support;
    // where 100dvh exists it wins, so mobile browser toolbars don't clip the view.
    <div className="flex h-screen w-full overflow-hidden [@supports(height:100dvh)]:h-dvh">
      <aside className="hidden w-56 flex-col border-r border-border bg-card md:flex">
        <div className="px-4 py-3">
          <h1 className="text-lg font-bold tracking-tight">Victrola</h1>
        </div>
        <nav aria-label="Primary" className="flex flex-col gap-1 px-2">
          {navItems.map((item) => {
            const active = location.pathname.startsWith(item.to);
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  "flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  active
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                )}
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <nav
          ref={mobileNavRef}
          aria-label="Sections"
          className="flex items-center gap-1 overflow-x-auto border-b border-border bg-card px-2 py-2 md:hidden"
        >
          {navItems.map((item) => {
            const active = location.pathname.startsWith(item.to);
            return (
              <Link
                key={item.to}
                to={item.to}
                data-active={active ? "true" : undefined}
                className={cn(
                  "flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium",
                  active
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground",
                )}
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>
        <main className="min-w-0 flex-1 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
