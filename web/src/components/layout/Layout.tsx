import { Outlet, Link, useLocation } from "react-router-dom";
import { MessageSquare, Wrench, Key, Clock, FileText } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { to: "/sessions", label: "Sessions", icon: MessageSquare },
  { to: "/tools", label: "Tools", icon: Wrench },
  { to: "/secrets", label: "Secrets", icon: Key },
  { to: "/schedules", label: "Schedules", icon: Clock },
  { to: "/system-prompt", label: "Prompt", icon: FileText },
];

export function Layout() {
  const location = useLocation();

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <aside className="flex w-56 flex-col border-r border-border bg-card">
        <div className="px-4 py-3">
          <h1 className="text-lg font-bold tracking-tight">Victrola</h1>
        </div>
        <nav className="flex flex-col gap-1 px-2">
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
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
