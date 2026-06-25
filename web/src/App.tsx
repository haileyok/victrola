import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import { SessionList } from "./components/sessions/SessionList";
import { ChatView } from "./components/chat/ChatView";
import { ToolsView } from "./components/tools/ToolsView";
import { ToolDetail } from "./components/tools/ToolDetail";
import { SecretsView } from "./components/secrets/SecretsView";
import { SchedulesView } from "./components/schedules/SchedulesView";
import { SystemPromptView } from "./components/system-prompt/SystemPromptView";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/sessions" replace />} />
          <Route path="/sessions" element={<SessionList />} />
          <Route path="/sessions/:id" element={<ChatView />} />
          <Route path="/tools" element={<ToolsView />} />
          <Route path="/tools/:name" element={<ToolDetail />} />
          <Route path="/secrets" element={<SecretsView />} />
          <Route path="/schedules" element={<SchedulesView />} />
          <Route path="/system-prompt" element={<SystemPromptView />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
