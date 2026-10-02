import { createRoot } from "react-dom/client";
import App, { ErrorBoundary } from "./App";
import { UiProvider } from "./components/ui";
import "./style.css";
createRoot(document.getElementById("root")!).render(
  <ErrorBoundary>
    <UiProvider>
      <App />
    </UiProvider>
  </ErrorBoundary>,
);
