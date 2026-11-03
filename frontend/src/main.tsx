import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ApiProvider } from "./api-context.tsx";
import { App } from "./App.tsx";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode><ApiProvider><App /></ApiProvider></StrictMode>,
);
