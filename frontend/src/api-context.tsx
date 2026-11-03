import { createContext, useContext, type ReactNode } from "react";
import { createClient } from "./lib/api.ts";
import { makeApi, type Api } from "./lib/endpoints.ts";

// Same-origin by default (Vite proxies /api in dev). No keys or secrets exist in the frontend.
const baseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "/api/v1";
const defaultApi: Api = makeApi(createClient({ baseUrl }));
const Ctx = createContext<Api>(defaultApi);

export function ApiProvider({ api, children }: { api?: Api; children: ReactNode }) {
  return <Ctx.Provider value={api ?? defaultApi}>{children}</Ctx.Provider>;
}
export const useApi = () => useContext(Ctx);
export const API_BASE_URL = baseUrl;
