import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { createClient } from "./lib/api.ts";
import { makeApi, type Api } from "./lib/endpoints.ts";

const baseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "/api/v1";
const TOKEN_KEY = "aiml01_access_token";
const USER_KEY = "aiml01_user";

function readToken() {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
}
function readUser(): AuthUser | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) as AuthUser : null;
  } catch { return null; }
}

export interface AuthUser { id: number; username: string; role: "admin" | "analyst" }
interface AuthContextValue {
  token: string | null;
  user: AuthUser | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  authenticated: boolean;
}
const AuthCtx = createContext<AuthContextValue | null>(null);

const defaultApi: Api = makeApi(createClient({
  baseUrl,
  getToken: readToken,
}));
const Ctx = createContext<Api>(defaultApi);

export function ApiProvider({ api, children }: { api?: Api; children: ReactNode }) {
  const [token, setToken] = useState<string | null>(readToken);
  const [user, setUser] = useState<AuthUser | null>(readUser);

  const auth = useMemo<AuthContextValue>(() => ({
    token,
    user,
    authenticated: Boolean(token && user),
    async login(username, password) {
      const result = await defaultApi.login(username, password) as {
        access_token: string;
        user: AuthUser;
      };
      if (!result.access_token || !result.user || !["admin", "analyst"].includes(result.user.role)) {
        throw new Error("Invalid login response from server.");
      }
      localStorage.setItem(TOKEN_KEY, result.access_token);
      localStorage.setItem(USER_KEY, JSON.stringify(result.user));
      setToken(result.access_token);
      setUser(result.user);
    },
    logout() {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
      setToken(null);
      setUser(null);
      window.location.hash = "/";
    },
  }), [token, user]);

  return (
    <AuthCtx.Provider value={auth}>
      <Ctx.Provider value={api ?? defaultApi}>{children}</Ctx.Provider>
    </AuthCtx.Provider>
  );
}
export const useApi = () => useContext(Ctx);
export function useAuth() {
  const value = useContext(AuthCtx);
  if (!value) throw new Error("useAuth must be used inside ApiProvider");
  return value;
}
export const API_BASE_URL = baseUrl;
