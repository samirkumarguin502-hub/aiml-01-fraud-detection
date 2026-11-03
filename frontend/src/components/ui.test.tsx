import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError, NetworkError } from "../lib/api.ts";
import { Async, EmptyState, ErrorState, Pagination, SyntheticBanner } from "./ui.tsx";

describe("shared components", () => {
  it("shows an offline message without inventing data", () => {
    render(<ErrorState error={new NetworkError()} />);
    expect(screen.getByRole("alert")).toHaveTextContent(/cannot reach the api/i);
    expect(screen.getByRole("alert")).toHaveTextContent(/nothing on this page is made up/i);
  });
  it("shows API error code and request id and offers retry", async () => {
    const retry = vi.fn();
    render(<ErrorState error={new ApiError(503, "model_unavailable", "No usable model", "req1")} onRetry={retry} />);
    expect(screen.getByText(/model_unavailable/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(retry).toHaveBeenCalled();
  });
  it("Async renders loading, error and data states", () => {
    const { rerender } = render(<Async state={{ status: "loading" }}>{() => <p>data</p>}</Async>);
    expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
    rerender(<Async state={{ status: "error", error: new Error("x") }}>{() => <p>data</p>}</Async>);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    rerender(<Async state={{ status: "ready", data: 1 }}>{(d) => <p>data {d}</p>}</Async>);
    expect(screen.getByText("data 1")).toBeInTheDocument();
  });
  it("pagination disables edges and reports totals", async () => {
    const onPage = vi.fn();
    render(<Pagination page={1} pageSize={10} total={25} onPage={onPage} />);
    expect(screen.getByRole("button", { name: /previous/i })).toBeDisabled();
    expect(screen.getByText(/page 1 of 3/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /next/i }));
    expect(onPage).toHaveBeenCalledWith(2);
  });
  it("empty state and synthetic banner", () => {
    render(<><EmptyState title="Nothing yet" /><SyntheticBanner origin="SYNTHETIC test fixture" /></>);
    expect(screen.getByText("Nothing yet")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(/synthetic/i);
  });
});
