import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";
import { ApiProvider } from "../api-context.tsx";
import { ApiError, NetworkError } from "../lib/api.ts";
import type { Api } from "../lib/endpoints.ts";
import type { Summary } from "../lib/types.ts";
import { Model } from "./Model.tsx";
import { Monitoring } from "./Monitoring.tsx";
import { Overview } from "./Overview.tsx";
import { Score } from "./Score.tsx";

const withApi = (partial: Partial<Api>, ui: ReactElement) => render(<ApiProvider api={partial as Api}>{ui}</ApiProvider>);
const noModel = { state: "unavailable", reason: "none", model: null };
const preBalanceModel = { state: "ready", reason: null, model: { kind: "hist_gbm", sha256: "x", threshold: { threshold: 0.5, policy: "p", fallback_to_max_f1: false }, feature_version: "1", feature_count: 20, type_vocabulary: ["TRANSFER"], score_semantics: "n", feature_set: "pre_balance", input_fields: ["step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "oldbalanceDest"], metrics: { origin: "SYNTHETIC test fixture", metrics_are_synthetic: true, dataset_sha256: null } } };
const emptySummary: Summary = { empty: true, total_transactions: 0, flagged_transactions: 0, score_histogram: [], by_day: [], by_type: [], case_status_counts: {}, feedback: { total: 0, labelled_fraud: 0, labelled_not_fraud: 0 }, model: { state: "unavailable", reason: "No trained model", kind: null, metrics_origin: null }, notes: [] };

describe("pages", () => {
  it("overview: truthful empty state and no-model banner (no fake counts)", async () => {
    withApi({ summary: vi.fn().mockResolvedValue(emptySummary) }, <Overview />);
    expect(await screen.findByText(/no transactions scored yet/i)).toBeInTheDocument();
    expect(screen.getByText(/no usable model is loaded/i)).toBeInTheDocument();
    expect(screen.queryByText(/flag rate/i)).not.toBeInTheDocument();
  });
  it("overview: offline API shows an error, not data", async () => {
    withApi({ summary: vi.fn().mockRejectedValue(new NetworkError()) }, <Overview />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/cannot reach the api/i);
  });
  it("score form: validates before calling the API and labels the score honestly", async () => {
    const predict = vi.fn().mockResolvedValue({ transaction_id: 7, score: 0.91, threshold: 0.8, flagged: true, score_note: "n", warnings: [], model: { kind: "hist_gbm", sha256: "x", metrics_origin: "SYNTHETIC test fixture" } });
    withApi({ predict, model: vi.fn().mockResolvedValue(noModel) }, <Score />);
    await userEvent.click(screen.getByRole("button", { name: /score transaction/i }));
    expect(predict).not.toHaveBeenCalled();
    expect(screen.getAllByRole("alert").length).toBeGreaterThan(0);
    await userEvent.type(screen.getByLabelText(/step/i), "5");
    await userEvent.type(screen.getByLabelText(/amount/i), "100");
    await userEvent.type(screen.getByLabelText(/sender/i), "C1");
    await userEvent.type(screen.getByLabelText(/receiver/i), "C2");
    await userEvent.click(screen.getByRole("button", { name: /score transaction/i }));
    await waitFor(() => expect(predict).toHaveBeenCalledWith({ step: 5, type: "TRANSFER", amount: 100, nameOrig: "C1", nameDest: "C2" }));
    expect(await screen.findByText(/flagged for review/i)).toBeInTheDocument();
    expect(screen.getByText(/not a calibrated probability/i)).toBeInTheDocument();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });
  it("score form: shows 503 when no model is loaded", async () => {
    const predict = vi.fn().mockRejectedValue(new ApiError(503, "model_unavailable", "No usable trained model is loaded"));
    withApi({ predict, model: vi.fn().mockResolvedValue(noModel) }, <Score />);
    await userEvent.type(screen.getByLabelText(/step/i), "5"); await userEvent.type(screen.getByLabelText(/amount/i), "1");
    await userEvent.type(screen.getByLabelText(/sender/i), "C1"); await userEvent.type(screen.getByLabelText(/receiver/i), "C2");
    await userEvent.click(screen.getByRole("button", { name: /score transaction/i }));
    expect(await screen.findByText(/service unavailable/i)).toBeInTheDocument();
  });
  it("score form: shows and requires balance fields when the loaded model needs them", async () => {
    const predict = vi.fn().mockResolvedValue({ transaction_id: 1, score: 0.1, threshold: 0.5, flagged: false, score_note: "n", warnings: [], model: { kind: "hist_gbm", sha256: "x", metrics_origin: null } });
    withApi({ predict, model: vi.fn().mockResolvedValue(preBalanceModel) }, <Score />);
    expect(await screen.findByLabelText(/sender balance before/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/sender balance after/i)).not.toBeInTheDocument();
    await userEvent.type(screen.getByLabelText(/step/i), "5"); await userEvent.type(screen.getByLabelText(/amount/i), "1");
    await userEvent.type(screen.getByLabelText(/sender account/i), "C1"); await userEvent.type(screen.getByLabelText(/receiver account/i), "C2");
    await userEvent.click(screen.getByRole("button", { name: /score transaction/i }));
    expect(predict).not.toHaveBeenCalled();  // balances missing
    await userEvent.type(screen.getByLabelText(/sender balance before/i), "100"); await userEvent.type(screen.getByLabelText(/receiver balance before/i), "0");
    await userEvent.click(screen.getByRole("button", { name: /score transaction/i }));
    await waitFor(() => expect(predict).toHaveBeenCalledWith({ step: 5, type: "TRANSFER", amount: 1, nameOrig: "C1", nameDest: "C2", oldbalanceOrg: 100, oldbalanceDest: 0 }));
  });
  it("model page: curves, report and importance appear only when the artifact has them", async () => {
    const withEval = { ...preBalanceModel, model: { ...preBalanceModel.model, metrics: { ...preBalanceModel.model.metrics, test_curves: { pr: { precision: [1, 0.5], recall: [0, 1] }, roc: { fpr: [0, 1], tpr: [0, 1] }, prevalence: 0.1 }, feature_importance: { type: "gain", features: ["amount"], values: [1] } } } };
    withApi({ model: vi.fn().mockResolvedValue(withEval) }, <Model />);
    expect(await screen.findByRole("img", { name: /precision-recall curve/i })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /top feature importances/i })).toBeInTheDocument();
  });
  it("model page: no model -> no metrics shown", async () => {
    withApi({ model: vi.fn().mockResolvedValue({ state: "unavailable", reason: "none", model: null }) }, <Model />);
    expect(await screen.findByText(/no trained model is loaded/i)).toBeInTheDocument();
    expect(screen.queryByText(/PR-AUC/)).not.toBeInTheDocument();
  });
  it("monitoring: insufficient data shows no drift numbers", async () => {
    const report = { status: "insufficient_data", model_state: "ready", generated_utc: "2026-01-01T00:00:00+00:00", retraining_note: "A drift alert does not retrain.", drift_alert: false, alert_reasons: [], volume: { total_transactions: 3, flagged_transactions: 0, flag_rate: 0 }, data_quality: { rows: 3, unknown_type_rows: 0, unknown_type_rate: 0, rejected_requests_by_code: {}, rejected_requests_total: 0, note: "n" }, drift: { status: "insufficient_data", reason: "Need at least 100 scored transactions; have 3.", rows_compared: 3, min_rows: 100, threshold_note: "t", thresholds: { warning: 0.1, alert: 0.25 } }, performance: { status: "insufficient_labels", labelled: 0, warning: "biased", reason: "Need labels" } };
    withApi({ monitoring: vi.fn().mockResolvedValue(report), retrainList: vi.fn().mockResolvedValue({ items: [], note: "n" }) }, <Monitoring />);
    expect(await screen.findByText(/insufficient data for drift analysis/i)).toBeInTheDocument();
    expect(screen.queryByText(/psi/i, { selector: ".psi *" })).not.toBeInTheDocument();
    expect(await screen.findByText(/no retraining requests/i)).toBeInTheDocument();
  });
});
