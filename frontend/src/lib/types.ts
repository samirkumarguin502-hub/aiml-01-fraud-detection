// Types mirror the FastAPI responses documented in docs/api.md. They are checked at runtime by guards.ts.
export interface Page<T> { items: T[]; total: number; page: number; page_size: number; note?: string }

export interface TransactionRow {
  id: number; created_utc: string; source: "single" | "batch"; batch_id: string | null;
  step: number; type: string; amount: number; name_orig: string; name_dest: string;
  score: number; threshold: number; flagged: boolean; type_in_training_vocab: boolean;
  model_kind: string; model_sha256: string; model_data_label: string | null;
  case_status: string | null; case_note?: string | null; feedback_label: number | null;
  case_history?: { from_status: string | null; to_status: string; created_utc: string }[];
}

export interface Prediction {
  transaction_id: number; score: number; threshold: number; flagged: boolean; score_note: string; warnings: string[];
  model: { kind: string; sha256: string; metrics_origin: string | null };
}

export interface BatchResult {
  batch_id: string; rows: number; flagged: number; score_note: string; unknown_type_rows: number;
  results: { row: number; transaction_id: number; score: number; flagged: boolean }[];
}

export interface Health { status: string }
export interface Ready { ready: boolean; checks: { database: string; model: string }; model_reason?: string | null }

export interface ThresholdRow { threshold: number; tp: number; fp: number; fn: number; tn: number; precision: number; recall: number; f1: number; flagged_rate: number }
export interface SetMetrics { pr_auc: number | null; roc_auc: number | null; precision: number; recall: number; f1: number; tp: number; fp: number; fn: number; tn: number; n: number; n_fraud: number; prevalence: number | null; threshold: number }
export interface ClassMetrics { precision: number; recall: number; "f1-score": number; support: number }
export interface ClassReport { threshold: number; not_fraud: ClassMetrics; fraud: ClassMetrics; "macro avg": ClassMetrics; "weighted avg": ClassMetrics }
export interface Curves { pr: { precision: number[]; recall: number[] }; roc: { fpr: number[]; tpr: number[] }; prevalence: number }
export interface Importance { type: string; features: string[]; values: number[] }
export interface ModelInfo {
  state: "ready" | "unavailable" | "invalid"; reason: string | null;
  model: null | {
    kind: string; sha256: string; created_utc: string | null; threshold: { threshold: number; policy: string; fallback_to_max_f1: boolean };
    feature_version: string; feature_count: number; type_vocabulary: string[]; score_semantics: string | null;
    feature_set?: string; input_fields?: string[]; model_version?: string | null;
    training?: { fit_seconds?: number; peak_memory_mb?: number | null; cpu_count?: number; python?: string } | null;
    metrics: { origin: string; metrics_are_synthetic: boolean; dataset_sha256: string | null; validation?: SetMetrics; test?: SetMetrics; test_threshold_table?: ThresholdRow[];
      test_classification_report?: ClassReport | null; test_curves?: Curves | null; feature_importance?: Importance | null };
  };
}

export interface Summary {
  empty: boolean; total_transactions: number; flagged_transactions: number;
  score_histogram: { bin_start: number; bin_end: number; count: number }[];
  by_day: { day: string; transactions: number; flagged: number }[];
  by_type: { type: string; transactions: number; flagged: number }[];
  case_status_counts: Record<string, number>;
  feedback: { total: number; labelled_fraud: number; labelled_not_fraud: number };
  model: { state: string; reason: string | null; kind: string | null; metrics_origin: string | null };
  notes: string[];
}

export interface DriftFeature { feature: string; psi: number; level: "stable" | "warning" | "alert" }
export interface MonitoringReport {
  status: "ok" | "no_model" | "insufficient_data" | "no_reference"; reason?: string; model_state: string;
  generated_utc: string; retraining_note: string; drift_alert: boolean; alert_reasons: string[]; retraining_recommended?: boolean;
  volume: { total_transactions: number; flagged_transactions: number; flag_rate: number | null };
  data_quality: { rows: number; unknown_type_rows: number; unknown_type_rate: number | null; rejected_requests_by_code: Record<string, number>; rejected_requests_total: number; note: string };
  drift: null | { status: string; reason?: string; rows_compared: number; min_rows: number; threshold_note: string; thresholds: { warning: number; alert: number };
    features?: DriftFeature[]; score?: { psi: number; level: "stable" | "warning" | "alert"; reference: string } };
  performance: null | { status: "ok" | "insufficient_labels"; labelled: number; reason?: string; warning: string; precision?: number; recall?: number; f1?: number; tp?: number; fp?: number; fn?: number; tn?: number };
}

export interface PipelineStatus {
  optional_components: Record<string, { library_installed: boolean }>; note: string;
  last_replay_summary: null | { mode?: string; started_utc?: string; finished?: boolean; processed?: number; outcomes?: Record<string, number>; note?: string; error?: string };
  model_state: string; database: string;
}

export interface RetrainRequest {
  id: number; created_utc: string; updated_utc: string; status: string; reason: string; note: string | null; candidate: string | null;
  evaluation: null | { passed: boolean; rule: string; candidate_test_pr_auc: number | null; active_test_pr_auc: number | null; candidate_origin: string | null; active_origin: string | null; caveat: string };
  events: { from_status: string | null; to_status: string; note: string | null; created_utc: string }[]; allowed_next: string[];
}
