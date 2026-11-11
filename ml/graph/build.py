"""Account-transfer graph for PaySim.

Only the supported relationship is sender (nameOrig) -> receiver (nameDest).
No device, card, IP or merchant-attribute relationships are inferred.

GraphSAGE is an optional CPU node-classification experiment. Its graph edges
and node features are built exclusively from the chronological training split.
Validation is used for model selection; test is evaluated only after selection.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from ml.pipeline.evaluate import ranking_metrics
from ml.pipeline.schema import TARGET
from ml.pipeline.split import time_split


REQUIRED = ("nameOrig", "nameDest", "step")
UNSUPPORTED_IDENTIFIER_HINTS = (
    "deviceId", "device_id", "cardId", "card_id", "ipAddress", "ip_address"
)


class GraphInputError(ValueError):
    """Raised when the input cannot be used for account-graph analysis."""


def graph_status() -> dict:
    def has(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            return False

    return {
        "torch_installed": has("torch"),
        "torch_geometric_installed": has("torch_geometric"),
        "gnn_training": (
            "available as optional CPU PyG training; "
            "requires torch and torch_geometric"
        ),
        "core_demo_requires_graph": False,
    }


def validate_graph_input(df: pd.DataFrame) -> dict:
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise GraphInputError(
            f"Missing columns for graph analysis: {missing}"
        )

    if len(df) == 0:
        raise GraphInputError("No rows.")

    if df[list(REQUIRED)].isna().any().any():
        raise GraphInputError("Null identifiers/steps present.")

    return {
        "identifier_columns_used": ["nameOrig", "nameDest"],
        "unsupported_identifier_columns_present_but_unused": [
            column
            for column in UNSUPPORTED_IDENTIFIER_HINTS
            if column in df.columns
        ],
        "device_card_relationships": (
            "not available in the supported schema; none are constructed"
        ),
    }


def _components(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    """Return weakly connected component IDs using union-find."""
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(src.tolist(), dst.tolist()):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    return np.array([find(i) for i in range(n)])


def graph_summary(df: pd.DataFrame) -> dict:
    facts = validate_graph_input(df)

    nodes, inverse = np.unique(
        np.concatenate([
            df["nameOrig"].astype(str).to_numpy(),
            df["nameDest"].astype(str).to_numpy(),
        ]),
        return_inverse=True,
    )

    source = inverse[: len(df)]
    destination = inverse[len(df):]

    edges = pd.DataFrame({
        "source": source,
        "destination": destination,
    }).drop_duplicates()

    components = _components(
        edges["source"].to_numpy(),
        edges["destination"].to_numpy(),
        len(nodes),
    )
    component_sizes = np.bincount(
        np.unique(components, return_inverse=True)[1]
    )

    out_degree = edges.groupby("source").size()
    in_degree = edges.groupby("destination").size()
    sender_counts = df.groupby("nameOrig").size()

    return {
        **facts,
        "transactions": int(len(df)),
        "nodes": int(len(nodes)),
        "distinct_edges": int(len(edges)),
        "senders": int(sender_counts.size),
        "receivers": int(df["nameDest"].nunique()),
        "sender_repeat_fraction": float((sender_counts > 1).mean()),
        "merchant_receiver_fraction": float(
            df["nameDest"].astype(str).str.startswith("M").mean()
        ),
        "max_out_degree": int(out_degree.max()),
        "max_in_degree": int(in_degree.max()),
        "weak_components": int(len(component_sizes)),
        "largest_component_nodes": int(component_sizes.max()),
        "caveat": (
            "Structure only. Sparse graphs (few repeated accounts) limit "
            "what any graph model can learn."
        ),
    }


def dest_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate receiver history using strictly earlier time steps.

    Features:
      - dest_prev_txn_count
      - dest_prev_unique_senders

    Transactions from the same step are not treated as earlier transactions.
    """
    validate_graph_input(df)

    data = pd.DataFrame({
        "nameOrig": df["nameOrig"].astype(str).to_numpy(),
        "nameDest": df["nameDest"].astype(str).to_numpy(),
        "step": df["step"].to_numpy(),
    }, index=df.index)

    per_step = (
        data.groupby(["nameDest", "step"])
        .size()
        .rename("transaction_count")
        .reset_index()
        .sort_values(["nameDest", "step"])
    )

    per_step["previous_count"] = (
        per_step.groupby("nameDest")["transaction_count"].cumsum()
        - per_step["transaction_count"]
    )

    first_seen = (
        data.groupby(["nameDest", "nameOrig"])["step"]
        .min()
        .rename("first_step")
        .reset_index()
    )

    first_steps_by_receiver = {
        receiver: np.sort(group["first_step"].to_numpy())
        for receiver, group in first_seen.groupby("nameDest")
    }

    merged = data[["nameDest", "step"]].merge(
        per_step[["nameDest", "step", "previous_count"]],
        on=["nameDest", "step"],
        how="left",
        sort=False,
    )

    unique_previous_senders = np.array([
        int(np.searchsorted(
            first_steps_by_receiver[receiver],
            step,
            side="left",
        ))
        for receiver, step in zip(merged["nameDest"], merged["step"])
    ])

    return pd.DataFrame(
        {
            "dest_prev_txn_count": (
                merged["previous_count"].to_numpy(dtype=float)
            ),
            "dest_prev_unique_senders": (
                unique_previous_senders.astype(float)
            ),
        },
        index=df.index,
    )


def evaluate_graph_baseline(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict:
    """Evaluate causal receiver-history features on the held-out test range."""
    if TARGET not in df.columns:
        raise GraphInputError(f"'{TARGET}' column required for evaluation")

    ordered = df.reset_index(drop=True)
    features = dest_graph_features(ordered)

    _, _, test, boundaries = time_split(
        ordered,
        train_frac,
        val_frac,
    )

    labels = test[TARGET].to_numpy()

    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {
            "status": "insufficient_data",
            "reason": "Test range lacks both classes.",
            "split_steps": boundaries,
        }

    results = {
        column: ranking_metrics(
            labels,
            features.loc[test.index, column].to_numpy(),
        )
        for column in features.columns
    }

    return {
        "status": "ok",
        "split_steps": boundaries,
        "test_rows": int(len(test)),
        "features": results,
        "combination_with_xgboost": (
            "not validated; evaluate any blend on the validation range, "
            "then report once on test"
        ),
    }


def _transaction_node_scores(
    split_df: pd.DataFrame,
    account_to_id: dict[str, int],
    node_scores: np.ndarray,
    fallback_score: float,
) -> tuple[np.ndarray, dict]:
    """Score every transaction, including transactions with unseen accounts.

    A transaction score is the maximum score of its sender and receiver.
    If neither account appeared in the training graph, use the training
    account fraud prior rather than silently dropping that transaction.
    """
    scores = []
    unseen_rows = 0
    one_seen_rows = 0

    for sender, receiver in zip(
        split_df["nameOrig"].astype(str),
        split_df["nameDest"].astype(str),
    ):
        sender_id = account_to_id.get(sender)
        receiver_id = account_to_id.get(receiver)

        account_scores = []

        if sender_id is not None:
            account_scores.append(float(node_scores[sender_id]))

        if receiver_id is not None:
            account_scores.append(float(node_scores[receiver_id]))

        if not account_scores:
            scores.append(float(fallback_score))
            unseen_rows += 1
        else:
            scores.append(max(account_scores))
            if len(account_scores) == 1:
                one_seen_rows += 1

    return np.asarray(scores, dtype=float), {
        "rows": int(len(split_df)),
        "rows_with_no_training_graph_account": int(unseen_rows),
        "rows_with_only_one_known_account": int(one_seen_rows),
    }


def _evaluate_transaction_scores(
    split_df: pd.DataFrame,
    scores: np.ndarray,
    fallback_details: dict,
) -> dict:
    labels = split_df[TARGET].to_numpy(dtype=int)

    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {
            "status": "insufficient_data",
            "rows": int(len(labels)),
            "account_coverage": fallback_details,
            "reason": "Split lacks both classes.",
        }

    return {
        "status": "ok",
        **ranking_metrics(labels, scores),
        "account_coverage": fallback_details,
    }


def train_gnn(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    epochs: int = 100,
    hidden_dim: int = 32,
    seed: int = 42,
    out_path: Path | None = None,
    patience: int = 15,
) -> dict:
    """Train an optional CPU GraphSAGE account-node classifier.

    Leakage controls:
      1. Graph edges are created only from training-period transactions.
      2. Node features use training-period transaction counts only.
      3. Fraud labels are used only as node targets, never as input features.
      4. Validation selects the best epoch.
      5. Test data is evaluated only after model selection.

    Account labels indicate whether an account participated in at least one
    fraudulent training transaction. Transaction scores are the maximum of
    the sender and receiver account scores. This is an experimental account
    risk signal, not a direct transaction-level GraphSAGE classifier.
    """
    try:
        import torch
        import torch.nn.functional as torch_functional
        from torch import nn
        from torch_geometric.nn import SAGEConv
        from sklearn.metrics import average_precision_score
    except ImportError as exc:
        raise GraphInputError(
            "GraphSAGE training requires torch, torch_geometric and scikit-learn "
            "in the active environment."
        ) from exc

    validate_graph_input(df)

    if TARGET not in df.columns:
        raise GraphInputError(f"'{TARGET}' column required for training")

    if epochs < 1:
        raise GraphInputError("epochs must be at least 1.")

    if hidden_dim < 1:
        raise GraphInputError("hidden_dim must be at least 1.")

    if patience < 1:
        raise GraphInputError("patience must be at least 1.")

    if not 0 < train_frac < 1:
        raise GraphInputError("train_frac must be between 0 and 1.")

    if not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise GraphInputError(
            "val_frac must be positive and train_frac + val_frac must be < 1."
        )

    ordered = df.reset_index(drop=True).copy()

    train_df, validation_df, test_df, boundaries = time_split(
        ordered,
        train_frac,
        val_frac,
    )

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_labels = train_df[TARGET].to_numpy(dtype=int)

    if len(np.unique(train_labels)) < 2:
        return {
            "status": "insufficient_data",
            "reason": "Training transactions lack both classes.",
            "split_steps": boundaries,
        }

    # Only training-period accounts become nodes.
    accounts = np.unique(np.concatenate([
        train_df["nameOrig"].astype(str).to_numpy(),
        train_df["nameDest"].astype(str).to_numpy(),
    ]))

    account_to_id = {
        account: index
        for index, account in enumerate(accounts)
    }

    sender_ids = (
        train_df["nameOrig"].astype(str)
        .map(account_to_id)
        .to_numpy(dtype=np.int64)
    )
    receiver_ids = (
        train_df["nameDest"].astype(str)
        .map(account_to_id)
        .to_numpy(dtype=np.int64)
    )

    edge_index_np = np.unique(
        np.stack([sender_ids, receiver_ids], axis=0),
        axis=1,
    )

    if edge_index_np.shape[1] == 0:
        return {
            "status": "insufficient_data",
            "reason": "Training graph has no edges.",
            "split_steps": boundaries,
        }

    # Input features are structural counts only. No TARGET-derived feature
    # is permitted here.
    out_degree = np.bincount(
        sender_ids,
        minlength=len(accounts),
    ).astype(np.float32)

    in_degree = np.bincount(
        receiver_ids,
        minlength=len(accounts),
    ).astype(np.float32)

    # Log transform reduces scale differences between high- and low-degree
    # accounts while preserving the ordering of counts.
    node_features = np.stack([
        np.log1p(out_degree),
        np.log1p(in_degree),
    ], axis=1).astype(np.float32)

    # Node targets are constructed only from training transactions.
    node_labels = torch.full(
        (len(accounts),),
        -1,
        dtype=torch.long,
    )

    account_fraud = np.zeros(len(accounts), dtype=np.int8)

    for sender, receiver, label in zip(
        sender_ids,
        receiver_ids,
        train_labels,
    ):
        if int(label) == 1:
            account_fraud[sender] = 1
            account_fraud[receiver] = 1

    node_labels = torch.tensor(
        account_fraud,
        dtype=torch.long,
    )

    train_mask = node_labels >= 0

    positive_count = int((node_labels[train_mask] == 1).sum())
    negative_count = int((node_labels[train_mask] == 0).sum())

    if positive_count == 0 or negative_count == 0:
        return {
            "status": "insufficient_data",
            "reason": "Training graph nodes lack both classes.",
            "split_steps": boundaries,
        }

    # The training prior is used only as a fallback for accounts absent
    # from the training graph.
    training_fraud_prior = float(train_labels.mean())

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Explicit CPU execution: no NVIDIA GPU is required.
    device = torch.device("cpu")

    x = torch.tensor(
        node_features,
        dtype=torch.float32,
        device=device,
    )
    edge_index = torch.tensor(
        edge_index_np,
        dtype=torch.long,
        device=device,
    )
    node_labels = node_labels.to(device)
    train_mask = train_mask.to(device)

    class_weights = torch.tensor(
        [
            1.0,
            float(negative_count / max(positive_count, 1)),
        ],
        dtype=torch.float32,
        device=device,
    )

    class GraphSAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(2, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, 2)

        def forward(self, features, edges):
            hidden = self.conv1(features, edges).relu()
            return self.conv2(hidden, edges)

    model = GraphSAGE().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=1e-4,
    )

    def get_node_scores() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            return (
                logits.softmax(dim=1)[:, 1]
                .detach()
                .cpu()
                .numpy()
            )

    def score_split(split_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
        current_node_scores = get_node_scores()
        return _transaction_node_scores(
            split_df,
            account_to_id,
            current_node_scores,
            training_fraud_prior,
        )

    def validation_pr_auc() -> tuple[float | None, dict]:
        scores, coverage = score_split(validation_df)
        labels = validation_df[TARGET].to_numpy(dtype=int)

        if len(labels) == 0 or len(np.unique(labels)) < 2:
            return None, coverage

        return float(average_precision_score(labels, scores)), coverage

    best_state = None
    best_validation_pr_auc = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    actual_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index)

        loss = torch_functional.cross_entropy(
            logits[train_mask],
            node_labels[train_mask],
            weight=class_weights,
        )

        loss.backward()
        optimizer.step()

        actual_epochs = epoch

        current_pr_auc, _ = validation_pr_auc()

        if current_pr_auc is None:
            # Validation lacks both classes; train for the requested
            # number of epochs and clearly report the limitation.
            continue

        if current_pr_auc > best_validation_pr_auc:
            best_validation_pr_auc = current_pr_auc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate validation and test with the selected model.
    validation_scores, validation_coverage = score_split(validation_df)
    validation_metrics = _evaluate_transaction_scores(
        validation_df,
        validation_scores,
        validation_coverage,
    )

    test_scores, test_coverage = score_split(test_df)
    test_metrics = _evaluate_transaction_scores(
        test_df,
        test_scores,
        test_coverage,
    )

    model_file = None

    if out_path is not None:
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "accounts": accounts.tolist(),
                "feature_names": [
                    "log1p_out_degree",
                    "log1p_in_degree",
                ],
                "hidden_dim": int(hidden_dim),
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "training_fraud_prior": training_fraud_prior,
                "model_type": "experimental_account_node_graphsage",
            },
            output_path,
        )
        model_file = str(output_path)

    return {
        "status": "ok",
        "model_type": "experimental_account_node_graphsage",
        "device": "cpu",
        "split_steps": boundaries,
        "nodes": int(len(accounts)),
        "edges": int(edge_index_np.shape[1]),
        "input_features": [
            "log1p_out_degree",
            "log1p_in_degree",
        ],
        "target_leakage_features": [],
        "training_account_fraud_prior": training_fraud_prior,
        "requested_epochs": int(epochs),
        "actual_epochs": int(actual_epochs),
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_validation_pr_auc": (
            float(best_validation_pr_auc)
            if np.isfinite(best_validation_pr_auc)
            else None
        ),
        "early_stopping_patience": int(patience),
        "validation": validation_metrics,
        "test": test_metrics,
        "model_file": model_file,
        "caveat": (
            "Experimental account-node risk model, not a direct "
            "transaction-level classifier. Account labels are derived only "
            "from training-period transactions. Transaction scores use the "
            "maximum sender/receiver node score; unseen accounts use the "
            "training fraud prior. Do not blend with XGBoost without "
            "validation-based evaluation."
        ),
    }