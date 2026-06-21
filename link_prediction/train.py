"""
Training loop for Phase 3 link prediction.

Two tasks are trained jointly:
  Edge task  — predict removed edges  (BCEWithLogitsLoss)
  Node task  — predict nodes with hidden neighbours (BCEWithLogitsLoss)

Combined loss = node_loss + edge_loss_weight * edge_loss

Best checkpoint selected by combined val score:
    val_score = 0.5 * node_val_auc + 0.5 * edge_val_auc
"""

import copy
import math
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .model import build_encoder, MLPEdgePredictor, MLPNodePredictor
from .masking import apply_node_mask, is_valid_for_edge_task, is_valid_for_node_task
from .splits import recompute_structural_features, transductive_split


def build_models(in_channels=6, hidden=128, out_dim=64, heads=4, dropout=0.3,
                 model_name='gat'):
    encoder   = build_encoder(model_name, in_channels, hidden, out_dim,
                               heads=heads, dropout=dropout)
    edge_pred = MLPEdgePredictor(out_dim, hidden=hidden, dropout=dropout)
    node_pred = MLPNodePredictor(out_dim, hidden=hidden // 2, dropout=dropout)
    return encoder, edge_pred, node_pred


def _compute_node_pos_weight(dataset, mask_frac: float) -> float:
    """
    Compute the positive class weight for the node task from training data.
    Returns neg_count / pos_count — the true imbalance ratio.
    """
    total_pos = total_neg = 0
    for g in dataset:
        if not is_valid_for_node_task(g):
            continue
        _, labels, _, eval_mask = apply_node_mask(g, mask_frac=mask_frac, seed=0)
        ev = labels[eval_mask]
        total_pos += int((ev == 1).sum().item())
        total_neg += int((ev == 0).sum().item())
    if total_pos == 0:
        return 1.0
    return total_neg / total_pos


def _warmup_cosine(epoch: int, warmup_epochs: int, total_epochs: int, eta_min_ratio: float = 0.01) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return eta_min_ratio + (1.0 - eta_min_ratio) * cosine


def _edge_auc(encoder, edge_pred, splits, device):
    """Val AUC using hard-negative supervision pairs (consistent with final eval)."""
    encoder.eval(); edge_pred.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for train_d, _train_eli, _train_lbl, val_ei, val_labels in splits:
            if val_ei.size(1) == 0:
                continue
            x   = train_d.x.float().to(device)
            ei  = train_d.edge_index.to(device)
            ea  = train_d.edge_attr.float().to(device) if train_d.edge_attr is not None else None
            z   = encoder(x, ei, ea)
            prob = torch.sigmoid(edge_pred(z, val_ei.to(device))).cpu().numpy()
            lbl  = val_labels.numpy()
            if len(np.unique(lbl)) == 2:
                preds_all.extend(prob)
                labels_all.extend(lbl)
    if not labels_all or len(np.unique(labels_all)) < 2:
        return 0.5
    return roc_auc_score(labels_all, preds_all)


def _node_auc(encoder, node_pred, graphs, device, mask_frac=0.20, seed=99):
    encoder.eval(); node_pred.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for i, g in enumerate(graphs):
            if not is_valid_for_node_task(g):
                continue
            masked, node_labels, _, eval_mask = apply_node_mask(g, mask_frac=mask_frac, seed=seed + i)
            if eval_mask.sum() == 0:
                continue
            labels_ev = node_labels[eval_mask]
            if len(torch.unique(labels_ev)) < 2:
                continue
            x  = masked.x.float().to(device)
            ei = masked.edge_index.to(device)
            ea = masked.edge_attr.float().to(device) if masked.edge_attr is not None else None
            z     = encoder(x, ei, ea)
            probs = torch.sigmoid(node_pred(z))[eval_mask].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels_ev.numpy())
    if not all_labels or len(np.unique(all_labels)) < 2:
        return 0.5
    return roc_auc_score(all_labels, all_probs)


def train(
    train_dataset: list,
    epochs: int = 300,
    hidden: int = 128,
    out_dim: int = 64,
    heads: int = 4,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    edge_loss_weight: float = 1.0,
    node_mask_frac: float = 0.20,
    dropout: float = 0.3,
    accum_steps: int = 8,
    warmup_epochs: int = 10,
    patience: int = 20,
    device: torch.device = torch.device('cpu'),
    log_every: int = 10,
    model_name: str = 'gat',
) -> dict:
    """
    Train the encoder + edge predictor + node predictor jointly.

    patience: stop if val_score hasn't improved for this many epochs (0 = disabled).

    Returns dict with trained models and training history.
    """
    # ── Prepare edge splits with hard negatives (done once, reused every epoch) ─
    # Each split: (train_data, train_eli, train_labels, val_ei, val_labels)
    #
    # Correct training protocol (avoids train/val distribution shift):
    #   - 80% edges → message passing (train_data.edge_index)
    #   - 10% edges → HIDDEN training supervision positives  (train split of training graphs)
    #   - 10% edges → HIDDEN val supervision positives       (val split, checkpoint selection)
    #   Hard negatives are used for ALL supervision (train + val), not random negatives.
    #
    # Key insight: training on HIDDEN edges teaches the model to predict missing
    # connections from structural context, consistent with how val/test work.
    # Training on VISIBLE (mp) edges would cause a distribution shift: the model
    # learns "mp neighbors = positive" but val tests "hidden edges = positive".

    edge_splits = []
    for i, g in enumerate(train_dataset):
        if not is_valid_for_edge_task(g):
            continue
        # 80% mp / 10% train-supervision / 10% val-supervision
        s = transductive_split(g, num_val=0.10, num_test=0.10, seed=i)
        if s is None:
            continue

        train_d    = s['train_data']    # x recomputed from 80% mp edges
        # Use the "test" split of each training graph as training supervision:
        # these are HIDDEN edges (not in mp), exactly what we want to predict.
        train_eli    = s['test_ei']     # hidden positives + hard negatives
        train_labels = s['test_labels']
        val_ei       = s['val_ei']      # hidden positives + hard negatives (for AUC)
        val_labels   = s['val_labels']

        if train_eli.size(1) == 0 or val_ei.size(1) == 0:
            continue

        edge_splits.append((train_d, train_eli, train_labels, val_ei, val_labels))

    node_graphs = [g for g in train_dataset if is_valid_for_node_task(g)]
    print(f'  Node task  : {len(node_graphs)} / {len(train_dataset)} graphs usable')
    print(f'  Edge task  : {len(edge_splits)} / {len(train_dataset)} graphs usable')

    in_channels = train_dataset[0].x.size(1)
    encoder, edge_pred, node_pred = build_models(in_channels, hidden, out_dim, heads, dropout,
                                                  model_name=model_name)
    encoder.to(device); edge_pred.to(device); node_pred.to(device)

    params    = list(encoder.parameters()) + list(edge_pred.parameters()) + list(node_pred.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: _warmup_cosine(e, warmup_epochs, epochs),
    )
    pw_full        = _compute_node_pos_weight(train_dataset, node_mask_frac)
    # Cap pos_weight at 5× — the full imbalance ratio (10×) maximises recall but
    # produces too many false positives.  5× balances precision and recall better.
    pw             = min(pw_full, 5.0)
    print(f'  Node pos_weight: {pw:.1f}x  (capped from {pw_full:.1f}x imbalance ratio)')
    criterion      = torch.nn.BCEWithLogitsLoss()
    node_criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], device=device)
    )

    best_val_score   = 0.0
    best_state       = None
    history          = []
    patience_counter = 0

    pbar = tqdm(range(1, epochs + 1), desc='Training', unit='epoch', file=sys.stdout)

    for epoch in pbar:
        encoder.train(); edge_pred.train(); node_pred.train()
        total_edge_loss = total_node_loss = 0.0
        n_edge = n_node = 0

        # ── Task 1: Node prediction ───────────────────────────────────────────
        optimizer.zero_grad()
        accum_count = 0
        for i, g in enumerate(node_graphs):
            masked, node_labels, _, eval_mask = apply_node_mask(g, mask_frac=node_mask_frac)
            if eval_mask.sum() == 0:
                continue
            labels_ev = node_labels[eval_mask]
            if len(torch.unique(labels_ev)) < 2:
                continue

            x   = masked.x.float().to(device)
            ei  = masked.edge_index.to(device)
            ea  = masked.edge_attr.float().to(device) if masked.edge_attr is not None else None
            lbl = node_labels.float().to(device)
            ev  = eval_mask.to(device)

            z     = encoder(x, ei, ea)
            logit = node_pred(z)[ev]
            loss  = node_criterion(logit, lbl[ev])
            (loss / accum_steps).backward()
            total_node_loss += loss.item()
            n_node += 1
            accum_count += 1

            if accum_count % accum_steps == 0 or i == len(node_graphs) - 1:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # ── Task 2: Edge prediction ───────────────────────────────────────────
        optimizer.zero_grad()
        accum_count = 0
        for i, (train_d, train_eli, train_labels, _, _) in enumerate(edge_splits):
            if train_eli.size(1) == 0:
                continue
            x   = train_d.x.float().to(device)
            ei  = train_d.edge_index.to(device)
            ea  = train_d.edge_attr.float().to(device) if train_d.edge_attr is not None else None
            eli = train_eli.to(device)
            lbl = train_labels.float().to(device)

            z     = encoder(x, ei, ea)
            logit = edge_pred(z, eli)
            loss  = criterion(logit, lbl)
            (edge_loss_weight * loss / accum_steps).backward()
            total_edge_loss += loss.item()
            n_edge += 1
            accum_count += 1

            if accum_count % accum_steps == 0 or i == len(edge_splits) - 1:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        # ── Validation: combined node + edge AUC ─────────────────────────────
        node_val_auc = _node_auc(encoder, node_pred, node_graphs, device,
                                 mask_frac=node_mask_frac, seed=99)
        edge_val_auc = _edge_auc(encoder, edge_pred, edge_splits, device)
        val_score    = 0.5 * node_val_auc + 0.5 * edge_val_auc

        avg_el = total_edge_loss / max(n_edge, 1)
        avg_nl = total_node_loss / max(n_node, 1)
        history.append({
            'epoch': epoch,
            'edge_loss': avg_el,
            'node_loss': avg_nl,
            'node_val_auc': node_val_auc,
            'edge_val_auc': edge_val_auc,
            'val_score': val_score,
        })

        is_best = val_score > best_val_score
        if is_best:
            best_val_score   = val_score
            patience_counter = 0
            best_state = {
                'encoder':   copy.deepcopy(encoder.state_dict()),
                'edge_pred': copy.deepcopy(edge_pred.state_dict()),
                'node_pred': copy.deepcopy(node_pred.state_dict()),
                'epoch':     epoch,
            }
        else:
            patience_counter += 1

        pbar.set_postfix({
            'n_loss':  f'{avg_nl:.4f}',
            'e_loss':  f'{avg_el:.4f}',
            'n_auc':   f'{node_val_auc:.4f}',
            'e_auc':   f'{edge_val_auc:.4f}',
            'best':    f'{best_val_score:.4f}',
        })

        if epoch % log_every == 0:
            marker = ' ← best' if is_best else ''
            tqdm.write(
                f'  Epoch {epoch:03d} | n_loss={avg_nl:.4f}  e_loss={avg_el:.4f}'
                f'  n_auc={node_val_auc:.4f}  e_auc={edge_val_auc:.4f}'
                f'  score={val_score:.4f}  best={best_val_score:.4f}{marker}'
            )

        if patience > 0 and patience_counter >= patience:
            tqdm.write(f'  Early stop at epoch {epoch} (no improvement for {patience} epochs)')
            break

    # Restore best checkpoint
    encoder.load_state_dict(best_state['encoder'])
    edge_pred.load_state_dict(best_state['edge_pred'])
    node_pred.load_state_dict(best_state['node_pred'])

    print(f'\n  Best epoch: {best_state["epoch"]}  |  Best val score: {best_val_score:.4f}')

    return {
        'encoder':        encoder,
        'edge_pred':      edge_pred,
        'node_pred':      node_pred,
        'best_val_score': best_val_score,
        'best_epoch':     best_state['epoch'],
        'history':        history,
    }
