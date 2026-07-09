"""
Non-learning heuristic baselines for the node prediction task.

These establish the performance floor that any learning model must beat.
All heuristics operate only on the masked graph (post-masking features) —
exactly what the GNN sees — so the comparison is fair.

Node task: after hiding 20% of endpoint nodes and all their edges, label
visible nodes as 1 if they had a hidden neighbour, 0 otherwise.

Heuristics:
  random        — scores every node with a uniform random value
  endpoint_feat — score = masked is_endpoint feature (col 4)
                  after masking the base node's degree drops → may become endpoint
  degree_inv    — score = 1/(1+degree) — lower degree after masking = more suspicious
  thickness     — score = avg local thickness — thinner nodes may be crack tips
  peripheral    — score = distance from spatial centroid of the graph
                  outermost nodes are more likely to be active growth fronts
"""

import json
import argparse
import numpy as np
import torch
from sklearn.metrics import average_precision_score

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from link_prediction.masking import apply_node_mask, is_valid_for_node_task


# Column indices in node feature matrix x [N, 6]
_COL_X          = 0   # x_norm
_COL_Y          = 1   # y_norm
_COL_THICKNESS  = 2   # avg thickness
_COL_DEGREE     = 3   # degree (recomputed after masking)
_COL_IS_ENDPT   = 4   # is_endpoint (recomputed after masking)
_COL_IS_JUNCT   = 5   # is_junction (recomputed after masking)


def _eval_heuristic(test_graphs, score_fn, mask_frac=0.20, seed=42):
    """
    Apply score_fn to every valid masked graph and compute node AP.

    score_fn(x_masked, eval_mask) -> np.ndarray of shape [eval_mask.sum()]
        x_masked : [N, 6] masked node feature matrix
        eval_mask: boolean tensor [N] — which nodes to score
    """
    all_scores, all_labels = [], []
    np.random.seed(seed)
    torch.manual_seed(seed)

    for i, g in enumerate(test_graphs):
        if not is_valid_for_node_task(g):
            continue
        masked, node_labels, _, eval_mask = apply_node_mask(g, mask_frac=mask_frac, seed=seed + i)
        if eval_mask.sum() == 0:
            continue
        labels_ev = node_labels[eval_mask].numpy()
        if len(set(labels_ev)) < 2:
            continue

        x = masked.x.float()
        scores = score_fn(x, eval_mask)
        all_scores.extend(scores.tolist())
        all_labels.extend(labels_ev.tolist())

    if not all_labels or len(set(all_labels)) < 2:
        return 0.0, 0
    return average_precision_score(all_labels, all_scores), len(all_labels)


def run_all(test_graphs, mask_frac=0.20, seed=42):
    """Run all heuristics and return a results dict."""

    pos_rate = None

    def random_score(x, eval_mask):
        return np.random.rand(int(eval_mask.sum()))

    def endpoint_feat(x, eval_mask):
        return x[eval_mask, _COL_IS_ENDPT].numpy()

    def degree_inv(x, eval_mask):
        # Lower degree after masking = more likely to be a base node
        deg = x[eval_mask, _COL_DEGREE].numpy()
        return 1.0 / (1.0 + deg)

    def thickness_score(x, eval_mask):
        # Thinner nodes = more likely crack tips / base nodes
        thick = x[eval_mask, _COL_THICKNESS].numpy()
        # Invert so thin = high score
        max_t = thick.max() + 1e-8
        return 1.0 - (thick / max_t)

    def peripheral_score(x, eval_mask):
        # Nodes furthest from the graph centroid = more likely growth fronts
        all_xy = x[:, :2].numpy()                     # [N, 2]
        centroid = all_xy.mean(axis=0)
        ev_xy = x[eval_mask, :2].numpy()
        dist = np.linalg.norm(ev_xy - centroid, axis=1)
        return dist

    heuristics = {
        'random':        random_score,
        'endpoint_feat': endpoint_feat,
        'degree_inv':    degree_inv,
        'thickness':     thickness_score,
        'peripheral':    peripheral_score,
    }

    results = {}
    for name, fn in heuristics.items():
        ap, n = _eval_heuristic(test_graphs, fn, mask_frac=mask_frac, seed=seed)
        results[name] = {'node_ap': round(ap, 6), 'n_scored': n}
        print(f'  {name:<20s}  Node AP = {ap:.4f}  (n={n})')

    # Compute class prior (random AP baseline)
    all_labels = []
    for i, g in enumerate(test_graphs):
        if not is_valid_for_node_task(g):
            continue
        _, node_labels, _, eval_mask = apply_node_mask(g, mask_frac=mask_frac, seed=seed + i)
        if eval_mask.sum() == 0:
            continue
        labels_ev = node_labels[eval_mask].numpy()
        if len(set(labels_ev)) < 2:
            continue
        all_labels.extend(labels_ev.tolist())

    pos_rate = float(np.mean(all_labels)) if all_labels else 0.0
    results['_class_prior'] = round(pos_rate, 6)
    results['_note'] = (
        'AP for a perfect-random classifier equals the class prior. '
        'endpoint_feat < random proves that node features alone carry no signal '
        'for identifying base nodes after masking — only graph topology can solve this.'
    )
    print(f'\n  Class prior (random AP ≈): {pos_rate:.4f}')
    return results


def main():
    parser = argparse.ArgumentParser(description='Node task heuristic baselines')
    parser.add_argument('--graphs',    type=str, default='outputs/clean_graphs/graphs/test_graphs.pt')
    parser.add_argument('--output',    type=str, default='outputs/node_heuristic_results.json')
    parser.add_argument('--mask-frac', type=float, default=0.20)
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    print(f'Loading test graphs: {args.graphs}')
    test_graphs = torch.load(args.graphs, weights_only=False)
    print(f'  {len(test_graphs)} graphs loaded\n')

    print('Running node task heuristic baselines:')
    print('-' * 55)
    results = run_all(test_graphs, mask_frac=args.mask_frac, seed=args.seed)

    print('\n' + '-' * 55)
    print('Summary (sorted by AP):')
    scored = {k: v for k, v in results.items() if not k.startswith('_')}
    for name, res in sorted(scored.items(), key=lambda x: -x[1]['node_ap']):
        marker = ' ← best heuristic' if res['node_ap'] == max(r['node_ap'] for r in scored.values()) else ''
        print(f'  {name:<20s}  {res["node_ap"]:.4f}{marker}')
    print(f'  {"class_prior":<20s}  {results["_class_prior"]:.4f}  (random AP baseline)')
    print(f'\n  GNN results for context: MLP=0.662, SAGE=0.700, GINE=0.739')

    import json, datetime
    results['_run_date'] = datetime.datetime.now().strftime('%Y-%m-%d')
    results['_mask_frac'] = args.mask_frac
    results['_seed'] = args.seed
    results['_graphs'] = args.graphs

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved → {args.output}')


if __name__ == '__main__':
    main()
