"""Extract a reproducible real subgraph, retaining paths from every input to DN."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.population import input_map


def main():
    out = ROOT / 'data/plasticity'
    if (out / 'graph.npz').exists():
        raise SystemExit('Graph already exists; refusing to replace it.')
    out.mkdir(parents=True, exist_ok=True)
    w = sparse.load_npz(ROOT / 'data/male-cns/weights.npz').tocsr()
    meta = np.load(ROOT / 'data/male-cns/brain.npz', allow_pickle=False)
    inputs, channels = input_map(meta['cell_type'], meta['side'], meta['ids'])
    dn = np.flatnonzero(meta['superclass'] == 'descending_neuron')
    n = w.shape[0]
    # Rows of the search graph are sources; stored connectome rows are targets.
    roots = sparse.csr_matrix((np.ones(len(inputs)), (np.zeros(len(inputs)), inputs)), shape=(1, n))
    search = sparse.bmat([[w.T, None], [roots, sparse.csr_matrix((1, 1))]], format='csr')
    _, pred = breadth_first_order(search, n, directed=True)
    required = set(inputs.tolist())
    for node in dn:
        cursor = int(node)
        while cursor != n and cursor not in required:
            required.add(cursor)
            cursor = int(pred[cursor])
            if cursor < 0:
                raise ValueError(f'Unreachable DN: {node}')
    # Fill by two-hop input-to-output path mass, then by index for deterministic ties.
    a = np.asarray(abs(w[:, inputs]).sum(1)).ravel()
    b = np.asarray(abs(w[dn]).sum(0)).ravel()
    order = np.lexsort((np.arange(n), -(a * b)))
    selected = set(required)
    for node in order:
        if len(selected) >= max(4096, len(required)): break
        selected.add(int(node))
    selected = np.array(sorted(selected), dtype=np.int64)
    local = np.full(n, -1, np.int64); local[selected] = np.arange(len(selected))
    sub = w[selected][:, selected].tocsr().astype(np.float32)
    denominator = np.asarray(abs(sub).sum(1)).ravel()
    sub = sparse.diags(1 / np.maximum(denominator, 1e-8)) @ sub
    sub = sub.tocsr(); sub.sort_indices()
    coo = sub.tocoo()
    # Verify all outputs remain reachable after inducing and renormalizing.
    local_roots = sparse.csr_matrix((np.ones(len(inputs)), (np.zeros(len(inputs)), local[inputs])), shape=(1, len(selected)))
    check = sparse.bmat([[sub.T, None], [local_roots, sparse.csr_matrix((1, 1))]], format='csr')
    visited, _ = breadth_first_order(check, len(selected), directed=True)
    assert np.isin(local[dn], visited).all()
    np.savez_compressed(out / 'graph.npz', source=coo.col.astype(np.int64), target=coo.row.astype(np.int64),
                        weight=coo.data.astype(np.float32), inputs=local[inputs], channels=channels,
                        descending=local[dn], global_indices=selected, body_ids=meta['ids'][selected])
    digest = hashlib.sha256((out / 'graph.npz').read_bytes()).hexdigest()
    manifest = {'neurons': len(selected), 'edges': sub.nnz, 'inputs': len(inputs), 'descending': len(dn),
                'required_path_nodes': len(required), 'all_outputs_reachable': True, 'graph_sha256': digest,
                'source_neurons': n, 'source_edges': w.nnz,
                'selection': 'multi-source BFS path union, filled by two-hop absolute path mass',
                'initial_weights': 'signed MaleCNS synapse-count weights, incoming L1 renormalized within subgraph',
                'fixed_signs': 'inherited engineering transmitter signs; not measured synaptic efficacy'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
