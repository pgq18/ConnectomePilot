"""Download the official MaleCNS tables, or import fly.ai's prepared arrays."""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.download import download, resolve_proxy, proxy_label, check_network, DownloadError
BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {"annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
         "transmitters": "body-neurotransmitters-male-cns-v1.0.feather",
         "connections": "connectome-weights-male-cns-v1.0-minconf-0.5.feather"}


def validate_feather(path):
    import pyarrow as pa
    try:
        with pa.memory_map(str(path), "r") as source:
            pa.ipc.open_file(source)
    except Exception as exc:
        raise DownloadError(f"{path.name} 不是完整有效的 Feather V2 文件。") from exc


def build_official(destination, *, proxy=None, attempts=4):
    import numpy as np
    import pyarrow.feather as feather
    from scipy import sparse
    raw = destination/"raw"
    raw.mkdir(exist_ok=True)
    for name in FILES.values():
        download(BUCKET+name, raw/name, proxy=proxy, attempts=attempts, validator=validate_feather)
    print("数据文件已校验，正在整理神经元注释和递质符号……", flush=True)
    annotations = feather.read_table(raw/FILES["annotations"]).to_pylist()
    by_id = {int(row["bodyId"]): row for row in annotations if row.get("superclass")}
    ids = np.asarray(sorted(by_id), dtype=np.int64)
    if len(ids) == 0:
        raise ValueError("No annotated neurons found")
    rows = [by_id[int(i)] for i in ids]
    del annotations, by_id
    nt_rows = feather.read_table(raw/FILES["transmitters"], columns=["body", "consensus_nt"]).to_pylist()
    nt = {int(row["body"]): str(row["consensus_nt"] or "unclear").lower() for row in nt_rows}
    signs = np.asarray([-1 if any(x in nt.get(int(i), "") for x in ("gaba", "glut", "hist")) else 1 for i in ids], dtype=np.float32)
    del nt, nt_rows
    cell_types = np.asarray([r.get("flywireType") or r.get("type") or "" for r in rows])
    sides = np.asarray([str(r.get("somaSide") or r.get("rootSide") or "").upper() for r in rows])
    # Use an explicit instance suffix only where side annotations are absent.
    for i, row in enumerate(rows):
        if sides[i] not in ("L", "R"):
            instance = str(row.get("instance") or "").upper()
            if instance.endswith(("_L", "_R")):
                sides[i] = instance[-1]
    superclass = np.asarray([r["superclass"] for r in rows])
    del rows
    print(f"正在为 {len(ids):,} 个神经元构建稀疏连接图……", flush=True)
    table = feather.read_table(raw/FILES["connections"], columns=["body_pre", "body_post", "weight"], memory_map=True)
    pre_parts, post_parts, values = [], [], []
    for batch in table.to_batches(max_chunksize=1000000):
        pre_id = batch.column(0).to_numpy(zero_copy_only=False)
        post_id = batch.column(1).to_numpy(zero_copy_only=False)
        pre = np.minimum(np.searchsorted(ids, pre_id), len(ids)-1)
        post = np.minimum(np.searchsorted(ids, post_id), len(ids)-1)
        keep = (ids[pre] == pre_id) & (ids[post] == post_id)
        pre_parts.append(pre[keep].astype(np.int32))
        post_parts.append(post[keep].astype(np.int32))
        values.append(batch.column(2).to_numpy(zero_copy_only=False)[keep].astype(np.float32))
    del table
    pre, post, value = np.concatenate(pre_parts), np.concatenate(post_parts), np.concatenate(values)
    del pre_parts, post_parts, values
    value *= signs[pre]
    incoming = np.bincount(post, weights=np.abs(value), minlength=len(ids)).astype(np.float32)
    value /= np.maximum(incoming[post], 1)
    weights = sparse.csr_matrix((value, (post, pre)), shape=(len(ids),len(ids)), dtype=np.float32)
    sparse.save_npz(destination/"weights.npz", weights, compressed=False)
    np.savez(destination/"brain.npz", ids=ids, cell_type=cell_types, side=sides, superclass=superclass)
    print(f"Prepared {len(ids):,} neurons / {weights.nnz:,} edges", flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-dir", type=Path, help="Import existing fly.ai brain.npz and weights.npz")
    parser.add_argument("--source", choices=["official", "fly-ai"], default="official")
    network = parser.add_mutually_exclusive_group()
    network.add_argument("--proxy", help="Explicit HTTP proxy, e.g. http://127.0.0.1:10808")
    network.add_argument("--no-proxy", action="store_true", help="Connect directly, ignoring proxy configuration")
    parser.add_argument("--check-network", action="store_true", help="Read only 8 bytes to check the network")
    parser.add_argument("--attempts", type=int, choices=range(1, 11), default=4)
    args = parser.parse_args()
    proxy = resolve_proxy(args.proxy, args.no_proxy)
    if not args.from_dir and (args.source == "official" or args.check_network):
        print(f"下载代理：{proxy_label(proxy)}", flush=True)
    if args.check_network:
        check_network(BUCKET+FILES["annotations"], proxy)
        return
    if args.source == "fly-ai" and not args.from_dir and (args.proxy or args.no_proxy):
        parser.error("代理参数仅用于官方数据下载；请使用默认 --source official。")
    destination = ROOT/"data"/"male-cns"
    destination.mkdir(parents=True, exist_ok=True)
    if args.from_dir:
        for name in ("brain.npz", "weights.npz"):
            source = args.from_dir/name
            if not source.is_file():
                raise FileNotFoundError(source)
        for name in ("brain.npz", "weights.npz", "brain.json"):
            source = args.from_dir/name
            if source.is_file() and source.resolve() != (destination/name).resolve():
                shutil.copy2(source, destination/name)
    elif args.source == "fly-ai":
        print("fly.ai 的上游下载器仅使用其自身网络配置；本项目代理配置用于默认的 official 下载方式。", flush=True)
        try:
            from flybrain.data import ensure_data
        except ImportError as exc:
            raise SystemExit("请先在本项目 Conda 环境安装 flybrain==0.1.0，或使用 --from-dir 导入现有数据。") from exc
        ensure_data(destination)
    else:
        build_official(destination, proxy=proxy, attempts=args.attempts)
    from flylab.brain import FlyBrain
    brain = FlyBrain(destination)
    manifest = {"dataset": "MaleCNS v1.0", "license": "CC-BY-4.0",
                "dataset_url": "https://male-cns.janelia.org/download/",
                "processed_by": "FruitFlyLab; compatible with fly.ai's count/sign/normalization recipe" if args.source == "official" and not args.from_dir else "External prepared arrays (imported or fly.ai)",
                "imported_from": str(args.from_dir) if args.from_dir else None,
                "upstream_reference": "https://github.com/alextitonis/fly.ai",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "neurons": brain.n, "edges": brain.weights.nnz,
                "files": {name: digest(destination/name) for name in ("brain.npz", "weights.npz")},
                "transformations": "Annotated neurons only; synapse counts, predicted transmitter signs, incoming-weight normalization. Unknown signs assigned positive.",
                "hash_note": "Local SHA256 identifies the downloaded artifacts; not an independently verified upstream checksum."}
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except DownloadError as exc:
        print(f"\n模型准备未完成：{exc}", file=sys.stderr)
        print("请保持代理开启，检查 config/network.local.json 后重试。无需重装 Conda 或 Python。", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\n下载已中断，已完成的文件和续传信息会保留。", file=sys.stderr)
        raise SystemExit(130)
