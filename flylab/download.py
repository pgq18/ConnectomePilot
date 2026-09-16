"""Verified, resumable downloads with an explicit project-local proxy."""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETWORK_CONFIG = ROOT / "config" / "network.local.json"


class DownloadError(RuntimeError):
    pass


def resolve_proxy(explicit=None, direct=False):
    if direct:
        return ""
    if explicit is not None:
        proxy = explicit
    elif "FLY_LAB_PROXY" in os.environ:
        proxy = os.environ["FLY_LAB_PROXY"]
    elif NETWORK_CONFIG.is_file():
        try:
            proxy = json.loads(NETWORK_CONFIG.read_text()).get("proxy")
        except (OSError, ValueError, AttributeError) as exc:
            raise DownloadError("config/network.local.json 不是有效的代理配置。") from exc
    else:
        proxy = None
    if proxy:
        try:
            parsed = urllib.parse.urlsplit(proxy)
            valid = parsed.scheme in ("http", "https") and parsed.hostname and parsed.port
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise DownloadError("代理地址应为 http://主机:端口；此下载器不支持 SOCKS 代理。")
    elif proxy is not None and proxy != "":
        raise DownloadError("代理配置应为 URL 字符串或 null。")
    return proxy


def proxy_label(proxy):
    if proxy is None:
        return "继承终端 / 系统配置"
    if not proxy:
        return "直接连接（不使用代理）"
    p = urllib.parse.urlsplit(proxy)
    return f"{p.scheme}://{p.hostname}:{p.port}"


def make_opener(proxy=None):
    # None inherits urllib's environment/macOS discovery. Empty string explicitly
    # disables it. Project configuration is independent of Finder's environment.
    handlers = [urllib.request.HTTPSHandler(context=ssl.create_default_context())]
    if proxy is not None:
        handlers.insert(0, urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {}))
    return urllib.request.build_opener(*handlers)


def explain_error(exc):
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "HTTPS 证书校验失败。请检查网络或代理的证书配置；下载器未关闭证书校验。"
    if isinstance(reason, ssl.SSLError):
        return "HTTPS 握手被远端或中间网络断开。请确认代理正在运行，且能访问 storage.googleapis.com。"
    if isinstance(exc, urllib.error.HTTPError):
        return f"服务器返回 HTTP {exc.code}。"
    if isinstance(reason, (ConnectionError, TimeoutError, OSError)):
        return "网络连接失败或超时。请检查本地代理和 Google Storage 的可达性。"
    return str(reason)


def _read_metadata(path, url):
    try:
        data = json.loads(path.read_text())
        return data if data.get("url") == url else {}
    except (OSError, ValueError, AttributeError):
        return {}


def _md5_header(headers):
    for part in headers.get("x-goog-hash", "").split(","):
        if part.strip().startswith("md5="):
            return part.strip().split("=", 1)[1]
    return None


def _validate(path, metadata, validator):
    expected = metadata.get("total")
    if expected is not None and path.stat().st_size != expected:
        raise DownloadError("文件大小与服务器声明不一致。")
    if metadata.get("md5"):
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(4*1024*1024), b""):
                h.update(chunk)
        if base64.b64encode(h.digest()).decode() != metadata["md5"]:
            raise DownloadError(f"{path.name} 的 Google Storage MD5 校验失败；请移走该临时文件后重试。")
    if validator:
        try:
            validator(path)
        except Exception as exc:
            raise DownloadError(f"{path.name} 的格式检查失败；请移走该文件后重试。") from exc


def download(url, path, *, proxy=None, attempts=4, timeout=30, validator=None,
             opener=None, sleep=time.sleep):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            _validate(path, {}, validator)
        except Exception as exc:
            raise DownloadError(f"已有文件 {path.name} 无法通过格式检查。请移走该文件后重新下载。") from exc
        print(f"复用已完成文件：{path.name}", flush=True)
        return
    if attempts < 1:
        raise ValueError("attempts must be positive")
    client = opener or make_opener(proxy)
    temporary = path.with_suffix(path.suffix+".part")
    sidecar = path.with_suffix(path.suffix+".part.json")
    metadata = _read_metadata(sidecar, url)
    last = None
    for attempt in range(attempts):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            # Resume only a partial response whose URL and remote entity are known.
            if not metadata.get("etag") and not metadata.get("last_modified"):
                offset = 0
            if offset and metadata.get("total") is not None and offset > metadata["total"]:
                raise DownloadError(f"{temporary.name} 超过预期文件大小；请移走该临时文件后重试。")
            if offset and metadata.get("total") == offset:
                _validate(temporary, metadata, validator)
                temporary.replace(path)
                sidecar.unlink(missing_ok=True)
                print(f"完成并校验：{path.name}", flush=True)
                return
            headers = {"User-Agent": "FruitFlyLab/0.2", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
                headers["If-Range"] = metadata.get("etag") or metadata["last_modified"]
            print(f"下载 {path.name}（第 {attempt+1}/{attempts} 次，从 {offset/1048576:.1f} MiB 开始）", flush=True)
            request = urllib.request.Request(url, headers=headers)
            with client.open(request, timeout=timeout) as response:
                status = response.status
                length = response.headers.get("Content-Length")
                if length is not None and not length.isdigit():
                    raise DownloadError("服务器返回了无效的文件长度。")
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[2]) < offset:
                        raise DownloadError("服务器续传范围与本地文件不一致，已停止写入。")
                    total, end = int(match[3]), int(match[2])
                    if end >= total or (length is not None and int(length) != end-offset+1):
                        raise DownloadError("服务器返回了不一致的续传长度。")
                    if offset and metadata.get("etag") and response.headers.get("ETag") != metadata["etag"]:
                        raise DownloadError("服务器文件版本发生变化，无法安全续传。")
                    if offset and not metadata.get("etag") and response.headers.get("Last-Modified") != metadata.get("last_modified"):
                        raise DownloadError("服务器文件修改时间发生变化，无法安全续传。")
                elif status == 200:
                    # If-Range failed or the server ignored Range: replace, never append.
                    offset = 0
                    total = int(length) if length is not None else None
                else:
                    raise DownloadError(f"下载返回了非预期状态 HTTP {status}。")
                metadata = {"url": url, "total": total, "etag": response.headers.get("ETag"),
                            "last_modified": response.headers.get("Last-Modified"),
                            "md5": _md5_header(response.headers) or (metadata.get("md5") if offset else None)}
                received, last_report = offset, time.monotonic()
                with temporary.open("ab" if offset else "wb") as out:
                    # Truncate an obsolete partial file before recording a new
                    # entity, so interruption cannot pair old bytes with new metadata.
                    sidecar.write_text(json.dumps(metadata))
                    while True:
                        try:
                            chunk = response.read(1024*1024)
                        except http.client.IncompleteRead as exc:
                            out.write(exc.partial)
                            raise
                        if not chunk:
                            break
                        out.write(chunk)
                        received += len(chunk)
                        if time.monotonic()-last_report > 3:
                            suffix = f" / {total/1048576:.1f} MiB ({100*received/total:.0f}%)" if total else " MiB"
                            print(f"  {received/1048576:.1f}{suffix}", flush=True)
                            last_report = time.monotonic()
                if total is not None and received != total:
                    raise OSError("Connection ended before the full file was received")
            _validate(temporary, metadata, validator)
            temporary.replace(path)
            sidecar.unlink(missing_ok=True)
            print(f"完成并校验：{path.name} ({path.stat().st_size/1048576:.1f} MiB)", flush=True)
            return
        except DownloadError:
            raise
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            last = exc
            print("  "+explain_error(exc), flush=True)
            if attempt+1 < attempts:
                sleep(min(2**attempt, 4))
    raise DownloadError(f"{path.name} 下载失败。已保留下载进度；修复网络后重新运行即可。\n"
                        f"当前代理：{proxy_label(proxy)}\n{explain_error(last)}") from last


def check_network(url, proxy=None):
    request = urllib.request.Request(url, headers={"Range": "bytes=0-7", "Accept-Encoding": "identity"})
    try:
        with make_opener(proxy).open(request, timeout=15) as response:
            prefix = response.read(8)
            if response.status not in (200, 206) or not prefix.startswith(b"ARROW1"):
                raise DownloadError("端点可连接，但返回内容不是预期的 Feather V2 数据。")
        print("网络检查通过：官方 Feather 数据可访问，HTTPS 校验已开启。", flush=True)
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise DownloadError(explain_error(exc)) from exc
