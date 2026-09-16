import base64
import hashlib
import http.client
import io
import json
import ssl
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from flylab.download import DownloadError, download, make_opener, proxy_label, resolve_proxy


URL = "https://storage.googleapis.com/test/annotations.feather"
DATA = b"ARROW1 download test data ARROW1"
MD5 = base64.b64encode(hashlib.md5(DATA).digest()).decode()


class Response:
    def __init__(self, chunks, status=200, headers=None):
        self.chunks = iter(chunks)
        self.status = status
        self.headers = headers or {}

    def read(self, _size):
        chunk = next(self.chunks, b"")
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def response_headers(**extra):
    return {"Content-Length": str(len(DATA)), "ETag": '"version1"',
            "x-goog-hash": "crc32c=unused,md5="+MD5, **extra}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/"data.feather"
        self.part = self.path.with_suffix(".feather.part")
        self.sidecar = self.path.with_suffix(".feather.part.json")

    def run_download(self, client, **kwargs):
        with redirect_stdout(io.StringIO()):
            download(URL, self.path, opener=client, sleep=lambda _: None, **kwargs)

    def save_partial(self, content=DATA[:5]):
        self.part.write_bytes(content)
        self.sidecar.write_text(json.dumps({"url": URL, "etag": '"version1"',
                                           "total": len(DATA), "md5": MD5}))

    def test_midstream_disconnect_resumes_and_verifies_full_file(self):
        first = Response([DATA[:3], http.client.IncompleteRead(DATA[3:5], len(DATA)-5)],
                         headers=response_headers())
        second = Response([DATA[5:]], status=206, headers=response_headers(**{
            "Content-Length": str(len(DATA)-5),
            "Content-Range": f"bytes 5-{len(DATA)-1}/{len(DATA)}"}))
        client = Client(first, second)
        self.run_download(client)
        self.assertEqual(self.path.read_bytes(), DATA)
        self.assertEqual(client.requests[1].get_header("Range"), "bytes=5-")
        self.assertEqual(client.requests[1].get_header("If-range"), '"version1"')
        self.assertFalse(self.part.exists())
        self.assertFalse(self.sidecar.exists())

    def test_later_invocation_resumes_saved_partial(self):
        self.save_partial()
        client = Client(Response([DATA[5:]], status=206, headers=response_headers(**{
            "Content-Length": str(len(DATA)-5),
            "Content-Range": f"bytes 5-{len(DATA)-1}/{len(DATA)}"})))
        self.run_download(client)
        self.assertEqual(client.requests[0].get_header("Range"), "bytes=5-")
        self.assertEqual(self.path.read_bytes(), DATA)

    def test_ignored_range_restarts_instead_of_appending(self):
        self.save_partial()
        self.run_download(Client(Response([DATA], headers=response_headers())))
        self.assertEqual(self.path.read_bytes(), DATA)

    def test_incorrect_range_cannot_modify_partial(self):
        self.save_partial()
        with self.assertRaisesRegex(DownloadError, "续传范围"):
            self.run_download(Client(Response([DATA], status=206, headers=response_headers(**{
                "Content-Range": f"bytes 0-{len(DATA)-1}/{len(DATA)}"}))))
        self.assertEqual(self.part.read_bytes(), DATA[:5])
        self.assertFalse(self.path.exists())

    def test_wrong_checksum_is_never_promoted(self):
        with self.assertRaisesRegex(DownloadError, "MD5"):
            self.run_download(Client(Response([b"X"+DATA[1:]], headers=response_headers())))
        self.assertFalse(self.path.exists())

    def test_ssl_eof_retries_without_leaving_completed_file(self):
        error = urllib.error.URLError(ssl.SSLEOFError(8, "Unexpected EOF"))
        client = Client(error, error)
        with self.assertRaisesRegex(DownloadError, "HTTPS 握手"):
            self.run_download(client, attempts=2)
        self.assertEqual(len(client.requests), 2)
        self.assertFalse(self.path.exists())

    def test_unidentified_partial_restarts(self):
        self.part.write_bytes(b"old unverified download")
        client = Client(Response([DATA], headers=response_headers()))
        self.run_download(client)
        self.assertIsNone(client.requests[0].get_header("Range"))
        self.assertEqual(self.path.read_bytes(), DATA)

    def test_proxy_config_precedence_and_redaction(self):
        config = Path(self.directory.name)/"network.json"
        config.write_text('{"proxy":"http://127.0.0.1:10808"}')
        with patch("flylab.download.NETWORK_CONFIG", config), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_proxy(), "http://127.0.0.1:10808")
            self.assertEqual(resolve_proxy("http://localhost:9999"), "http://localhost:9999")
            self.assertEqual(resolve_proxy(direct=True), "")
            with patch.dict("os.environ", {"FLY_LAB_PROXY": "http://localhost:7777"}):
                self.assertEqual(resolve_proxy(), "http://localhost:7777")
            config.write_text("[]")
            with self.assertRaises(DownloadError):
                resolve_proxy()
        self.assertEqual(proxy_label("http://user:secret@localhost:9999"), "http://localhost:9999")
        with self.assertRaises(DownloadError):
            resolve_proxy("socks5://localhost:1080")

    def test_proxy_keeps_tls_certificate_verification_enabled(self):
        import urllib.request
        opener = make_opener("http://127.0.0.1:10808")
        handler = next(h for h in opener.handlers if isinstance(h, urllib.request.HTTPSHandler))
        self.assertEqual(handler._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(handler._context.check_hostname)


if __name__ == "__main__":
    unittest.main()
