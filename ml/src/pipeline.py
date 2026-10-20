"""Phase 8: text extraction (PDF, URL) and ToS-aware sentence splitting."""
import ipaddress
import re
import socket
import urllib.request
from urllib.parse import urljoin, urlparse

MIN_WORDS = 4
MAX_URL_BYTES = 5 * 1024 * 1024
URL_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (compatible; FinePrintGuard/1.0; +https://github.com/424103-achu/fineguard)"

_nlp = None


def _sentencizer():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.blank("en")
        _nlp.add_pipe("sentencizer")
        _nlp.max_length = 2_000_000
    return _nlp


def text_from_pdf(data: bytes) -> str:
    import pymupdf   # PyMuPDF (older code imports it as `fitz`)

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _is_public_host(hostname: str) -> bool:
    """Block localhost / private addresses so the URL feature can't probe your network."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, UnicodeError):
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            return False
    return bool(infos)


def _check_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only http(s) URLs are supported")
    if not _is_public_host(parsed.hostname):
        raise ValueError("URL host is not allowed")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch_html(url: str, max_redirects: int = 5) -> str:
    """Download a page, re-checking the host on every redirect hop."""
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(max_redirects + 1):
        _check_url(url)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "text/html,*/*;q=0.8"})
        try:
            resp = opener.open(req, timeout=URL_TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                url = urljoin(url, e.headers["Location"])
                continue
            raise ValueError(f"Could not download the page (HTTP {e.code})") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ValueError("Could not download the page") from None
        with resp:
            data = resp.read(MAX_URL_BYTES + 1)
            if len(data) > MAX_URL_BYTES:
                raise ValueError("Page is too large")
            charset = resp.headers.get_content_charset() or "utf-8"
        return data.decode(charset, errors="replace")
    raise ValueError("Too many redirects")


def text_from_url(url: str) -> str:
    import trafilatura

    html = fetch_html(url)
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if not text:
        raise ValueError("Could not extract readable text (the page may need JavaScript); "
                         "paste the text or upload a PDF instead")
    return text


_BLOCK_SPLIT = re.compile(r"\n\s*\n|\n(?=\s*(?:[-•*]|\d+[.)]|\([a-z0-9]+\))\s)")
_LIST_MARKER = re.compile(r"^(?:[-•*]|\d+[.)]|\([a-z0-9]+\))\s+")


def split_sentences(text: str) -> list[str]:
    """Split on line breaks first (ToS use lists and headings), then on sentence boundaries.
    Fragments shorter than MIN_WORDS words (headings, page numbers) are dropped."""
    nlp = _sentencizer()
    sentences = []
    for block in _BLOCK_SPLIT.split(text):
        block = re.sub(r"\s+", " ", block).strip()
        block = _LIST_MARKER.sub("", block)
        if not block:
            continue
        for sent in nlp(block).sents:
            s = sent.text.strip()
            if len(s.split()) >= MIN_WORDS:
                sentences.append(s)
    return sentences
