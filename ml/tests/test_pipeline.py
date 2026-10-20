import pytest

from src.pipeline import _is_public_host, fetch_html, split_sentences, text_from_pdf


def test_splits_lists_and_sentences():
    text = ("Terms of use.\n\n1. We may change these terms at any time. You agree to this."
            "\n2) We can delete your content.")
    sents = split_sentences(text)
    assert "We may change these terms at any time." in sents
    assert any("delete your content" in s for s in sents)


def test_list_markers_stripped():
    sents = split_sentences("Rules:\n- we may suspend your account at any time.")
    assert sents == ["we may suspend your account at any time."]


def test_drops_tiny_fragments():
    assert split_sentences("Hi.\n\nOk.") == []


def test_joins_wrapped_lines():
    sents = split_sentences("We are not liable for any\nindirect damages whatsoever.")
    assert sents == ["We are not liable for any indirect damages whatsoever."]


def test_private_hosts_blocked():
    assert not _is_public_host("localhost")
    assert not _is_public_host("127.0.0.1")
    assert not _is_public_host("169.254.169.254")
    assert not _is_public_host("10.0.0.1")


@pytest.mark.parametrize("url", ["ftp://example.com/x", "http://127.0.0.1:8000/health",
                                 "file:///etc/passwd", "http://localhost/"])
def test_bad_urls_rejected(url):
    with pytest.raises(ValueError):
        fetch_html(url)


def test_pdf_extraction():
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "We may terminate your account at any time.")
    data = doc.tobytes()
    assert "terminate your account" in text_from_pdf(data)
