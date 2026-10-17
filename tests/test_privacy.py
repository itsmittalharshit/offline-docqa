import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from privacy import Redactor  # noqa: E402


@pytest.fixture(scope="module")
def redactor():
    return Redactor()


def test_email_is_redacted(redactor):
    text, types = redactor.redact("Contact helpdesk@example.org for details.")
    assert "example.org" not in text
    assert "EMAIL_ADDRESS" in types


def test_pan_is_redacted(redactor):
    text, types = redactor.redact("The applicant's PAN is ABCDE1234F.")
    assert "ABCDE1234F" not in text


def test_aadhaar_with_context_is_redacted(redactor):
    text, _ = redactor.redact("Aadhaar number 2345 6789 0123 was verified.")
    assert "6789" not in text


def test_mobile_is_redacted(redactor):
    text, _ = redactor.redact("Call the branch on +91 9876543210 today.")
    assert "9876543210" not in text


def test_public_official_names_are_kept(redactor):
    sentence = "The Group was chaired by Dr. Poonam Gupta."
    assert redactor.redact(sentence)[0] == sentence


def test_financial_figures_are_kept(redactor):
    sentence = ("The Committee recommends an aggregate WMA limit of ₹67,839 crore, "
                "an increase of 11.2 per cent, for 2025-30.")
    assert redactor.redact(sentence)[0] == sentence