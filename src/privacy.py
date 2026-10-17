from collections import Counter

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

# Policy: contact details and ID numbers. Names of public officials are NOT redacted.
DEFAULT_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE",
                    "PAN_NUMBER", "AADHAAR_NUMBER", "IN_MOBILE"]


def indian_recognizers():
    return [
        PatternRecognizer(
            supported_entity="PAN_NUMBER",
            patterns=[Pattern("PAN", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", 0.85)],
        ),
        PatternRecognizer(
            supported_entity="AADHAAR_NUMBER",
            patterns=[Pattern("Aadhaar", r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b", 0.4)],
            context=["aadhaar", "aadhar", "uid", "uidai"],
        ),
        PatternRecognizer(
            supported_entity="IN_MOBILE",
            patterns=[Pattern("Indian mobile", r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)", 0.7)],
        ),
    ]


class Redactor:
    def __init__(self, entities=DEFAULT_ENTITIES, threshold=0.5):
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(),
                                       supported_languages=["en"])
        for recognizer in indian_recognizers():
            self.analyzer.registry.add_recognizer(recognizer)
        self.anonymizer = AnonymizerEngine()
        self.entities = entities
        self.threshold = threshold

    def redact(self, text):
        """Return (redacted_text, [entity types found]). Never returns the PII itself."""
        results = self.analyzer.analyze(text=text, entities=self.entities, language="en",
                                        score_threshold=self.threshold)
        if not results:
            return text, []
        redacted = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return redacted.text, [r.entity_type for r in results]


def redact_pages(pages, redactor):
    """Redact every paragraph in place; return counts per entity type."""
    counts = Counter()
    for p in pages:
        cleaned = []
        for para in p["paragraphs"]:
            text, types = redactor.redact(para)
            cleaned.append(text)
            counts.update(types)
        p["paragraphs"] = cleaned
        p["text"] = "\n\n".join(cleaned)
    return counts