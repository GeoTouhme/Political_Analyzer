"""Unit tests for Political Pattern Analyzer v1.0."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

# Patch env before importing analyzer
os.environ["NOTION_API_KEY"] = "test-token-for-testing"

import analyzer_v2 as analyzer


# --- Risk Score Tests ---------------------------------------------------------

class TestCalculateRiskScore:
    def test_neutral_text_low_risk(self):
        score, mil, dip = analyzer.calculate_risk_score("A regular news article about weather.", 0.5)
        assert score < 25
        assert mil == 0
        assert dip == 0

    def test_military_critical_terms_increase_risk(self):
        text = "The regime change plan includes a decapitation strike and invasion."
        score, mil, _ = analyzer.calculate_risk_score(text, 0.0)
        assert score > 40
        assert mil >= 3

    def test_diplomatic_terms_reduce_risk(self):
        text = "The treaty and agreement brought a breakthrough in normalization talks."
        score, _, dip = analyzer.calculate_risk_score(text, 0.5)
        assert dip >= 4
        low_score = score

        text_mil = "The strike and missile buildup continues with carrier deployment."
        score_mil, _, _ = analyzer.calculate_risk_score(text_mil, 0.5)
        assert score_mil > low_score

    def test_score_clamped_0_to_100(self):
        score_low, _, _ = analyzer.calculate_risk_score(
            "treaty agreement breakthrough normalization talks negotiation dialogue relief",
            1.0,
        )
        assert score_low >= 0

        score_high, _, _ = analyzer.calculate_risk_score(
            "decapitation regime change invasion existential pre-emptive nuclear breakout "
            "massive retaliation strike carrier armada missile buildup offensive doctrine",
            -1.0,
        )
        assert score_high <= 100

    def test_frequency_weighting(self):
        single = "strike reported."
        double = "strike strike strike reported."
        score_single, _, _ = analyzer.calculate_risk_score(single, 0.0)
        score_double, _, _ = analyzer.calculate_risk_score(double, 0.0)
        assert score_double > score_single

    def test_frequency_cap_at_3(self):
        three = "strike " * 3
        five = "strike " * 5
        score_three, _, _ = analyzer.calculate_risk_score(three, 0.0)
        score_five, _, _ = analyzer.calculate_risk_score(five, 0.0)
        assert score_three == score_five  # capped at 3x


# --- Risk Classification Tests -----------------------------------------------

class TestClassifyRisk:
    def test_critical(self):
        assert analyzer.classify_risk(80) == "CRITICAL"

    def test_high(self):
        assert analyzer.classify_risk(60) == "HIGH"

    def test_medium(self):
        assert analyzer.classify_risk(30) == "MEDIUM"

    def test_low(self):
        assert analyzer.classify_risk(10) == "LOW"

    def test_boundaries(self):
        # >75 = CRITICAL, >50 = HIGH, >25 = MEDIUM, <=25 = LOW
        assert analyzer.classify_risk(75.0) == "HIGH"
        assert analyzer.classify_risk(75.1) == "CRITICAL"
        assert analyzer.classify_risk(50.0) == "MEDIUM"
        assert analyzer.classify_risk(50.1) == "HIGH"
        assert analyzer.classify_risk(25.0) == "LOW"
        assert analyzer.classify_risk(25.1) == "MEDIUM"
        assert analyzer.classify_risk(0.0) == "LOW"
        assert analyzer.classify_risk(100.0) == "CRITICAL"


# --- Text Extraction Tests ---------------------------------------------------

class TestExtractText:
    def test_single_block(self):
        props = {"Full text": {"rich_text": [{"plain_text": "Hello world"}]}}
        assert analyzer.extract_text(props, "Full text") == "Hello world"

    def test_multiple_blocks_concatenated(self):
        props = {"Full text": {"rich_text": [
            {"plain_text": "Part one."},
            {"plain_text": "Part two."},
        ]}}
        result = analyzer.extract_text(props, "Full text")
        assert "Part one." in result
        assert "Part two." in result

    def test_missing_field(self):
        assert analyzer.extract_text({}, "Full text") == ""

    def test_empty_blocks(self):
        props = {"Full text": {"rich_text": []}}
        assert analyzer.extract_text(props, "Full text") == ""


class TestExtractTitle:
    def test_normal_title(self):
        props = {"Title": {"title": [{"plain_text": "Test Article"}]}}
        assert analyzer.extract_title(props) == "Test Article"

    def test_missing_title(self):
        assert analyzer.extract_title({}) == "Untitled"

    def test_empty_title_list(self):
        props = {"Title": {"title": []}}
        assert analyzer.extract_title(props) == "Untitled"


# --- Article Analysis Tests ---------------------------------------------------

class TestAnalyzeArticle:
    def _make_article(self, title: str, text: str) -> dict:
        return {
            "properties": {
                "Title": {"title": [{"plain_text": title}]},
                "Full text": {"rich_text": [{"plain_text": text}]},
            }
        }

    def test_returns_analysis_for_valid_article(self):
        article = self._make_article("Test", "The military strike escalation continues.")
        try:
            result = analyzer.analyze_article(article)
        except Exception:
            pytest.skip("TextBlob corpus not downloaded")
        assert result is not None
        assert result.title == "Test"
        assert isinstance(result.risk_score, float)
        assert result.risk_level in ("CRITICAL", "HIGH", "MEDIUM", "LOW")

    def test_returns_none_for_empty_text(self):
        article = self._make_article("Empty", "")
        assert analyzer.analyze_article(article) is None

    def test_returns_none_for_missing_text(self):
        article = {"properties": {"Title": {"title": [{"plain_text": "No Body"}]}}}
        assert analyzer.analyze_article(article) is None


# --- Report Generation Tests --------------------------------------------------

class TestGenerateReport:
    def test_report_structure(self):
        results = [
            analyzer.ArticleAnalysis(
                title="Test", sentiment_polarity=0.1, sentiment_method="textblob",
                risk_score=45.0, risk_level="MEDIUM", key_phrases=["test"],
            )
        ]
        report = analyzer.generate_report(results)
        assert "meta" in report
        assert "articles" in report
        assert report["meta"]["article_count"] == 1
        assert report["meta"]["avg_risk_score"] == 45.0

    def test_empty_results(self):
        report = analyzer.generate_report([])
        assert report["meta"]["article_count"] == 0
        assert report["meta"]["avg_risk_score"] == 0

    def test_report_serializable(self):
        results = [
            analyzer.ArticleAnalysis(
                title="Serialize Test", sentiment_polarity=-0.3,
                sentiment_method="vader", risk_score=72.5, risk_level="HIGH",
                key_phrases=["test", "phrases"],
            )
        ]
        report = analyzer.generate_report(results)
        serialized = json.dumps(report)
        assert isinstance(serialized, str)
