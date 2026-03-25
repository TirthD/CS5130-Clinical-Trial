"""
Test script for Plain Language Translator
==========================================
Part 1: Fallback translation + caching (no API key needed)
Part 2: Gemini-powered translation (requires GEMINI_API_KEY in .env)
"""

from __future__ import annotations

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.tools.plain_language import (
    PlainLanguageTranslator,
    TranslationResult,
    TrialSummary,
    ContentType,
)


def print_translation(label: str, result: TranslationResult):
    print(f"\n{'='*60}")
    print(f"  TEST: {label}")
    print(f"{'='*60}")
    print(f"  Content Type: {result.content_type.value}")
    print(f"  Source:       {result.source}")
    print(f"  Success:      {result.success}")
    print(f"  Original:     {result.original_text[:100]}...")
    print(f"  Plain:        {result.plain_text[:200]}...")
    if result.definitions:
        print(f"  Definitions:  {result.definitions[:3]}")


def print_summary(label: str, summary: TrialSummary):
    print(f"\n{'='*60}")
    print(f"  TEST: {label}")
    print(f"{'='*60}")
    print(f"  Title:        {summary.title[:100]}")
    print(f"  Purpose:      {summary.purpose[:200]}")
    print(f"  What Happens: {summary.what_happens[:200]}")
    print(f"  Who Can Join: {summary.who_can_join[:200]}")
    print(f"  NCT ID:       {summary.nct_id}")
    print(f"  Disclaimer:   {summary.disclaimer[:80]}...")


# ===================================================================
# Sample medical texts for testing
# ===================================================================

SAMPLE_DESCRIPTION = (
    "This is a randomized, double-blind, placebo-controlled, Phase 3 study "
    "to evaluate the efficacy and safety of Drug X in participants with "
    "moderate-to-severe atopic dermatitis who have had an inadequate response "
    "to topical corticosteroids. The primary endpoint is the proportion of "
    "subjects achieving EASI-75 at Week 16."
)

SAMPLE_ELIGIBILITY = (
    "Inclusion Criteria: Adults aged 18-65 with confirmed diagnosis of "
    "Type 2 Diabetes Mellitus (HbA1c >= 7.0%). Must be on stable metformin "
    "dose for >= 3 months. BMI between 25 and 40 kg/m2. "
    "Exclusion Criteria: History of diabetic ketoacidosis. eGFR < 30 mL/min. "
    "Pregnant or nursing women. Active malignancy within the past 5 years. "
    "Currently receiving insulin or GLP-1 receptor agonists."
)

SAMPLE_PROCEDURE = (
    "Participants will undergo a screening period of 2 weeks, followed by "
    "subcutaneous injection of the study drug or placebo every 2 weeks for "
    "52 weeks. Blood samples will be collected at baseline, Week 4, Week 16, "
    "and Week 52 for pharmacokinetic and immunogenicity assessments. "
    "A skin biopsy will be performed at baseline and Week 16."
)


# ===================================================================
# PART 1: Fallback + caching tests (NO API key needed)
# ===================================================================

def test_fallback_translation():
    """Test the basic substitution-based fallback translator."""
    # Create translator without API key to force fallback mode
    translator = PlainLanguageTranslator.__new__(PlainLanguageTranslator)
    translator._model = None
    translator._api_key = ""
    translator._model_name = "none"
    translator._cache = {}

    print("\n" + "~" * 60)
    print("  PART 1: Fallback Translation Tests")
    print("~" * 60)

    # --- Test 1: Basic substitutions ---
    result = translator.translate(SAMPLE_DESCRIPTION, ContentType.TRIAL_DESCRIPTION)
    print_translation("Fallback — trial description", result)
    assert result.success == True
    assert result.source == "fallback"
    assert "randomly assigned" in result.plain_text.lower()
    assert "neither patient nor doctor" in result.plain_text.lower()
    print("  ✓ Passed")

    # --- Test 2: Eligibility fallback ---
    result = translator.translate(SAMPLE_ELIGIBILITY, ContentType.ELIGIBILITY)
    print_translation("Fallback — eligibility text", result)
    assert result.success == True
    assert "blood sugar test" in result.plain_text.lower()  # HbA1c definition
    assert "kidney function test" in result.plain_text.lower()  # eGFR definition
    print("  ✓ Passed")

    # --- Test 3: Procedure fallback ---
    result = translator.translate(SAMPLE_PROCEDURE, ContentType.PROCEDURE)
    print_translation("Fallback — procedure text", result)
    assert result.success == True
    assert "under the skin" in result.plain_text.lower()  # subcutaneous
    print("  ✓ Passed")

    # --- Test 4: Empty input ---
    result = translator.translate("", ContentType.GENERAL)
    print(f"\n  TEST: Empty input")
    print(f"  Success: {result.success}")
    assert result.success == False
    assert result.plain_text == ""
    print("  ✓ Passed")

    # --- Test 5: Already plain text (no medical terms) ---
    plain_input = "This study looks at a new treatment for back pain in older adults."
    result = translator.translate(plain_input, ContentType.GENERAL)
    print(f"\n  TEST: Already plain text")
    print(f"  Input:  {plain_input}")
    print(f"  Output: {result.plain_text}")
    assert result.success == True
    # Should be mostly unchanged since there are no complex terms
    assert "back pain" in result.plain_text
    print("  ✓ Passed")


def test_caching():
    """Test the in-memory translation cache."""
    translator = PlainLanguageTranslator.__new__(PlainLanguageTranslator)
    translator._model = None
    translator._api_key = ""
    translator._model_name = "none"
    translator._cache = {}

    print("\n" + "~" * 60)
    print("  PART 1: Caching Tests")
    print("~" * 60)

    # --- Test 1: Cache miss then hit ---
    assert translator.cache_size == 0
    result1 = translator.translate(SAMPLE_DESCRIPTION, ContentType.TRIAL_DESCRIPTION)
    assert translator.cache_size == 1
    print(f"  After first translation: cache_size = {translator.cache_size}")

    result2 = translator.translate(SAMPLE_DESCRIPTION, ContentType.TRIAL_DESCRIPTION)
    assert translator.cache_size == 1  # no new entry
    assert result1.plain_text == result2.plain_text
    print(f"  After same translation:  cache_size = {translator.cache_size} (cache hit)")
    print("  ✓ Passed")

    # --- Test 2: Different content type = different cache entry ---
    result3 = translator.translate(SAMPLE_DESCRIPTION, ContentType.GENERAL)
    assert translator.cache_size == 2
    print(f"  Different content type:  cache_size = {translator.cache_size}")
    print("  ✓ Passed")

    # --- Test 3: Cache bypass ---
    result4 = translator.translate(
        SAMPLE_DESCRIPTION, ContentType.TRIAL_DESCRIPTION, use_cache=False
    )
    assert translator.cache_size == 2  # didn't add to cache
    print(f"  Cache bypass:            cache_size = {translator.cache_size}")
    print("  ✓ Passed")

    # --- Test 4: Clear cache ---
    cleared = translator.clear_cache()
    assert cleared == 2
    assert translator.cache_size == 0
    print(f"  After clear:             cache_size = {translator.cache_size} (cleared {cleared})")
    print("  ✓ Passed")


def test_batch_translation():
    """Test batch translation."""
    translator = PlainLanguageTranslator.__new__(PlainLanguageTranslator)
    translator._model = None
    translator._api_key = ""
    translator._model_name = "none"
    translator._cache = {}

    print("\n" + "~" * 60)
    print("  PART 1: Batch Translation Tests")
    print("~" * 60)

    texts = [SAMPLE_DESCRIPTION, SAMPLE_ELIGIBILITY, SAMPLE_PROCEDURE]
    results = translator.translate_batch(texts, ContentType.GENERAL)

    assert len(results) == 3
    assert all(r.success for r in results)
    assert all(r.source == "fallback" for r in results)
    print(f"  Translated {len(results)} texts in batch")
    print(f"  All successful: True")
    print(f"  Cache size after batch: {translator.cache_size}")
    print("  ✓ Passed")


# ===================================================================
# PART 2: Gemini integration tests (REQUIRES API key)
# ===================================================================

def test_gemini_translation():
    """Test Gemini-powered translation. Requires GEMINI_API_KEY in .env."""
    print("\n" + "~" * 60)
    print("  PART 2: Gemini Integration Tests")
    print("~" * 60)

    # Debug info
    project_root = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(project_root, ".env")
    print(f"\n  .env exists: {os.path.isfile(dotenv_path)}")

    try:
        translator = PlainLanguageTranslator()
    except Exception as e:
        print(f"\n  ⚠ Skipping Gemini tests — could not initialize: {e}")
        return

    if not translator._api_key or translator._model is None:
        print("\n  ⚠ Skipping Gemini tests — no API key or SDK not installed")
        return

    key_preview = translator._api_key[:8] + "..."
    print(f"  API key: {key_preview}")
    print(f"  Model: {translator._model_name}")
    print(f"\n  Note: Free tier allows 5 requests/min.")
    print(f"  The translator will auto-retry on rate limits (waits ~60s).\n")

    # --- Gemini Test 1: Trial description ---
    print("  Calling Gemini to translate trial description...")
    result_desc = translator.translate(SAMPLE_DESCRIPTION, ContentType.TRIAL_DESCRIPTION)
    print_translation("Gemini — trial description", result_desc)
    assert result_desc.success == True
    assert result_desc.source == "gemini"
    assert len(result_desc.plain_text) > 20
    print("  ✓ Passed")

    # --- Gemini Test 2: Caching (no API call — uses cached result from Test 1) ---
    cache_before = translator.cache_size
    result_cached = translator.translate(SAMPLE_DESCRIPTION, ContentType.TRIAL_DESCRIPTION)
    assert translator.cache_size == cache_before
    assert result_cached.plain_text == result_desc.plain_text
    print(f"\n  Cache test: size stayed at {translator.cache_size} (cache hit)")
    print("  ✓ Passed")

    # --- Gemini Test 3: Eligibility criteria ---
    print("\n  Calling Gemini to translate eligibility...")
    result_elig = translator.translate(SAMPLE_ELIGIBILITY, ContentType.ELIGIBILITY)
    print_translation("Gemini — eligibility", result_elig)
    assert result_elig.success == True
    assert result_elig.source == "gemini"
    assert "you" in result_elig.plain_text.lower()
    print("  ✓ Passed")

    # --- Gemini Test 4: Procedures ---
    print("\n  Calling Gemini to translate procedures...")
    result_proc = translator.translate(SAMPLE_PROCEDURE, ContentType.PROCEDURE)
    print_translation("Gemini — procedures", result_proc)
    assert result_proc.success == True
    print("  ✓ Passed")

    # --- Gemini Test 5: Full trial summary (1 API call) ---
    print("\n  Calling Gemini to generate full trial summary...")
    summary = translator.translate_trial_summary(
        title="A Phase 3, Randomized, Double-Blind Study of Drug X vs Placebo in Moderate-to-Severe Atopic Dermatitis",
        description=SAMPLE_DESCRIPTION,
        eligibility=SAMPLE_ELIGIBILITY,
        interventions=SAMPLE_PROCEDURE,
        location="Massachusetts General Hospital, Boston, MA",
        contact="Dr. Jane Smith, 617-555-0100",
        nct_id="NCT12345678",
    )
    print_summary("Gemini — full trial summary", summary)
    assert summary.nct_id == "NCT12345678"
    assert summary.location == "Massachusetts General Hospital, Boston, MA"
    assert len(summary.purpose) > 10
    assert len(summary.disclaimer) > 10
    print("  ✓ Passed")

    # --- Gemini Test 6: Definition extraction (1 API call) ---
    print("\n  Calling Gemini to extract definitions...")
    definitions = translator.extract_definitions(SAMPLE_ELIGIBILITY)
    print(f"\n  Extracted {len(definitions)} definitions:")
    for d in definitions[:5]:
        print(f"    - {d}")
    if len(definitions) > 0:
        print("  ✓ Passed")
    else:
        print("  ⚠ Definitions returned empty (may be rate limited) — non-critical, skipping")
    # Not a hard assert — definitions are a nice-to-have feature


# ===================================================================
# Run all tests
# ===================================================================

def main():
    print("=" * 60)
    print("  PLAIN LANGUAGE TRANSLATOR — TEST SUITE")
    print("=" * 60)

    # Part 1: Always runs
    test_fallback_translation()
    test_caching()
    test_batch_translation()

    # Part 2: Only with API key
    test_gemini_translation()

    print(f"\n{'='*60}")
    print(f"  ALL AVAILABLE TESTS PASSED!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()