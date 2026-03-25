"""
Test script for Eligibility Parser
===================================
Part 1: Tests regex parsing + eligibility checking (runs without API key)
Part 2: Tests Gemini-based parsing (requires GEMINI_API_KEY in .env)
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.tools.eligibility_parser import (
    EligibilityParser,
    EligibilityCriteria,
    EligibilityResult,
    UserProfile,
    Gender,
)


def print_criteria(label, criteria):
    print(f"\n{'='*60}")
    print(f"  PARSE TEST: {label}")
    print(f"{'='*60}")
    print(f"  Age Range:     {criteria.min_age} - {criteria.max_age}")
    print(f"  Gender:        {criteria.gender.value}")
    print(f"  Inclusions:    {criteria.inclusion_criteria}")
    print(f"  Exclusions:    {criteria.exclusion_criteria}")
    print(f"  Required Meds: {criteria.required_medications}")
    print(f"  Excluded Meds: {criteria.excluded_medications}")
    print(f"  Confidence:    {criteria.parse_confidence}")


def print_result(label, result):
    print(f"\n{'='*60}")
    print(f"  CHECK TEST: {label}")
    print(f"{'='*60}")
    print(f"  Eligible:    {result.is_eligible}")
    print(f"  Met:         {result.met_criteria}")
    print(f"  Unmet:       {result.unmet_criteria}")
    print(f"  Uncertain:   {result.uncertain_criteria}")
    print(f"  Summary:     {result.summary}")


# ===================================================================
# PART 1: Regex fallback + eligibility check tests (NO API key needed)
# ===================================================================

def test_regex_parsing():
    """Test the regex fallback parser on various eligibility text formats."""
    parser = EligibilityParser.__new__(EligibilityParser)  # skip __init__ (no API key)

    # --- Test 1: Standard age range ---
    print("\n" + "~"*60)
    print("  PART 1: Regex Parsing Tests")
    print("~"*60)

    text1 = """
    Inclusion Criteria:
    - Patients aged 18-65 with confirmed diagnosis of Type 2 Diabetes
    - HbA1c >= 7.0%
    - On stable metformin dose for at least 3 months

    Exclusion Criteria:
    - History of diabetic ketoacidosis
    - eGFR < 30 mL/min
    - Pregnant or nursing women
    - Active malignancy within the past 5 years
    """
    criteria = parser._parse_with_regex(text1)
    print_criteria("Standard format — aged 18-65", criteria)
    assert criteria.min_age == 18, f"Expected min_age 18, got {criteria.min_age}"
    assert criteria.max_age == 65, f"Expected max_age 65, got {criteria.max_age}"
    assert criteria.gender == Gender.ALL
    assert len(criteria.inclusion_criteria) > 0
    assert len(criteria.exclusion_criteria) > 0
    print("  ✓ Passed")

    # --- Test 2: "≥18 years" format ---
    text2 = "Eligible participants must be ≥18 years of age with a BMI ≥30."
    criteria = parser._parse_with_regex(text2)
    print_criteria("Min-only — ≥18 years", criteria)
    assert criteria.min_age == 18
    assert criteria.max_age is None
    print("  ✓ Passed")

    # --- Test 3: "at least X, up to Y" format ---
    text3 = "Participants must be at least 21 years old, up to 70 years of age."
    criteria = parser._parse_with_regex(text3)
    print_criteria("Text format — at least 21, up to 70", criteria)
    assert criteria.min_age == 21
    assert criteria.max_age == 70
    print("  ✓ Passed")

    # --- Test 4: Female only ---
    text4 = """
    Inclusion Criteria:
    - Female only
    - Ages 18 to 45
    - Diagnosed with polycystic ovary syndrome
    """
    criteria = parser._parse_with_regex(text4)
    print_criteria("Female only trial", criteria)
    assert criteria.gender == Gender.FEMALE
    assert criteria.min_age == 18
    assert criteria.max_age == 45
    print("  ✓ Passed")

    # --- Test 5: No age mentioned ---
    text5 = """
    Inclusion Criteria:
    - Confirmed diagnosis of major depressive disorder
    Exclusion Criteria:
    - Active suicidal ideation
    """
    criteria = parser._parse_with_regex(text5)
    print_criteria("No age mentioned", criteria)
    assert criteria.min_age is None
    assert criteria.max_age is None
    print("  ✓ Passed")


def test_eligibility_checking():
    """Test the deterministic eligibility checking logic."""
    parser = EligibilityParser.__new__(EligibilityParser)

    print("\n" + "~"*60)
    print("  PART 1: Eligibility Check Tests")
    print("~"*60)

    # --- Test 1: Fully eligible user ---
    criteria = EligibilityCriteria(
        min_age=18,
        max_age=65,
        gender=Gender.ALL,
        inclusion_criteria=["type 2 diabetes"],
        exclusion_criteria=["pregnant", "active cancer"],
    )
    user = UserProfile(age=45, gender="male", conditions=["type 2 diabetes"])
    result = parser.check_eligibility(criteria, user)
    print_result("Fully eligible — 45yo male with T2D", result)
    assert result.is_eligible == True
    print("  ✓ Passed")

    # --- Test 2: Too young ---
    user_young = UserProfile(age=16, gender="male", conditions=["type 2 diabetes"])
    result = parser.check_eligibility(criteria, user_young)
    print_result("Too young — age 16, min is 18", result)
    assert result.is_eligible == False
    assert any("below minimum" in r for r in result.unmet_criteria)
    print("  ✓ Passed")

    # --- Test 3: Too old ---
    user_old = UserProfile(age=70, gender="female", conditions=["type 2 diabetes"])
    result = parser.check_eligibility(criteria, user_old)
    print_result("Too old — age 70, max is 65", result)
    assert result.is_eligible == False
    assert any("above maximum" in r for r in result.unmet_criteria)
    print("  ✓ Passed")

    # --- Test 4: Excluded by condition ---
    user_cancer = UserProfile(age=45, gender="male", conditions=["type 2 diabetes", "active cancer"])
    result = parser.check_eligibility(criteria, user_cancer)
    print_result("Excluded — has active cancer", result)
    assert result.is_eligible == False
    assert any("Excluded by" in r for r in result.unmet_criteria)
    print("  ✓ Passed")

    # --- Test 5: Gender mismatch ---
    criteria_female = EligibilityCriteria(
        min_age=18,
        max_age=50,
        gender=Gender.FEMALE,
        inclusion_criteria=["polycystic ovary syndrome"],
    )
    user_male = UserProfile(age=30, gender="male", conditions=["polycystic ovary syndrome"])
    result = parser.check_eligibility(criteria_female, user_male)
    print_result("Gender mismatch — male for female-only trial", result)
    assert result.is_eligible == False
    assert any("requires female" in r for r in result.unmet_criteria)
    print("  ✓ Passed")

    # --- Test 6: Missing user info (age not provided) ---
    criteria_age = EligibilityCriteria(min_age=18, max_age=65)
    user_no_age = UserProfile(age=None, gender="male")
    result = parser.check_eligibility(criteria_age, user_no_age)
    print_result("Missing age — uncertain", result)
    assert len(result.uncertain_criteria) > 0
    assert result.is_eligible == True  # no hard failure, just uncertainty
    print("  ✓ Passed")

    # --- Test 7: Medication checks ---
    criteria_meds = EligibilityCriteria(
        min_age=18,
        max_age=65,
        required_medications=["metformin"],
        excluded_medications=["insulin glargine"],
    )
    user_meds = UserProfile(
        age=45, gender="male",
        conditions=[], medications=["metformin", "lisinopril"]
    )
    result = parser.check_eligibility(criteria_meds, user_meds)
    print_result("Medication check — on metformin, not on insulin", result)
    assert result.is_eligible == True
    assert any("metformin" in r.lower() for r in result.met_criteria)
    print("  ✓ Passed")

    # --- Test 8: On excluded medication ---
    user_excluded_med = UserProfile(
        age=45, gender="male",
        conditions=[], medications=["insulin glargine"]
    )
    result = parser.check_eligibility(criteria_meds, user_excluded_med)
    print_result("Excluded medication — on insulin glargine", result)
    assert result.is_eligible == False
    assert any("excluded medication" in r.lower() for r in result.unmet_criteria)
    print("  ✓ Passed")


# ===================================================================
# PART 2: Gemini integration tests (REQUIRES API key)
# ===================================================================

def test_gemini_parsing():
    """Test full Gemini-based parsing. Requires GEMINI_API_KEY in .env."""
    print("\n" + "~"*60)
    print("  PART 2: Gemini Integration Tests")
    print("~"*60)

    # Debug: show where .env should be
    project_root = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(project_root, ".env")
    print(f"\n  Project root: {project_root}")
    print(f"  Looking for .env at: {dotenv_path}")
    print(f"  .env exists: {os.path.isfile(dotenv_path)}")

    # Check if the key loaded
    key_from_env = os.getenv("GEMINI_API_KEY", "")
    key_status = f"{key_from_env[:8]}..." if len(key_from_env) > 8 else "(empty)"
    print(f"  GEMINI_API_KEY: {key_status}")

    try:
        parser = EligibilityParser()
    except Exception as e:
        print(f"\n  ⚠ Skipping Gemini tests — could not initialize: {e}")
        return

    if not parser._api_key or parser._model is None:
        print("\n  ⚠ Skipping Gemini tests — no API key or SDK not installed")
        print("    To run these tests:")
        print("    1. pip install google-generativeai")
        print("    2. Add GEMINI_API_KEY=your_key to .env file")
        return

    # --- Gemini Test 1: Standard eligibility text ---
    text = """
    Inclusion Criteria:
    - Adults aged 18-65
    - Confirmed diagnosis of Type 2 Diabetes Mellitus (HbA1c ≥ 7.0%)
    - Currently on stable dose of metformin for at least 3 months
    - BMI between 25 and 40 kg/m²

    Exclusion Criteria:
    - History of diabetic ketoacidosis in the past 12 months
    - eGFR < 30 mL/min/1.73m²
    - Pregnant or nursing women
    - Active malignancy within the past 5 years
    - Currently taking insulin or GLP-1 receptor agonists
    """

    print("\n  Calling Gemini to parse eligibility text...")
    criteria = parser.parse_eligibility(text)
    print_criteria("Gemini — standard diabetes trial", criteria)

    assert criteria.min_age == 18, f"Expected 18, got {criteria.min_age}"
    assert criteria.max_age == 65, f"Expected 65, got {criteria.max_age}"
    assert criteria.parse_confidence > 0.5
    assert len(criteria.inclusion_criteria) > 0
    assert len(criteria.exclusion_criteria) > 0
    print("  ✓ Passed")

    # --- Gemini Test 2: Check eligibility for the parsed criteria ---
    user = UserProfile(
        age=45,
        gender="male",
        conditions=["type 2 diabetes"],
        medications=["metformin"]
    )
    result = parser.check_eligibility(criteria, user)
    print_result("Gemini parsed + eligible user check", result)
    assert result.is_eligible == True
    print("  ✓ Passed")

    # --- Gemini Test 3: Messy, unstructured text ---
    messy_text = """
    Patients must be between 21 and 55 years of age, have a documented
    history of chronic migraine (15 or more headache days per month for
    at least 3 months), and must not be pregnant or planning to become
    pregnant. Patients who have had botulinum toxin injections within
    the past 4 months are not eligible. Must not have medication overuse
    headache or a history of stroke.
    """
    print("\n  Calling Gemini to parse messy unstructured text...")
    criteria2 = parser.parse_eligibility(messy_text)
    print_criteria("Gemini — messy migraine trial text", criteria2)

    assert criteria2.min_age == 21
    assert criteria2.max_age == 55
    print("  ✓ Passed")


# ===================================================================
# Run all tests
# ===================================================================

def main():
    print("="*60)
    print("  ELIGIBILITY PARSER — TEST SUITE")
    print("="*60)

    # Part 1: Always runs (no API key needed)
    test_regex_parsing()
    test_eligibility_checking()

    # Part 2: Only runs if API key is available
    test_gemini_parsing()

    print(f"\n{'='*60}")
    print(f"  ALL AVAILABLE TESTS PASSED!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()