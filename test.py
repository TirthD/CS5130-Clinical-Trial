"""Test script for Medical Term Mapper — validates all matching strategies."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.tools.medical_term_mapper import MedicalTermMapper

def print_result(label, result):
    print(f"\n{'='*60}")
    print(f"  TEST: {label}")
    print(f"{'='*60}")
    print(f"  Input:       {result.original_term}")
    print(f"  Preferred:   {result.preferred_term}")
    print(f"  Alternatives:{result.alternatives[:3]}")
    print(f"  Confidence:  {result.confidence:.2f}")
    print(f"  Match Type:  {result.match_type}")
    print(f"  High Conf?:  {result.is_high_confidence()}")

def main():
    # Initialize mapper with our synonyms file
    mapper = MedicalTermMapper(
        synonyms_path=os.path.join(os.path.dirname(__file__), "data", "medical_synonyms.json")
    )
    print(f"Loaded mapper: {mapper}")
    print(f"Known terms: {len(mapper.get_all_known_terms())}")

    # ---------------------------------------------------------------
    # Test 1: Exact match — user types a known lay term
    # ---------------------------------------------------------------
    result = mapper.map_term("heart attack")
    print_result("Exact match — 'heart attack'", result)
    assert result.match_type == "exact"
    assert result.preferred_term == "myocardial infarction"
    assert result.confidence == 1.0

    # ---------------------------------------------------------------
    # Test 2: Exact match — another common term
    # ---------------------------------------------------------------
    result = mapper.map_term("type 2 diabetes")
    print_result("Exact match — 'type 2 diabetes'", result)
    assert result.match_type == "exact"
    assert result.preferred_term == "diabetes mellitus type 2"

    # ---------------------------------------------------------------
    # Test 3: Reverse match — user typed the medical term directly
    # ---------------------------------------------------------------
    result = mapper.map_term("myocardial infarction")
    print_result("Reverse match — 'myocardial infarction'", result)
    assert result.match_type == "reverse"
    assert result.preferred_term == "myocardial infarction"
    assert result.confidence == 0.95

    # ---------------------------------------------------------------
    # Test 4: Reverse match — user typed an abbreviation
    # ---------------------------------------------------------------
    result = mapper.map_term("COPD")
    print_result("Exact match (abbreviation key) — 'COPD'", result)
    assert result.match_type == "exact"
    assert result.preferred_term == "pulmonary disease, chronic obstructive"

    # ---------------------------------------------------------------
    # Test 4b: Reverse match — user typed an alternative term
    # ---------------------------------------------------------------
    result = mapper.map_term("CHF")
    print_result("Reverse match (abbreviation) — 'CHF'", result)
    assert result.match_type == "reverse"
    assert result.preferred_term == "heart failure"

    # ---------------------------------------------------------------
    # Test 5: Fuzzy match — misspelling
    # ---------------------------------------------------------------
    result = mapper.map_term("diebetes")
    print_result("Fuzzy match — 'diebetes' (misspelling)", result)
    assert result.match_type == "fuzzy"
    assert "diabetes" in result.preferred_term.lower()

    # ---------------------------------------------------------------
    # Test 6: Fuzzy match — close variant
    # ---------------------------------------------------------------
    result = mapper.map_term("alzheimer")
    print_result("Fuzzy match — 'alzheimer' (close variant)", result)
    assert result.match_type == "fuzzy"
    assert "alzheimer" in result.preferred_term.lower()

    # ---------------------------------------------------------------
    # Test 7: Pass-through — completely unknown term
    # ---------------------------------------------------------------
    result = mapper.map_term("xyzzy syndrome")
    print_result("Pass-through — 'xyzzy syndrome' (unknown)", result)
    assert result.match_type == "passthrough"
    assert result.confidence < 0.75
    assert result.is_high_confidence() == False

    # ---------------------------------------------------------------
    # Test 8: Case insensitivity
    # ---------------------------------------------------------------
    result = mapper.map_term("HIGH BLOOD PRESSURE")
    print_result("Case insensitive — 'HIGH BLOOD PRESSURE'", result)
    assert result.match_type == "exact"
    assert result.preferred_term == "hypertension"

    # ---------------------------------------------------------------
    # Test 9: Empty / whitespace input
    # ---------------------------------------------------------------
    result = mapper.map_term("")
    print_result("Empty input — ''", result)
    assert result.confidence == 0.0

    result = mapper.map_term("   ")
    print_result("Whitespace input — '   '", result)
    assert result.confidence == 0.0

    # ---------------------------------------------------------------
    # Test 10: extract_and_map — natural language query with multiple conditions
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  TEST: Natural language extraction")
    print(f"{'='*60}")

    multi = mapper.extract_and_map("I have diabetes and high blood pressure")
    print(f"  Input: 'I have diabetes and high blood pressure'")
    print(f"  Extracted {len(multi.results)} terms:")
    for r in multi.results:
        print(f"    '{r.original_term}' -> '{r.preferred_term}' ({r.match_type}, {r.confidence:.2f})")
    assert len(multi.results) == 2
    assert multi.all_preferred_terms[0] == "diabetes mellitus"
    assert multi.all_preferred_terms[1] == "hypertension"

    # ---------------------------------------------------------------
    # Test 11: extract_and_map — comma-separated
    # ---------------------------------------------------------------
    multi = mapper.extract_and_map("heart attack, stroke, asthma")
    print(f"\n  Input: 'heart attack, stroke, asthma'")
    print(f"  Extracted {len(multi.results)} terms:")
    for r in multi.results:
        print(f"    '{r.original_term}' -> '{r.preferred_term}' ({r.match_type}, {r.confidence:.2f})")
    assert len(multi.results) == 3

    # ---------------------------------------------------------------
    # Test 12: Low confidence detection
    # ---------------------------------------------------------------
    multi = mapper.extract_and_map("diabetes and xyzzy syndrome")
    low_conf = multi.low_confidence_terms
    print(f"\n  Input: 'diabetes and xyzzy syndrome'")
    print(f"  Low confidence terms: {[r.original_term for r in low_conf]}")
    assert len(low_conf) == 1
    assert low_conf[0].original_term == "xyzzy syndrome"

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ALL TESTS PASSED!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()