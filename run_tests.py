"""
Run all VISTA 2.0 tests and save results to test_results.txt
"""
import sys
import io
import unittest
import os

# Use a file-based stream to avoid Windows cp1252 encoding issues
results_stream = io.open("test_results.txt", "w", encoding="utf-8")

results_summary = {}

def run_suite(label, module_name):
    results_stream.write(f"\n{'='*60}\n  {label}\n{'='*60}\n")
    results_stream.flush()
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromName(module_name)
    runner = unittest.TextTestRunner(stream=results_stream, verbosity=2)
    result = runner.run(suite)
    results_summary[label] = {
        'ran': result.testsRun,
        'passed': result.testsRun - len(result.failures) - len(result.errors),
        'failed': len(result.failures),
        'errors': len(result.errors),
        'failures': [str(f[1]) for f in result.failures],
        'error_list': [str(e[1]) for e in result.errors],
    }

try:
    run_suite("UNIT TESTS", "test_unit")
    run_suite("INTEGRATION TESTS", "test_integration")
    run_suite("SYSTEM AND PERFORMANCE TESTS", "test_system_perf")
except Exception as ex:
    results_stream.write(f"\nFATAL ERROR DURING TEST RUN: {ex}\n")
finally:
    results_stream.write("\n\n===== SUMMARY =====\n")
    for label, r in results_summary.items():
        status = "ALL PASS" if r['failed'] == 0 and r['errors'] == 0 else "HAS FAILURES/ERRORS"
        results_stream.write(
            f"[{status}] {label}: ran={r['ran']} passed={r['passed']} "
            f"failed={r['failed']} errors={r['errors']}\n"
        )
        if r['failures']:
            for fl in r['failures']:
                results_stream.write(f"  FAIL: {fl[:300]}\n")
        if r['error_list']:
            for er in r['error_list']:
                results_stream.write(f"  ERROR: {er[:300]}\n")
    results_stream.close()

print("Tests complete. Results saved to test_results.txt")
