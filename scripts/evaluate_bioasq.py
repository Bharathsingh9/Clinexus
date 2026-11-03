#!/usr/bin/env python3
"""CLI script for running BioASQ evaluation harness.

Specification Section 9, 10, 12, Appendix A11-A13 (Req ID: EVL-1..6, CLI).
"""

import argparse
import json
import logging
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.config import settings
from backend.app.core.logging_config import setup_logging
from backend.app.evaluation.harness import EvaluationHarness

setup_logging()
logger = logging.getLogger("clinexus.evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BioASQ evaluation harness.")
    parser.add_argument(
        "--set",
        type=str,
        required=True,
        choices=["smoke30", "retrieval_dev", "retrieval_eval", "e2e_sample"],
        help="Evaluation set to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["retrieval_only", "baseline", "full", "full_online"],
        help="Evaluation configuration name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned question counts and estimated max LLM calls without calling anything.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint, skipping already completed questions.",
    )
    parser.add_argument(
        "--allow-rerun",
        type=str,
        default=None,
        help="Reason for rerunning a previously completed eval-split configuration.",
    )
    parser.add_argument(
        "--eval-max-llm-calls",
        type=int,
        default=None,
        help="Override for EVAL_MAX_LLM_CALLS setting.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory where evaluation JSON files and checkpoints are written.",
    )

    args = parser.parse_args()

    # Pass command-line override into settings if provided
    if args.eval_max_llm_calls is not None:
        settings.EVAL_MAX_LLM_CALLS = args.eval_max_llm_calls

    if args.config == "full":
        settings.EXTERNAL_MODE = "offline"
    elif args.config == "full_online":
        settings.EXTERNAL_MODE = "online"

    harness = EvaluationHarness(
        settings_obj=settings,
        output_dir=args.output_dir,
    )

    if args.dry_run:
        dry_res = harness.dry_run(
            set_name=args.set,
            config_name=args.config,
            limit=args.limit,
            eval_max_llm_calls=args.eval_max_llm_calls,
        )
        print("\n=== BioASQ Evaluation Dry Run Summary ===")
        print(f"Set:                     {dry_res['set_name']}")
        print(f"Config:                  {dry_res['config_name']}")
        print(f"Planned Questions:       {dry_res['planned_question_count']}")
        print(f"Questions by Type:       {dry_res['question_counts_by_type']}")
        print(f"Calls/Q (Worst-Case):    {dry_res['estimated_calls_per_question_worst_case']}")
        print(f"Estimated Call Ceiling:  {dry_res['estimated_call_ceiling']}")
        print(f"Effective Call Cap:      {dry_res['effective_eval_max_llm_calls_cap']}")
        print(f"LLM Required:            {dry_res['is_llm_required']}")
        print("=========================================\n")
        sys.exit(0)

    try:
        if args.config == "retrieval_only":
            res = harness.run_retrieval_only(
                set_name=args.set,
                limit=args.limit,
                resume=args.resume,
            )
        else:
            res = harness.run_e2e(
                set_name=args.set,
                config_name=args.config,
                limit=args.limit,
                eval_max_llm_calls=args.eval_max_llm_calls,
                allow_rerun=args.allow_rerun,
                resume=args.resume,
            )

        print("\n=== Evaluation Run Completed ===")
        print(f"Run ID:         {res['run_id']}")
        print(f"Status:         {res['status']}")
        print(f"Labels:         {', '.join(res.get('labels', []))}")
        print(f"Record Counts:  {res['record_counts']}")
        if "aggregated_metrics" in res:
            print("Aggregated Metrics (Overall):")
            for k, v in res["aggregated_metrics"].get("overall", {}).items():
                if isinstance(v, dict):
                    print(f"  {k}: {v['value']:.4f} (n={v['count']})")
                else:
                    print(f"  {k}: {v}")
        print(f"Total LLM Calls: {res.get('total_llm_calls', 0)}")
        print(f"Total In Tokens: {res.get('total_input_tokens', 0)}")
        print(f"Total Out Tokens: {res.get('total_output_tokens', 0)}")
        print("================================\n")

    except Exception as exc:
        print(f"\nEvaluation failed with error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
