r"""Fit the VIDRA GPN-Star transform on the ClinVar missense variants of a pipeline run.

Reads `output/variant` and `output/evidence_eva` of a run, fits `sigmoid(a + b * llr)` as the
probability that a ClinVar missense variant is pathogenic (see `pts.pyspark.vidra_utils.gpn_star`),
writes the fit and its diagnostics as JSON, and logs the YAML to paste into the `vidra_input`
step settings of `config.yaml`. One-off per variant-index release; re-run when the GPN-Star
scores or ClinVar change.

Usage, from `pts/`:

    uv run python tools/vidra_gpn_star_calibration.py \\
        --release-uri gs://open-targets-pipeline-runs/sz/platform-2612-4 \\
        --output /tmp/gpn_star_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from pts.pyspark.vidra_utils.gpn_star import calibrate
from pts.transformers.utils.dataset import scan_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--release-uri', required=True, help='run prefix holding output/variant and output/evidence_eva'
    )
    parser.add_argument('--variant', help='override: the variant index path')
    parser.add_argument('--evidence-eva', help='override: the EVA evidence path')
    parser.add_argument('--min-review', choices=['single', 'multiple'], default='single')
    parser.add_argument('--balanced', action='store_true', help='fit at 50:50 prevalence instead of ClinVar balance')
    parser.add_argument('--output', type=Path, required=True, help='JSON file for the fit and diagnostics')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    release = args.release_uri.rstrip('/')
    variant_path = args.variant or f'{release}/output/variant'
    eva_path = args.evidence_eva or f'{release}/output/evidence_eva'
    logger.info('variant index: {}', variant_path)
    logger.info('EVA evidence: {}', eva_path)

    calibration = calibrate(
        scan_dataset(variant_path),
        scan_dataset(eva_path),
        min_review=args.min_review,
        balanced=args.balanced,
    )
    fitted_on = datetime.now(UTC).strftime('%Y-%m-%d')
    report = {
        'fitted_on': fitted_on,
        'variant': variant_path,
        'evidence_eva': eva_path,
        **calibration.to_dict(),
    }
    args.output.write_text(json.dumps(report, indent=2))
    logger.info('wrote {}', args.output)
    for key, value in report.items():
        logger.info('{}: {}', key, value)

    logger.info(
        'config.yaml settings for vidra_input:\n'
        'gpn_star:\n'
        '  # as_gpnstar = 1 - sigmoid(a + b * llr). Fitted by pts/tools/vidra_gpn_star_calibration.py on {}\n'
        '  # ({}): {} pathogenic / {} benign ClinVar missense, AUROC {}, min_review={}, balanced={}\n'
        '  a: {:.4f}\n'
        '  b: {:.4f}',
        variant_path,
        fitted_on,
        calibration.n_pathogenic,
        calibration.n_benign,
        calibration.auroc_gpn_star,
        calibration.min_review,
        calibration.balanced,
        calibration.a,
        calibration.b,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
