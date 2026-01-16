"""
Depth to Groundwater (DTW) PR1 – Wells data audit & schema checks.

This script validates an input wells CSV against the canonical DTW schema by:
- Resolving canonical column names from common aliases
- Checking for required columns and null/malformed values
- Performing lightweight CRS range checks for NZTM2000 coordinates
- Writing a summary CSV and a detailed issues CSV (row id + issue type)

Usage:
    python wells_validation.py \\
        --input data/sample/Wells/Wells_and_Bores_-_All\\ \\(2\\).csv \\
        --output-dir reports/dtw_pr1
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


# Canonical DTW schema and known aliases present in the raw wells CSV
COLUMN_MAPPINGS: Dict[str, Iterable[str]] = {
    "well_id": ["well_id", "WELL_NO", "well_no"],
    "nztm_x": ["nztm_x", "NZTMX", "x"],
    "nztm_y": ["nztm_y", "NZTMY", "y"],
    "ground_surface_elev_m": ["ground_surface_elev_m", "GROUND_RL", "REFERENCE_RL"],
    "static_water_level_elev_m": [
        "static_water_level_elev_m",
        "INITIAL_SWL",
        "HIGHEST_WATER_LEVEL",
    ],
    "measurement_date": ["measurement_date", "DATE_DRILLED"],
    "well_depth_m": ["well_depth_m", "DEPTH", "drill_hole_depth"],
}

# Required vs optional fields for this audit
REQUIRED_FIELDS: Tuple[str, ...] = (
    "well_id",
    "nztm_x",
    "nztm_y",
    "ground_surface_elev_m",
    "static_water_level_elev_m",
)

WARNING_FIELDS: Tuple[str, ...] = ("measurement_date", "well_depth_m")

# NZTM2000 coordinate sanity bounds (meters)
CRS_BOUNDS = {
    "nztm_x": (1_000_000, 3_000_000),
    "nztm_y": (4_700_000, 6_500_000),
}


def resolve_columns(df_columns: Iterable[str]) -> Dict[str, str]:
    """Map canonical names to actual columns present in the dataframe."""
    lower_lookup = {col.lower(): col for col in df_columns}
    resolved: Dict[str, str] = {}
    for canonical, aliases in COLUMN_MAPPINGS.items():
        for alias in aliases:
            if alias.lower() in lower_lookup:
                resolved[canonical] = lower_lookup[alias.lower()]
                break
    return resolved


def coerce_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_issue(
    issues: List[Dict[str, str]],
    *,
    row_number: str | int,
    well_id: str,
    issue_type: str,
    severity: str,
    detail: str,
) -> None:
    issues.append(
        {
            "row_number": row_number,
            "well_id": well_id,
            "issue_type": issue_type,
            "severity": severity,
            "detail": detail,
        }
    )


def validate(df: pd.DataFrame) -> tuple[List[Dict[str, str]], Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    resolved = resolve_columns(df.columns)
    missing_required = [field for field in REQUIRED_FIELDS if field not in resolved]

    for field in missing_required:
        add_issue(
            issues,
            row_number="",
            well_id="",
            issue_type="missing_required_column",
            severity="error",
            detail=f"`{field}` not present in input",
        )

    for idx, row in df.iterrows():
        row_number = idx + 2  # include header row in CSV line numbering
        well_id = row[resolved["well_id"]] if "well_id" in resolved else ""

        # Required value checks
        for field in REQUIRED_FIELDS:
            col_name = resolved.get(field)
            if not col_name:
                continue

            raw_value = row[col_name]
            is_missing = pd.isna(raw_value) or (
                isinstance(raw_value, str) and not raw_value.strip()
            )

            if is_missing:
                add_issue(
                    issues,
                    row_number=row_number,
                    well_id=well_id,
                    issue_type="missing_required_value",
                    severity="error",
                    detail=f"`{field}` is missing",
                )
                continue

            if field in CRS_BOUNDS:
                numeric = coerce_float(raw_value)
                if numeric is None:
                    add_issue(
                        issues,
                        row_number=row_number,
                        well_id=well_id,
                        issue_type="non_numeric_coordinate",
                        severity="error",
                        detail=f"`{field}` value '{raw_value}' is not numeric",
                    )
                    continue

                lower, upper = CRS_BOUNDS[field]
                if not (lower <= numeric <= upper):
                    add_issue(
                        issues,
                        row_number=row_number,
                        well_id=well_id,
                        issue_type="invalid_coordinate_range",
                        severity="error",
                        detail=f"`{field}`={numeric} outside NZTM2000 bounds [{lower}, {upper}]",
                    )

        # Optional warning-only checks
        for field in WARNING_FIELDS:
            col_name = resolved.get(field)
            if not col_name:
                continue

            raw_value = row[col_name]
            is_missing = pd.isna(raw_value) or (
                isinstance(raw_value, str) and not raw_value.strip()
            )

            if is_missing:
                add_issue(
                    issues,
                    row_number=row_number,
                    well_id=well_id,
                    issue_type="missing_optional_value",
                    severity="warning",
                    detail=f"`{field}` is missing",
                )

    return issues, resolved


def write_outputs(
    issues: List[Dict[str, str]],
    resolved: Dict[str, str],
    output_dir: Path,
    total_rows: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    issues_path = output_dir / "wells_validation_issues.csv"
    pd.DataFrame(issues).to_csv(issues_path, index=False)

    issue_counter = Counter(issue["issue_type"] for issue in issues)
    severity_counter = Counter(issue["severity"] for issue in issues)
    rows_with_issue = {
        issue["row_number"] for issue in issues if isinstance(issue["row_number"], int)
    }

    summary_rows = [
        {"metric": "total_rows", "value": total_rows},
        {"metric": "rows_with_issue", "value": len(rows_with_issue)},
        {"metric": "rows_without_issue", "value": total_rows - len(rows_with_issue)},
    ]

    for field in REQUIRED_FIELDS + WARNING_FIELDS:
        summary_rows.append(
            {
                "metric": f"resolved.{field}",
                "value": resolved.get(field, ""),
            }
        )

    for issue_type, count in issue_counter.items():
        summary_rows.append({"metric": f"issues.{issue_type}", "value": count})

    for severity, count in severity_counter.items():
        summary_rows.append({"metric": f"severity.{severity}", "value": count})

    summary_path = output_dir / "wells_validation_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Wrote issue log to {issues_path}")
    print(f"Wrote summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate wells CSV against DTW PR1 schema requirements."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sample/Wells/Wells_and_Bores_-_All (2).csv"),
        help="Path to wells CSV input",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dtw_pr1"),
        help="Directory for validation outputs",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    issues, resolved = validate(df)
    write_outputs(issues, resolved, args.output_dir, total_rows=len(df))


if __name__ == "__main__":
    main()
