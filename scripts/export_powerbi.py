"""Run from the repository root; no ML downloads or database writes."""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.powerbi_export import ROOT, export_powerbi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / 'database/archivemind.db')
    parser.add_argument('--output', type=Path, default=ROOT / 'powerbi/exports')
    args = parser.parse_args()
    try:
        summary = export_powerbi(args.database, args.output)
    except (ValueError, OSError, sqlite3.Error, ImportError) as exc:
        parser.exit(1, f'Export failed: {exc}\nCheck database schema/path, output permissions, and install numpy for vector exports.\n')
    for name, count in summary['row_counts'].items():
        print(f'{args.output / (name + ".csv")}: {count} rows')
    for warning in summary['warnings']:
        print(f'NOTE: {warning}')


if __name__ == '__main__':
    main()
