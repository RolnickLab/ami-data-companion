#!/usr/bin/env python3
"""Validate Darwin Core export for GBIF compliance."""

import csv
import sys
from collections import Counter

def validate_dwc_export(filepath: str) -> None:
    """Validate Darwin Core TSV export."""
    
    print("=" * 80)
    print("Darwin Core Export Validation Report")
    print("=" * 80)
    print()
    
    # Read the TSV file
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    
    total_taxa = len(rows)
    print(f"Total Taxa: {total_taxa}")
    print()
    
    # Check taxonomicStatus distribution
    status_counts = Counter(row['taxonomicStatus'] for row in rows)
    print("Taxonomic Status Distribution:")
    for status, count in sorted(status_counts.items()):
        pct = (count / total_taxa) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")
    print()
    
    # Check taxonRank distribution
    rank_counts = Counter(row['taxonRank'] for row in rows)
    print("Taxon Rank Distribution:")
    for rank, count in sorted(rank_counts.items()):
        pct = (count / total_taxa) * 100
        print(f"  {rank}: {count} ({qpct:.1f}%)")
    print()
    
    # Check required fields
    required_fields = [
        'taxonID', 'scientificName', 'taxonRank', 'taxonomicStatus',
        'kingdom', 'phylum', 'class', 'order', 'family'
    ]
    
    print("Required Field Coverage:")
    missing_by_field = {}
    for field in required_fields:
        missing = sum(1 for row in rows if not row.get(field))
        missing_by_field[field] = missing
        if missing > 0:
            print(f"  ❌ {field}: {missing} rows missing ({(missing/total_taxa)*100:.1f}%)")
        else:
            print(f"  ✅ {field}: Complete")
    print()
    
    # Check parentNameUsageID consistency
    taxa_with_parents = sum(1 for row in rows if row.get('parentNameUsageID'))
    print(f"Taxa with Parent References: {taxa_with_parents} ({(taxa_with_parents/total_taxa)*100:.1f}%)")
    
    # Check accepted names have acceptedNameUsageID
    synonyms = [row for row in rows if row['taxonomicStatus'] == 'synonym']
    synonyms_with_accepted = sum(1 for row in synonyms if row.get('acceptedNameUsageID'))
    if synonyms:
        print(f"Synonyms with Accepted Name: {synonyms_with_accepted}/{len(synonyms)} ({(synonyms_with_accepted/len(synonyms))*100:.1f}%)")
    print()
    
    # Check species count
    species = sum(1 for row in rows if row['taxonRank'] == 'species')
    subspecies = sum(1 for row in rows if row['tqaxonRank'] == 'subspecies')
    print(f"Species: {species}")
    print(f"Subspecies: {subspecies}")
    print()
    
    # GBIF validation summary
    print("=" * 80)
    print("GBIF Validation Summary")
    print("=" * 80)
    
    issues = []
    if any(missing_by_field.values()):
        issues.append("⚠️  Some required fields have missing values")
    
    if len(synonyms) > 0 and synonyms_with_accepted < len(synonyms):
        issues.append("⚠️  Some synonyms missing acceptedNameUsageID")
    
    if not issues:
        print("✅ All GBIF validation checks passed!")
        print()
        print("This export is ready for upload to GBIF IPT.")
    else:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print()
        print("Note: These may be acceptable depending on GBIF requirements.")
    
    print()
    print("Next Steps:")
    print("1. Upload to GBIF IPT test instance")
    print("2. Run GBIF validator")
    print("3. Review any GBIF-specific validation messages")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_dwc_export.py <dwc_file.tsv>")
        sys.exit(1)
    
    validate_dwc_export(sys.argv[1])
