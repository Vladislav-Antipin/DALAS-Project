#!/usr/bin/env python3
"""
Analyze label coverage in train-test split for temporal cutoff.

This script examines:
1. How many test data points have labels (targets)
2. How that compares to training data label coverage
3. The temporal cutoff split details

Based on the DALAS drug repurposing project notebooks.
"""

import pandas as pd
import pickle
import os

def load_data():
    """Load the prepared data from the project."""
    print("Loading data...")
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load merged data and dates
    merged_path = os.path.join(project_root, "data", "03-result", "merged_df.pkl")
    dates_path = os.path.join(project_root, "data", "02-result", "dates_df.pkl")
    
    with open(merged_path, "rb") as f:
        merged_df = pickle.load(f)
    
    with open(dates_path, "rb") as f:
        dates_df = pickle.load(f)
    
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Dates data shape: {dates_df.shape}")
    
    return merged_df, dates_df

def analyze_temporal_split(merged_df, dates_df):
    """Analyze the temporal split and label coverage."""
    
    # Merge with dates to get first_trial_date
    df = merged_df.merge(
        dates_df[["drug_id", "disease_id", "first_trial_date"]], 
        how="left", 
        on=["drug_id", "disease_id"]
    ).sort_values(by="first_trial_date").reset_index(drop=True)
    
    print(f"\nData after merging with dates: {df.shape}")
    print(f"Date range: {df['first_trial_date'].min()} to {df['first_trial_date'].max()}")
    
    # Determine split point (85% train, 15% test)
    n_samples = len(df)
    n_train = int(n_samples * 0.85)
    n_test = n_samples - n_train
    
    # Get the cutoff date
    cutoff_date = df["first_trial_date"].iloc[n_train - 1]  # Last training sample
    test_start_date = df["first_trial_date"].iloc[n_train]  # First test sample
    
    print(f"\n=== TEMPORAL SPLIT DETAILS ===")
    print(f"Total samples: {n_samples}")
    print(f"Training samples: {n_train} ({n_train/n_samples:.1%})")
    print(f"Test samples: {n_test} ({n_test/n_samples:.1%})")
    print(f"Cutoff date: {cutoff_date}")
    print(f"Test period starts: {test_start_date}")
    
    # Split the data
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    
    print(f"\nActual split:")
    print(f"Train: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, test_df, cutoff_date

def analyze_labels(train_df, test_df):
    """Analyze label coverage in train and test sets."""
    
    print(f"\n=== LABEL COVERAGE ANALYSIS ===")
    
    # Overall label statistics
    print(f"\nTraining set:")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Samples with labels: {train_df['success'].notna().sum()}")
    print(f"  Samples without labels: {train_df['success'].isna().sum()}")
    print(f"  Label coverage: {train_df['success'].notna().sum()/len(train_df):.1%}")
    
    if train_df['success'].notna().any():
        label_counts = train_df['success'].value_counts()
        print(f"  Success cases: {label_counts.get(True, 0)}")
        print(f"  Failure cases: {label_counts.get(False, 0)}")
        print(f"  Success rate: {label_counts.get(True, 0)/(label_counts.sum()):.1%}")
    
    print(f"\nTest set:")
    print(f"  Total samples: {len(test_df)}")
    print(f"  Samples with labels: {test_df['success'].notna().sum()}")
    print(f"  Samples without labels: {test_df['success'].isna().sum()}")
    print(f"  Label coverage: {test_df['success'].notna().sum()/len(test_df):.1%}")
    
    if test_df['success'].notna().any():
        label_counts = test_df['success'].value_counts()
        print(f"  Success cases: {label_counts.get(True, 0)}")
        print(f"  Failure cases: {label_counts.get(False, 0)}")
        print(f"  Success rate: {label_counts.get(True, 0)/(label_counts.sum()):.1%}")
    
    # Comparison
    train_coverage = train_df['success'].notna().sum()/len(train_df)
    test_coverage = test_df['success'].notna().sum()/len(test_df)
    
    print(f"\n=== COMPARISON ===")
    print(f"Label coverage difference: {test_coverage - train_coverage:+.1%}")
    print(f"Test coverage is {test_coverage/train_coverage:.1f}x train coverage")
    
    # Missing labels by time period in test set
    print(f"\n=== TEST SET LABELS BY TIME PERIOD ===")
    test_df['year'] = pd.to_datetime(test_df['first_trial_date'], errors='coerce').dt.year
    
    yearly_stats = test_df.groupby('year').agg({
        'success': ['count', lambda x: x.notna().sum(), lambda x: x.isna().sum()]
    }).round(1)
    
    yearly_stats.columns = ['Total', 'With Labels', 'Missing Labels']
    yearly_stats['Coverage %'] = (yearly_stats['With Labels'] / yearly_stats['Total'] * 100).round(1)
    
    print(yearly_stats)
    
    return train_coverage, test_coverage

def main():
    """Main analysis function."""
    print("=" * 60)
    print("DALAS PROJECT - TEMPORAL SPLIT LABEL COVERAGE ANALYSIS")
    print("=" * 60)
    
    # Load data
    merged_df, dates_df = load_data()
    
    # Analyze temporal split
    train_df, test_df, cutoff_date = analyze_temporal_split(merged_df, dates_df)
    
    # Analyze labels
    train_coverage, test_coverage = analyze_labels(train_df, test_df)
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"✓ Temporal cutoff date: {cutoff_date}")
    print(f"✓ Test set represents ~15% of data ({len(test_df)} samples)")
    print(f"✓ Training label coverage: {train_coverage:.1%}")
    print(f"✓ Test label coverage: {test_coverage:.1%}")
    
    if test_coverage < train_coverage:
        print(f"⚠️  Test set has {train_coverage - test_coverage:.1%} lower label coverage")
        print("   This suggests recent trials may not have completed/reported results yet")
    elif test_coverage == train_coverage:
        print("✓ Test and train sets have similar label coverage")
    else:
        print(f"📈 Test set has {test_coverage - train_coverage:.1%} higher label coverage")
    
    print(f"\n🎯 Key Finding: {test_df['success'].notna().sum()}/{len(test_df)} test samples have labels")

if __name__ == "__main__":
    main()
