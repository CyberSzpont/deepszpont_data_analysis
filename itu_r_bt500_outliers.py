"""
ITU-R BT.500 Outlier Detection Script

This script implements outlier detection based on ITU-R BT.500 methodology
for subjective video quality assessment. It identifies outlier subjects/observers
using statistical methods defined in the recommendation.

ITU-R BT.500 Methods:
1. 2-sigma rule: Subjects whose scores deviate > 2 standard deviations
2. Beta2 (kurtosis) screening: Checks for abnormal score distributions
3. Pearson correlation: Checks correlation between subject and mean scores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
from typing import Dict, List, Tuple

# Configuration - Stricter thresholds for true outliers only
SIGMA_THRESHOLD = 2.5  # Standard deviation threshold for outlier detection (more stringent)
CORRELATION_THRESHOLD = 0.5  # Minimum correlation threshold (lower = stricter)
KURTOSIS_THRESHOLD = 3.0  # Kurtosis threshold for distribution checks (higher = stricter)
MIN_RATINGS_REQUIRED = 5  # Minimum ratings to consider a subject for analysis


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the CSV data."""
    if not os.path.exists(csv_path):
        print(f'Error: CSV file not found: {csv_path}')
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} ratings from {csv_path}")
    return df


def identify_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Identify the relevant columns in the dataframe."""
    # Find video column
    video_cols = [c for c in df.columns if c.lower() in ('videoid', 'video_id', 'video')]
    video_col = video_cols[0] if video_cols else None
    
    # Find rating column
    rating_cols = [c for c in df.columns if c.lower() in ('rating', 'score')]
    rating_col = rating_cols[0] if rating_cols else None
    
    # Find subject/user column
    subject_cols = [c for c in df.columns if c.lower() in ('uuid', 'subject', 'user', 'observer')]
    subject_col = subject_cols[0] if subject_cols else None
    
    if not all([video_col, rating_col, subject_col]):
        print(f"Error: Could not identify required columns")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"Using columns - Video: {video_col}, Rating: {rating_col}, Subject: {subject_col}")
    return video_col, rating_col, subject_col


def compute_subject_statistics(df: pd.DataFrame, video_col: str, rating_col: str, 
                               subject_col: str) -> pd.DataFrame:
    """
    Compute statistics for each subject according to ITU-R BT.500.
    
    Returns a DataFrame with subject-level statistics including:
    - Number of ratings
    - Mean rating
    - Standard deviation
    - Correlation with overall mean
    - Kurtosis
    """
    stats_list = []
    
    # Calculate overall mean for each video
    video_means = df.groupby(video_col)[rating_col].mean()
    
    subjects = df[subject_col].unique()
    print(f"\nAnalyzing {len(subjects)} subjects...")
    
    for subject in subjects:
        subject_data = df[df[subject_col] == subject]
        
        if len(subject_data) < MIN_RATINGS_REQUIRED:  # Skip subjects with too few ratings
            continue
        
        # Get subject's ratings and corresponding video means
        subject_ratings = []
        video_mean_ratings = []
        
        for _, row in subject_data.iterrows():
            vid = row[video_col]
            rating = row[rating_col]
            if vid in video_means.index:
                subject_ratings.append(rating)
                video_mean_ratings.append(video_means[vid])
        
        if len(subject_ratings) < 3:
            continue
        
        # Calculate statistics
        mean_rating = np.mean(subject_ratings)
        std_rating = np.std(subject_ratings, ddof=1)
        
        # Pearson correlation with overall means
        if len(subject_ratings) > 1 and np.std(subject_ratings) > 0:
            correlation, p_value = stats.pearsonr(subject_ratings, video_mean_ratings)
        else:
            correlation, p_value = 0, 1
        
        # Kurtosis (using Fisher's definition, where normal distribution has kurtosis=0)
        kurtosis = stats.kurtosis(subject_ratings, fisher=True)
        
        # Calculate deviation from mean (for 2-sigma rule)
        deviations = []
        for vid in subject_data[video_col]:
            if vid in video_means.index:
                subject_rating = subject_data[subject_data[video_col] == vid][rating_col].values[0]
                mean_rating_vid = video_means[vid]
                deviations.append(abs(subject_rating - mean_rating_vid))
        
        mean_absolute_deviation = np.mean(deviations) if deviations else 0
        
        stats_list.append({
            'subject': subject,
            'n_ratings': len(subject_ratings),
            'mean_rating': mean_rating,
            'std_rating': std_rating,
            'correlation': correlation,
            'correlation_pvalue': p_value,
            'kurtosis': kurtosis,
            'mean_abs_deviation': mean_absolute_deviation
        })
    
    return pd.DataFrame(stats_list)


def detect_outliers_2sigma(df: pd.DataFrame, video_col: str, rating_col: str, 
                           subject_col: str) -> Dict[str, List]:
    """
    Detect outliers using the 2-sigma rule from ITU-R BT.500.
    
    For each video, identify ratings that deviate more than 2 standard deviations
    from the mean.
    """
    outliers = []
    
    for video in df[video_col].unique():
        video_data = df[df[video_col] == video]
        ratings = video_data[rating_col].values
        
        if len(ratings) < 3:
            continue
        
        mean = np.mean(ratings)
        std = np.std(ratings, ddof=1)
        
        if std == 0:
            continue
        
        threshold = SIGMA_THRESHOLD * std
        
        for _, row in video_data.iterrows():
            rating = row[rating_col]
            deviation = abs(rating - mean)
            
            if deviation > threshold:
                outliers.append({
                    'video': video,
                    'subject': row[subject_col],
                    'rating': rating,
                    'video_mean': mean,
                    'video_std': std,
                    'deviation': deviation,
                    'n_sigma': deviation / std if std > 0 else 0
                })
    
    return pd.DataFrame(outliers) if outliers else pd.DataFrame()


def identify_outlier_subjects(subject_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Identify outlier subjects based on ITU-R BT.500 criteria.
    Uses stricter logic requiring MULTIPLE criteria to be met for true outliers.
    
    Criteria:
    1. Low correlation with mean scores
    2. Abnormal score distributions (high kurtosis)
    3. Excessive deviations from mean
    """
    outlier_subjects = subject_stats.copy()
    
    # Flag subjects based on different criteria
    outlier_subjects['is_low_correlation'] = outlier_subjects['correlation'] < CORRELATION_THRESHOLD
    outlier_subjects['is_high_kurtosis'] = abs(outlier_subjects['kurtosis']) > KURTOSIS_THRESHOLD
    
    # Calculate threshold for mean absolute deviation (using stricter sigma approach)
    mad_mean = outlier_subjects['mean_abs_deviation'].mean()
    mad_std = outlier_subjects['mean_abs_deviation'].std()
    mad_threshold = mad_mean + SIGMA_THRESHOLD * mad_std
    outlier_subjects['is_high_deviation'] = outlier_subjects['mean_abs_deviation'] > mad_threshold
    
    # Calculate outlier score (0-3 based on how many criteria are met)
    outlier_subjects['outlier_score'] = (
        outlier_subjects['is_low_correlation'].astype(int) +
        outlier_subjects['is_high_kurtosis'].astype(int) +
        outlier_subjects['is_high_deviation'].astype(int)
    )
    
    # Only flag as outlier if MULTIPLE criteria are met (score >= 2)
    # This ensures we only catch truly problematic subjects
    outlier_subjects['is_outlier'] = outlier_subjects['outlier_score'] >= 2
    
    # Additional check: if correlation is extremely low (< 0.3), flag even with single criterion
    outlier_subjects.loc[
        (outlier_subjects['correlation'] < 0.3) & (outlier_subjects['correlation'] > -0.3),
        'is_outlier'
    ] = True
    
    return outlier_subjects


def generate_report(subject_stats: pd.DataFrame, outlier_ratings: pd.DataFrame, 
                   outlier_subjects: pd.DataFrame, output_dir: str):
    """Generate visualizations and text reports."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Print text report
    print("\n" + "="*80)
    print("ITU-R BT.500 OUTLIER DETECTION REPORT")
    print("="*80)
    
    print(f"\nTotal subjects analyzed: {len(subject_stats)}")
    print(f"Subjects flagged as outliers: {outlier_subjects['is_outlier'].sum()}")
    print(f"Individual outlier ratings found: {len(outlier_ratings)}")
    
    if len(outlier_subjects[outlier_subjects['is_outlier']]) > 0:
        print("\n" + "-"*80)
        print("OUTLIER SUBJECTS:")
        print("-"*80)
        
        outlier_list = outlier_subjects[outlier_subjects['is_outlier']].sort_values(
            'correlation', ascending=True
        )
        
        for _, subject in outlier_list.iterrows():
            print(f"\nSubject: {subject['subject']}")
            print(f"  Outlier Score: {subject['outlier_score']:.0f}/3 ⚠️")
            print(f"  Ratings count: {subject['n_ratings']}")
            print(f"  Mean rating: {subject['mean_rating']:.2f}")
            print(f"  Correlation: {subject['correlation']:.3f} {'❌' if subject['is_low_correlation'] else '✓'}")
            print(f"  Kurtosis: {subject['kurtosis']:.3f} {'❌' if subject['is_high_kurtosis'] else '✓'}")
            print(f"  Mean deviation: {subject['mean_abs_deviation']:.3f} {'❌' if subject['is_high_deviation'] else '✓'}")
    
    # Save detailed CSV reports
    subject_stats.to_csv(os.path.join(output_dir, 'subject_statistics.csv'), index=False)
    outlier_subjects.to_csv(os.path.join(output_dir, 'outlier_subjects.csv'), index=False)
    
    if len(outlier_ratings) > 0:
        outlier_ratings.to_csv(os.path.join(output_dir, 'outlier_ratings.csv'), index=False)
    
    print(f"\nDetailed reports saved to: {output_dir}/")
    
    # Generate visualizations
    generate_visualizations(subject_stats, outlier_subjects, output_dir)


def generate_visualizations(subject_stats: pd.DataFrame, outlier_subjects: pd.DataFrame, 
                           output_dir: str):
    """Generate visualization plots."""
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Correlation distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    ax = axes[0, 0]
    ax.hist(subject_stats['correlation'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(CORRELATION_THRESHOLD, color='red', linestyle='--', 
               label=f'Threshold ({CORRELATION_THRESHOLD})')
    ax.set_xlabel('Pearson Correlation with Mean')
    ax.set_ylabel('Number of Subjects')
    ax.set_title('Distribution of Subject Correlations')
    ax.legend()
    
    # 2. Kurtosis distribution
    ax = axes[0, 1]
    ax.hist(subject_stats['kurtosis'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(KURTOSIS_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({KURTOSIS_THRESHOLD})')
    ax.axvline(-KURTOSIS_THRESHOLD, color='red', linestyle='--')
    ax.set_xlabel('Kurtosis (Fisher)')
    ax.set_ylabel('Number of Subjects')
    ax.set_title('Distribution of Rating Kurtosis')
    ax.legend()
    
    # 3. Mean absolute deviation
    ax = axes[1, 0]
    mad_mean = subject_stats['mean_abs_deviation'].mean()
    mad_std = subject_stats['mean_abs_deviation'].std()
    mad_threshold = mad_mean + SIGMA_THRESHOLD * mad_std
    
    ax.hist(subject_stats['mean_abs_deviation'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(mad_threshold, color='red', linestyle='--', 
               label=f'2σ Threshold ({mad_threshold:.2f})')
    ax.set_xlabel('Mean Absolute Deviation from Video Means')
    ax.set_ylabel('Number of Subjects')
    ax.set_title('Distribution of Subject Deviations')
    ax.legend()
    
    # 4. Scatter: Correlation vs Kurtosis
    ax = axes[1, 1]
    outlier_mask = outlier_subjects['is_outlier']
    ax.scatter(subject_stats[~outlier_mask]['correlation'], 
              subject_stats[~outlier_mask]['kurtosis'],
              alpha=0.6, label='Normal', s=50)
    ax.scatter(subject_stats[outlier_mask]['correlation'], 
              subject_stats[outlier_mask]['kurtosis'],
              color='red', alpha=0.8, label='Outlier', s=50, marker='x')
    ax.axvline(CORRELATION_THRESHOLD, color='red', linestyle='--', alpha=0.5)
    ax.axhline(KURTOSIS_THRESHOLD, color='red', linestyle='--', alpha=0.5)
    ax.axhline(-KURTOSIS_THRESHOLD, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Correlation with Mean')
    ax.set_ylabel('Kurtosis')
    ax.set_title('Subject Classification (Correlation vs Kurtosis)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {output_dir}/outlier_analysis.png")
    plt.close()
    
    # 5. Subject reliability chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    subject_stats_sorted = subject_stats.sort_values('correlation', ascending=False)
    outlier_flags = outlier_subjects.set_index('subject').loc[subject_stats_sorted['subject']]['is_outlier'].values
    
    colors = ['red' if flag else 'green' for flag in outlier_flags]
    
    ax.barh(range(len(subject_stats_sorted)), subject_stats_sorted['correlation'], color=colors, alpha=0.7)
    ax.axvline(CORRELATION_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({CORRELATION_THRESHOLD})')
    ax.set_xlabel('Correlation with Mean Scores')
    ax.set_ylabel('Subject (sorted by correlation)')
    ax.set_title('Subject Reliability (ITU-R BT.500)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_reliability.png'), dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {output_dir}/subject_reliability.png")
    plt.close()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ITU-R BT.500 Outlier Detection for Subjective Video Quality Assessment'
    )
    parser.add_argument('--csv', '-c', default='db.csv', 
                       help='Path to CSV file (default: db.csv)')
    parser.add_argument('--outdir', '-o', default='outliers_report', 
                       help='Output directory for reports (default: outliers_report)')
    parser.add_argument('--sigma', '-s', type=float, default=2.5,
                       help='Sigma threshold for outlier detection (default: 2.5, stricter)')
    parser.add_argument('--correlation', '-r', type=float, default=0.5,
                       help='Minimum correlation threshold (default: 0.5, stricter)')
    parser.add_argument('--kurtosis', '-k', type=float, default=3.0,
                       help='Kurtosis threshold (default: 3.0, stricter)')
    parser.add_argument('--min-ratings', '-m', type=int, default=5,
                       help='Minimum ratings per subject for analysis (default: 5)')
    
    args = parser.parse_args()
    
    # Update global thresholds
    global SIGMA_THRESHOLD, CORRELATION_THRESHOLD, KURTOSIS_THRESHOLD, MIN_RATINGS_REQUIRED
    SIGMA_THRESHOLD = args.sigma
    CORRELATION_THRESHOLD = args.correlation
    KURTOSIS_THRESHOLD = args.kurtosis
    MIN_RATINGS_REQUIRED = args.min_ratings
    
    # Load data
    df = load_data(args.csv)
    
    # Identify columns
    video_col, rating_col, subject_col = identify_columns(df)
    
    # Compute subject statistics
    subject_stats = compute_subject_statistics(df, video_col, rating_col, subject_col)
    
    if len(subject_stats) == 0:
        print("Error: No subjects found with sufficient ratings for analysis.")
        sys.exit(1)
    
    # Detect outliers using 2-sigma rule
    outlier_ratings = detect_outliers_2sigma(df, video_col, rating_col, subject_col)
    
    # Identify outlier subjects
    outlier_subjects = identify_outlier_subjects(subject_stats)
    
    # Generate report
    generate_report(subject_stats, outlier_ratings, outlier_subjects, args.outdir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
