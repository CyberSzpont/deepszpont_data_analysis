
from __future__ import annotations

import argparse
import os
import sys
import math
from collections import Counter
import re

def safe_filename(s: str) -> str:
	return "".join(c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in s).strip().replace(' ', '_')

def main():
	parser = argparse.ArgumentParser(description='Plot ratings per video from a CSV file')
	parser.add_argument('--csv', '-c', default='db.csv', help='Path to CSV (default: db.csv)')
	parser.add_argument('--outdir', '-o', default='plots', help='Output directory for images')
	parser.add_argument('--show', action='store_true', help='Show plots interactively (requires GUI backend)')
	parser.add_argument('--user', '-u', help='User id (uuid) to display that user\'s answers; if provided, uses the user\'s rating for each video when available, otherwise uses the overall average')
	args = parser.parse_args()

	csv_path = args.csv
	outdir = args.outdir

	try:
		import pandas as pd
		import matplotlib.pyplot as plt
		from matplotlib.ticker import MaxNLocator
	except Exception as e:
		print('Missing required packages: please install with:')
		print('  python -m pip install pandas matplotlib')
		print(f'Detailed error: {e}')
		sys.exit(1)

	if not os.path.exists(csv_path):
		print(f'CSV file not found: {csv_path}')
		sys.exit(1)

	df = pd.read_csv(csv_path)

	video_cols = [c for c in df.columns if c.lower() in ('videoid', 'video_id', 'video', 'videoid')]
	if video_cols:
		video_col = video_cols[0]
	else:
		candidates = [c for c in df.columns if 'video' in c.lower()]
		if candidates:
			video_col = candidates[0]
		else:
			print('Could not find a video id column in CSV. Columns available:')
		print('Could not find a video id column in CSV. Columns available:')
		print(list(df.columns))
		sys.exit(1)

	rating_cols = [c for c in df.columns if c.lower() in ('rating', 'score')]
	if rating_cols:
		rating_col = rating_cols[0]
	else:
		print('Could not find a rating column in CSV. Columns available:')
		print(list(df.columns))
		sys.exit(1)

	os.makedirs(outdir, exist_ok=True)

	df['_rating_numeric'] = pd.to_numeric(df[rating_col], errors='coerce')
	df['_rating_is_numeric'] = df[rating_col].notna() & (pd.to_numeric(df[rating_col], errors='coerce').notna())
	videos = df[video_col].unique()

	def _extract_num(s):
		m = re.search(r'\d+', str(s))
		return float(m.group(0)) if m else None

	# Videos with numbers first, then those without
	videos_with_num = [v for v in videos if _extract_num(v) is not None]
	videos_without_num = [v for v in videos if _extract_num(v) is None]
	video_list = sorted(videos_with_num, key=lambda v: (_extract_num(v), str(v).lower())) + \
				 sorted(videos_without_num, key=lambda v: str(v).lower())
	print(f'Found {len(video_list)} unique videos; writing combined plot to {outdir}/')

	user_cols = [c for c in df.columns if c.lower() in ('uuid', 'user', 'user_id', 'participant', 'participant_id')]
	user_col = user_cols[0] if user_cols else None
	participant_count = int(df[user_col].dropna().nunique()) if user_col else None

	counts_per_video = []
	non_numeric_counts = []
	for vid in video_list:
		sub = df[df[video_col] == vid].copy()
		numeric = sub[sub['_rating_is_numeric']]
		counts = numeric['_rating_numeric'].value_counts().sort_index()
		counts_per_video.append(counts)
		non_numeric_counts.append(len(sub) - len(numeric))

	if not video_list:
		print('No videos found in CSV.')
		print('Done.')
		return

	summary = []
	for i, vid in enumerate(video_list):
		# weighted mean from counts (handles non-consecutive rating values)
		counts = counts_per_video[i]
		total_numeric = int(counts.sum()) if not counts.empty else 0
		if total_numeric > 0:
			idx = counts.index.astype(float).to_numpy()
			mean_rating = float(sum(k * v for k, v in counts.items()) / total_numeric)
			rating_counts_str = ', '.join([f"{k}: {v}" for k, v in counts.items()])
		else:
			mean_rating = float('nan')
			rating_counts_str = ''

		non_numeric = non_numeric_counts[i]

		# If a user id was provided, try to extract the user's numeric rating for this video
		user_val = None
		if args.user and user_col:
			try:
				user_sub = df[(df[video_col] == vid) & (df[user_col] == args.user)].copy()
				if not user_sub.empty:
					if 'timestamp' in user_sub.columns:
						user_sub = user_sub.sort_values('timestamp')
					user_numeric = user_sub[user_sub['_rating_is_numeric']]
					if not user_numeric.empty:
						user_val = float(user_numeric['_rating_numeric'].iloc[-1])
			except Exception:
				user_val = None

		# Determine expected value: 1 for Ai, 5 for Real (case-insensitive)
		name_lower = str(vid).lower()
		if 'ai' in name_lower:
			expected = 1
		elif 'real' in name_lower:
			expected = 5
		else:
			expected = None

		summary.append({
			'video': vid,
			'total_responses': int(total_numeric + non_numeric),
			'numeric_count': total_numeric,
			'non_numeric': non_numeric,
			'mean_rating': mean_rating,
			'rating_counts': rating_counts_str,
			'expected': expected,
			'user_rating': user_val,
		})

	try:
		summary_df = pd.DataFrame(summary)
		# Order by video name for readability
		summary_df = summary_df.sort_values('video')

		# Convert expected to numeric where possible, then compute absolute difference using means
		summary_df['expected_num'] = pd.to_numeric(summary_df['expected'], errors='coerce')
		summary_df['abs_diff'] = (summary_df['mean_rating'] - summary_df['expected_num']).abs().round(3)

		# Select and print table
		if args.user:
			# compute per-video abs diff for the user (NaN if user didn't answer)
			summary_df['user_rating_num'] = pd.to_numeric(summary_df['user_rating'], errors='coerce')
			summary_df['user_abs_diff'] = (summary_df['user_rating_num'] - summary_df['expected_num']).abs().round(3)
			out_df = summary_df[['video', 'user_rating_num', 'expected', 'user_abs_diff']]
			out_df = out_df.rename(columns={'user_rating_num': 'user_rating', 'user_abs_diff': 'abs_diff_user'})
			print('\nSummary table (user answers):')
			print(out_df.to_string(index=False))
		else:
			out_df = summary_df[['video', 'mean_rating', 'expected', 'abs_diff']]
			print('\nSummary table:')
			print(out_df.to_string(index=False))

		# If a user was specified, compute comparison between user and all people
		if args.user:
			mean_diff = (summary_df['abs_diff']).sum()
			user_diff = None
			if 'user_rating' in summary_df.columns:
				user_series = summary_df['user_rating'].copy()
				user_series = user_series.fillna(summary_df['mean_rating'])
				user_diff = (user_series - summary_df['expected_num']).abs().sum(skipna=True)
				try:
					user_diff = float(user_diff)
				except Exception:
					user_diff = None

			if mean_diff is not None:
				print(f"Total abs_diff (mean): {round(mean_diff,3)}")
			else:
				print("Total abs_diff (mean): N/A")

			if user_diff is not None:
				print(f"Total abs_diff (user {args.user}): {round(user_diff,3)}")
			else:
				print(f"Total abs_diff (user {args.user}): N/A")
		else:
			# Print total abs_diff for all videos
			total_abs_diff = (summary_df['abs_diff']).sum()
			if total_abs_diff is not None:
				print(f"Total abs_diff (all videos): {round(total_abs_diff,3)}")
			else:
				print("Total abs_diff (all videos): N/A")

	except Exception as e:
		print('Failed to build/print summary table:', e)

	cols = 4
	n = len(video_list)
	rows = math.ceil(n / cols)

	fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, max(2.5 * rows, 4)))
	axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

	for i, vid in enumerate(video_list):
		ax = axes_flat[i]
		counts = counts_per_video[i]
		if counts.empty:
			ax.text(0.5, 0.5, 'No numeric ratings', ha='center', va='center')
		else:
			ax.bar(counts.index.astype(float), counts.values, color='steelblue', edgecolor='black')
			ax.yaxis.set_major_locator(MaxNLocator(integer=True))
			ax.set_xlabel('Rating', fontsize=8)
			ax.set_ylabel('Count', fontsize=8)
			ax.set_xticks(range(1, 6))
			ax.set_xlim(0.4, 5.6)

		note = f' ({non_numeric_counts[i]} non-numeric omitted)' if non_numeric_counts[i] > 0 else ''
		ax.set_title(f'{vid}{note}', fontsize=9)

	for j in range(n, len(axes_flat)):
		axes_flat[j].axis('off')

	title = f'Rating distributions — {n} videos'
	if participant_count is not None:
		title += f' — {participant_count} participants'
	fig.suptitle(title)
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])

	out_path = os.path.join(outdir, 'all_distributions.png')
	try:
		fig.savefig(out_path, dpi=100, bbox_inches='tight')
		print(f'Saved: {out_path}')
	except Exception as e:
		print(f'Failed to save plot: {e}')

	if args.show:
		print('Showing plot interactively...')
		plt.show()

	print('Done.')

if __name__ == '__main__':
	main()
