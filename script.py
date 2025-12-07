
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

	# Accept several possible video id column names
	video_cols = [c for c in df.columns if c.lower() in ('videoid', 'video_id', 'video', 'videoid')]
	if video_cols:
		video_col = video_cols[0]
	else:
		# fallback: try to find a column that contains 'video'
		candidates = [c for c in df.columns if 'video' in c.lower()]
		if candidates:
			video_col = candidates[0]
		else:
			print('Could not find a video id column in CSV. Columns available:')
			print(list(df.columns))
			sys.exit(1)

	# rating column
	rating_cols = [c for c in df.columns if c.lower() in ('rating', 'score')]
	if rating_cols:
		rating_col = rating_cols[0]
	else:
		print('Could not find a rating column in CSV. Columns available:')
		print(list(df.columns))
		sys.exit(1)

	# prepare outdir
	os.makedirs(outdir, exist_ok=True)

	# Coerce ratings to numeric when possible
	df['_rating_numeric'] = pd.to_numeric(df[rating_col], errors='coerce')
	df['_rating_is_numeric'] = ~df['_rating_numeric'].isna()

	videos = df[video_col].unique()

	# Sort videos by embedded numeric id if present (e.g. Video01 -> 1)
	def _extract_num(s):
		m = re.search(r"(\d+)", str(s))
		return int(m.group(1)) if m else float('inf')

	video_list = sorted(list(videos), key=lambda v: (_extract_num(v), str(v).lower()))
	print(f'Found {len(video_list)} unique videos; writing combined plot to {outdir}/')

	# Determine user column (if any)
	user_cols = [c for c in df.columns if c.lower() in ('uuid', 'user', 'user_id', 'participant', 'participant_id')]
	user_col = user_cols[0] if user_cols else None

	# Collect counts per video
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

	# Build and print summary table
	summary = []
	for i, vid in enumerate(video_list):
		counts = counts_per_video[i]
		total_numeric = int(counts.sum()) if not counts.empty else 0
		if total_numeric > 0:
			# weighted mean from counts (handles non-consecutive rating values)
			idx = counts.index.astype(float).to_numpy()
			vals = counts.values.astype(float)
			mean_rating = float((idx * vals).sum() / vals.sum())
			mean_rating = round(mean_rating, 3)
			rating_counts_str = ', '.join(f"{int(k)}:{v}" for k, v in counts.items())
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
					# prefer the last numeric answer by timestamp if available
					if 'timestamp' in user_sub.columns:
						user_sub['__ts'] = pd.to_datetime(user_sub['timestamp'], errors='coerce')
						user_numeric = user_sub[user_sub['_rating_is_numeric']].sort_values('__ts')
					else:
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
			expected = ''

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
		# abs difference computed from mean_rating vs expected
		summary_df['abs_diff'] = (summary_df['mean_rating'] - summary_df['expected_num']).abs().round(3)

		# Select and print table
		if args.user:
			# compute per-video abs diff for the user (NaN if user didn't answer)
			summary_df['user_rating_num'] = pd.to_numeric(summary_df['user_rating'], errors='coerce')
			summary_df['user_abs_diff'] = (summary_df['user_rating_num'] - summary_df['expected_num']).abs().round(3)
			# exclude mean_rating from the user-focused table
			out_df = summary_df[['video', 'user_rating_num', 'expected', 'user_abs_diff']]
			out_df = out_df.rename(columns={'user_rating_num': 'user_rating', 'user_abs_diff': 'abs_diff_user'})
			print('\nSummary table (user answers):')
			print(out_df.to_string(index=False))
		else:
			# Select only the requested columns: video, mean_rating, expected, abs_diff
			out_df = summary_df[['video', 'mean_rating', 'expected', 'abs_diff']]
			print('\nSummary table (means):')
			print(out_df.to_string(index=False))

		# If a user was specified, compute comparison between user and all people
		if args.user:
			# total using mean ratings only (all people)
			mean_diff = (summary_df['mean_rating'] - summary_df['expected_num']).abs().sum(skipna=True)
			try:
				mean_diff = float(mean_diff)
			except Exception:
				mean_diff = None

			# total for the user: use user's rating where present, otherwise fall back to mean (same as displayed)
			# we stored 'user_rating' and 'mean_rating'
			if 'user_rating' in summary_df.columns:
				user_series = summary_df['user_rating'].copy()
				# replace missing user ratings with mean_rating for fair comparison (as requested)
				user_series = user_series.fillna(summary_df['mean_rating'])
				user_diff = (user_series - summary_df['expected_num']).abs().sum(skipna=True)
				try:
					user_diff = float(user_diff)
				except Exception:
					user_diff = None
			else:
				user_diff = None

			# Print results
			if mean_diff is not None:
				print(f"Total abs_diff (all people, means): {round(mean_diff,3)}")
			else:
				print("Total abs_diff (all people, means): N/A")

			if user_diff is not None:
				print(f"Total abs_diff (user {args.user}): {round(user_diff,3)}")
			else:
				print(f"Total abs_diff (user {args.user}): N/A")

			if (mean_diff is not None) and (user_diff is not None):
				diff_val = user_diff - mean_diff
				print(f"Difference (user - all): {round(diff_val,3)}")
	except Exception as e:
		print('Failed to build/print summary table:', e)

	# Layout: choose number of columns, compute rows
	cols = 4
	n = len(video_list)
	rows = math.ceil(n / cols)

	fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, max(2.5 * rows, 4)))
	# flatten axes array for easy indexing
	axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

	for i, vid in enumerate(video_list):
		ax = axes_flat[i]
		counts = counts_per_video[i]
		if not counts.empty:
			x_vals = [str(v) for v in counts.index.tolist()]
			y_vals = counts.values.tolist()
			ax.bar(x_vals, y_vals, color='tab:blue')
			ax.set_xlabel('rating')
			ax.set_ylabel('count')
			# force integer y-axis ticks (step 1) since counts are integers
			ax.yaxis.set_major_locator(MaxNLocator(integer=True))
			# ensure y-axis starts at 0
			ax.set_ylim(bottom=0)
		else:
			ax.text(0.5, 0.5, 'No numeric ratings', ha='center', va='center')

		# show short title and note if any non-numeric were omitted
		note = f' ({non_numeric_counts[i]} non-numeric omitted)' if non_numeric_counts[i] > 0 else ''
		ax.set_title(f'{vid}{note}', fontsize=9)

	# Turn off any unused subplots
	for j in range(n, len(axes_flat)):
		axes_flat[j].axis('off')

	fig.suptitle(f'Rating distributions — {n} videos')
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])

	out_path = os.path.join(outdir, 'all_distributions.png')
	try:
		fig.savefig(out_path)
		print(f'Wrote {out_path}')
	except Exception as e:
		print(f'Failed to save combined plot: {e}')

	if args.show:
		plt.show()
	plt.close(fig)

	print('Done.')


if __name__ == '__main__':
	main()

