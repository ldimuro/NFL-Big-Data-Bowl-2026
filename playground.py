# # # Get first 5 interceptions
# interceptions = supplementary_data[supplementary_data['pass_result'] == 'IN']
# completions = supplementary_data[supplementary_data['pass_result'] == 'C']

# # trajectories = []
# # for idx, play in interceptions.iterrows():
# #     output_play = tracking_data_output[
# #         (tracking_data_output['game_id'] == play['game_id']) & 
# #         (tracking_data_output['play_id'] == play['play_id'])
# #     ]
    
# #     # Get ball_land
# #     ball_land = tracking_data_input[
# #         (tracking_data_input['game_id'] == play['game_id']) & 
# #         (tracking_data_input['play_id'] == play['play_id'])
# #     ][['ball_land_x', 'ball_land_y']].iloc[0]

# #     receiver_id = tracking_data_input[
# #         (tracking_data_input['game_id'] == play['game_id']) & 
# #         (tracking_data_input['play_id'] == play['play_id']) &
# #         (tracking_data_input['player_role'] == 'Targeted Receiver')
# #     ].iloc[0]['nfl_id']

# #     print('RECEIVER:', receiver_id)
    
# #     # Find intercepting defender (closest to ball_land at end)
# #     final_frame = output_play[output_play['frame_id'] == output_play['frame_id'].max()]
# #     print('FINAL FRAME:', final_frame)
# #     defenders = final_frame[final_frame['nfl_id'] != receiver_id]
    
# #     if defenders.empty:
# #         continue
    
# #     defenders['dist_to_ball'] = np.sqrt(
# #         (defenders['x'] - ball_land['ball_land_x'])**2 + 
# #         (defenders['y'] - ball_land['ball_land_y'])**2
# #     )
    
# #     int_defender_id = defenders.loc[defenders['dist_to_ball'].idxmin(), 'nfl_id']
    
# #     # Extract trajectory
# #     traj = output_play[output_play['nfl_id'] == int_defender_id][['frame_id', 'x', 'y']].values
    
# #     trajectories.append(traj)
# #     print(f"Play {play['play_id']}: Extracted {len(traj)} frames")

# # # Quick visualization
# # fig, axes = plt.subplots(1, 5, figsize=(20, 4))
# # for i, traj in enumerate(trajectories):
# #     axes[i].plot(traj[:, 1], traj[:, 2], 'r-', linewidth=2)
# #     axes[i].scatter(traj[0, 1], traj[0, 2], c='green', s=100, label='Start')
# #     axes[i].scatter(traj[-1, 1], traj[-1, 2], c='red', s=100, label='INT')
# #     axes[i].set_title(f'INT {i+1}')
# #     axes[i].legend()
# #     axes[i].grid(True, alpha=0.3)
# # plt.tight_layout()
# # plt.savefig('quick_test_trajectories_2.png')
# # # plt.show()

# # print(f"✅ Successfully extracted {len(trajectories)} trajectories")












# from sklearn.decomposition import PCA
# from scipy.spatial.distance import euclidean
# import seaborn as sns


# # --- helper: safe normalize to fixed length ---
# from scipy.interpolate import interp1d
# def quick_normalize_fixed(traj, target_length=20):
#     """traj: ndarray [N,3] columns (frame_id, x, y). Returns [target_length, 2] or None."""
#     if traj is None or len(traj) < 2:
#         return None
#     xs, ys = traj[:, 1].astype(float), traj[:, 2].astype(float)
#     n = len(xs)
#     old_idx = np.linspace(0, n - 1, n)
#     new_idx = np.linspace(0, n - 1, target_length)
#     fx = interp1d(old_idx, xs, kind="linear")
#     fy = interp1d(old_idx, ys, kind="linear")
#     xr = fx(new_idx)
#     yr = fy(new_idx)
#     # center at origin
#     xr -= xr[0]; yr -= yr[0]
#     out = np.column_stack([xr, yr])
#     if np.isnan(out).any() or np.isinf(out).any():
#         return None
#     return out

# # --- build trajectories (interceptor = nearest DEFENDER to ball_land at end) ---
# trajectories = []
# for _, play in interceptions.iterrows():
#     output_play = tracking_data_output[
#         (tracking_data_output['game_id'] == play['game_id']) &
#         (tracking_data_output['play_id'] == play['play_id'])
#     ].copy()
#     if output_play.empty:
#         continue

#     # ball landing
#     inp_play = tracking_data_input[
#         (tracking_data_input['game_id'] == play['game_id']) &
#         (tracking_data_input['play_id'] == play['play_id'])
#     ]
#     if inp_play.empty:
#         continue
#     ball_land = inp_play[['ball_land_x', 'ball_land_y']].iloc[0]

#     # ensure we can filter defenders in output_play
#     # if output doesn't have player_side, merge it in from input for this play
#     if 'player_side' not in output_play.columns:
#         side_map = inp_play[['nfl_id', 'player_side']].drop_duplicates()
#         output_play = output_play.merge(side_map, on='nfl_id', how='left')

#     # final frame(s) – take last two frames for robustness
#     max_f = output_play['frame_id'].max()
#     final_frame = output_play[output_play['frame_id'].isin([max_f, max_f-1])].copy()

#     defenders = final_frame[final_frame['player_side'] == 'Defense'].copy()
#     if defenders.empty:
#         continue

#     # distance to ball landing
#     dx = defenders['x'].astype(float) - float(ball_land['ball_land_x'])
#     dy = defenders['y'].astype(float) - float(ball_land['ball_land_y'])
#     defenders.loc[:, 'dist_to_ball'] = np.hypot(dx, dy)

#     int_defender_id = defenders.loc[defenders['dist_to_ball'].idxmin(), 'nfl_id']

#     # full trajectory for that defender
#     traj = output_play.loc[output_play['nfl_id'] == int_defender_id, ['frame_id','x','y']] \
#                        .sort_values('frame_id').to_numpy()
#     if traj is None or len(traj) < 5:
#         continue

#     trajectories.append(traj)

# print(f"Extracted raw trajectories: {len(trajectories)}")

# # --- normalize to fixed length and stack ---
# normalized = []
# for t in trajectories:
#     tn = quick_normalize_fixed(t, target_length=20)
#     if tn is not None:
#         normalized.append(tn)

# if len(normalized) < 2:
#     raise RuntimeError("Not enough normalized trajectories to run PCA.")

# X = np.stack([t.flatten() for t in normalized], axis=0)  # shape = (n_traj, 40)

# # --- PCA & distance plots ---
# pca = PCA(n_components=2, random_state=42)
# X_pca = pca.fit_transform(X)

# plt.figure(figsize=(10, 8))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)
# plt.xlabel('PC1'); plt.ylabel('PC2')
# plt.title(f'Interception Trajectories in 2D (n={len(normalized)})')
# plt.grid(True, alpha=0.3)
# plt.savefig('trajectory_pca.png')

# from scipy.spatial.distance import pdist, squareform
# distances = squareform(pdist(X, metric='euclidean'))

# plt.figure(figsize=(8, 6))
# sns.heatmap(distances, cmap='viridis', square=True)
# plt.title('Pairwise Trajectory Distances')
# plt.savefig('distance_matrix.png')
# print(f"Mean pairwise distance: {distances[np.triu_indices_from(distances, k=1)].mean():.2f}")
# print(f"Std of distances: {distances[np.triu_indices_from(distances, k=1)].std():.2f}")







# import numpy as np
# from scipy.interpolate import interp1d

# EPS = 1e-9

# def unit(vec):
#     n = np.linalg.norm(vec, axis=-1, keepdims=True)
#     return vec / np.clip(n, EPS, None)

# def resample_series(arr, target_len):
#     """arr: [T, d] → resampled to [target_len, d] (linear)"""
#     T = len(arr)
#     if T < 2:  # can't resample
#         return None
#     old = np.linspace(0, T-1, T)
#     new = np.linspace(0, T-1, target_len)
#     if arr.ndim == 1:
#         f = interp1d(old, arr, kind="linear")
#         return f(new)
#     out = []
#     for d in range(arr.shape[1]):
#         f = interp1d(old, arr[:, d], kind="linear")
#         out.append(f(new))
#     return np.stack(out, axis=1)

# def pick_window_idx(T, start_frac=0.0, end_frac=0.8):
#     """Indices covering early→mid flight (avoid late universal collapse)."""
#     a = int(np.floor(start_frac * (T-1)))
#     b = int(np.floor(end_frac   * (T-1)))
#     return np.arange(max(0,a), max(1,b+1))

# def get_tracks_for_play(game_id, play_id, output_df, input_df):
#     """
#     Returns:
#       def_xy:   ndarray [Td, 2] defender xy (post-throw frames only)
#       wr_xy:    ndarray [Tw, 2] targeted WR xy (post-throw)
#       ball_land: np.array [2] landing (x,y)
#       meta: dict with 'T_out' (frames), 'pass_length', etc.
#     """
#     outp = output_df[(output_df.game_id==game_id)&(output_df.play_id==play_id)].copy()
#     inp  = input_df[(input_df.game_id==game_id)&(input_df.play_id==play_id)].copy()
#     if outp.empty or inp.empty:
#         return None

#     # Landing point (stable reference)
#     ball_land = inp[['ball_land_x','ball_land_y']].iloc[0].to_numpy(dtype=float)

#     # Targeted WR id
#     wr_id = inp[inp['player_role']=='Targeted Receiver']['nfl_id'].iloc[0]

#     # Interceptor (for INTs) OR nearest contender (for comps, you’ll pick differently later)
#     # Here we just return WR; defender selection is outside this function.
#     # Build [frame_id, x, y] for WR (post-throw segment is already in output_*.csv)
#     wr_traj = outp[outp['nfl_id']==wr_id][['frame_id','x','y']].sort_values('frame_id').to_numpy(dtype=float)
#     meta = {'T_out': int(outp['frame_id'].max()),
#             'pass_length': float(inp['pass_length'].iloc[0]) if 'pass_length' in inp.columns else np.nan}
#     return wr_traj, ball_land, meta, outp

# def build_relative_features(def_traj, wr_traj, ball_land, target_len=20, end_frac=0.8):
#     """
#     def_traj, wr_traj: ndarray [T, 3] with columns [frame_id, x, y] (post-throw slice)
#     Returns: ndarray [target_len, F] with F≈7 features per frame
#     """
#     if def_traj is None or wr_traj is None: return None
#     if len(def_traj) < 3 or len(wr_traj) < 3: return None

#     # Align by overlapping frame_ids
#     f_def = def_traj[:,0].astype(int)
#     f_wr  = wr_traj[:,0].astype(int)
#     frames = np.intersect1d(f_def, f_wr)
#     if len(frames) < 5: return None

#     # Slice common window, then keep early→mid flight
#     def_xy = def_traj[np.isin(f_def, frames)][:,1:3]
#     wr_xy  = wr_traj [np.isin(f_wr , frames)][:,1:3]
#     T = len(frames)
#     idx = pick_window_idx(T, 0.0, end_frac)
#     def_xy = def_xy[idx]; wr_xy = wr_xy[idx]
#     if len(def_xy) < 5: return None

#     # Distances to references
#     vec_db = def_xy - ball_land  # def → ball_land
#     dist_db = np.linalg.norm(vec_db, axis=1)

#     vec_wb = wr_xy  - ball_land  # wr → ball_land
#     dist_wb = np.linalg.norm(vec_wb, axis=1)

#     dist_dw = np.linalg.norm(def_xy - wr_xy, axis=1)

#     # Velocities (simple first difference)
#     vel_def = np.vstack([def_xy[1:] - def_xy[:-1], def_xy[-1:] - def_xy[-2:-1]])
#     vel_wr  = np.vstack([wr_xy [1:] - wr_xy [:-1], wr_xy [-1:] - wr_xy [-2:-1]])

#     # Cosine alignment to ball (are you moving toward the ball?)
#     u_vel_def = unit(vel_def)
#     u_to_ball = unit(-vec_db)  # direction from defender to ball_land
#     cos_ang_ball = np.sum(u_vel_def * u_to_ball, axis=1)
#     cos_ang_ball = np.clip(cos_ang_ball, -1.0, 1.0)

#     # Closing speeds (finite diff of distances)
#     closing_db = np.hstack([0.0, -(np.diff(dist_db))])  # positive = getting closer to landing
#     closing_dw = np.hstack([0.0, -(np.diff(dist_dw))])  # positive = gaining on WR

#     # Advantage vs WR (positive if defender closer to landing than WR)
#     advantage = dist_wb - dist_db

#     # Lateral offset: perp distance to line from WR→ball
#     # line: ball_land + t*(wr - ball_land) ; vector n = rotate90(unit(wr-ball))
#     wr_ball_vec = wr_xy - ball_land
#     u_wb = unit(wr_ball_vec)
#     # Rotate (x,y) by +90°: (−y, x)
#     n_perp = np.stack([-u_wb[:,1], u_wb[:,0]], axis=1)
#     lateral_off = np.sum((def_xy - ball_land) * n_perp, axis=1)  # signed perp distance

#     # Stack features (shape [T', F])
#     feat = np.column_stack([
#         dist_db, closing_db,
#         cos_ang_ball,
#         dist_dw, closing_dw,
#         advantage,
#         lateral_off
#     ])

#     # Optional: simple per-feature normalization (robust)
#     # You can leave raw for clustering, or z-score within pass-length bins later.
#     feat_rs = resample_series(feat, target_len)
#     return feat_rs


# def pick_contender_ids_midflight(outp_play, ball_land, k=2, mid_frac=0.6):
#     """
#     Returns up to k defender ids that were closest to ball_land around mid-flight.
#     """
#     max_f = int(outp_play['frame_id'].max())
#     mid_f = int(np.floor(mid_frac * max_f))
#     mid = outp_play[outp_play['frame_id']==mid_f].copy()
#     # Ensure we filter defenders (merge player_side if needed)
#     if 'player_side' not in mid.columns:
#         # merge from input slice before calling this
#         pass
#     defs = mid[mid['player_side']=='Defense'].copy()
#     if defs.empty: return []
#     d = np.hypot(defs['x']-ball_land[0], defs['y']-ball_land[1])
#     defs = defs.assign(dist=d).sort_values('dist')
#     return defs['nfl_id'].head(k).tolist()



# TARGET_LEN = 20

# # INT group (one defender per play)
# int_feats = []
# for (game_id, play_id) in interceptions:
#     wr_traj, ball_land, meta, outp = get_tracks_for_play(game_id, play_id, tracking_data_output, tracking_data_input)
#     if wr_traj is None: continue

#     # find interceptor id as you already do (nearest to ball_land at final frame)
#     max_f = int(outp['frame_id'].max())
#     final = outp[outp['frame_id']==max_f].copy()
#     # ensure defender filter; merge player_side if needed
#     # ...
#     int_id = defenders.loc[...,'nfl_id']
#     def_traj = outp[outp['nfl_id']==int_id][['frame_id','x','y']].sort_values('frame_id').to_numpy(float)

#     feat = build_relative_features(def_traj, wr_traj, ball_land, TARGET_LEN, end_frac=0.8)
#     if feat is not None:
#         int_feats.append(feat)

# # Completion group (k contenders per play at mid-flight)
# comp_feats = []
# for (game_id, play_id) in completions:
#     wr_traj, ball_land, meta, outp = get_tracks_for_play(game_id, play_id, tracking_data_output, tracking_data_input)
#     if wr_traj is None: continue
#     # merge player_side onto outp if needed
#     # ...
#     ids = pick_contender_ids_midflight(outp, ball_land, k=2, mid_frac=0.6)
#     for did in ids:
#         def_traj = outp[outp['nfl_id']==did][['frame_id','x','y']].sort_values('frame_id').to_numpy(float)
#         feat = build_relative_features(def_traj, wr_traj, ball_land, TARGET_LEN, end_frac=0.8)
#         if feat is not None:
#             comp_feats.append(feat)



# # Statistics
# int_mean = np.mean(int_similarities)
# int_std = np.std(int_similarities)
# int_median = np.median(int_similarities)

# comp_mean = np.mean(comp_similarities)
# comp_std = np.std(comp_similarities)
# comp_median = np.median(comp_similarities)

# print(f"\nInterceptions (n={len(int_similarities)}):")
# print(f"  Mean similarity:   {int_mean:.4f}")
# print(f"  Median similarity: {int_median:.4f}")
# print(f"  Std similarity:    {int_std:.4f}")

# print(f"\nCompletions (n={len(comp_similarities)}):")
# print(f"  Mean similarity:   {comp_mean:.4f}")
# print(f"  Median similarity: {comp_median:.4f}")
# print(f"  Std similarity:    {comp_std:.4f}")

# difference = int_mean - comp_mean
# pct_increase = (difference / comp_mean * 100) if comp_mean > 0 else 0

# print(f"\nDifference: {difference:.4f} ({pct_increase:.1f}% {'increase' if difference > 0 else 'decrease'})")

# # Statistical test
# from scipy.stats import mannwhitneyu, ttest_ind

# # Mann-Whitney U (non-parametric)
# statistic_mw, pvalue_mw = mannwhitneyu(int_similarities, comp_similarities, alternative='greater')

# # Also do t-test for comparison
# statistic_t, pvalue_t = ttest_ind(int_similarities, comp_similarities)

# print(f"\nMann-Whitney U test (primary):")
# print(f"  Statistic: {statistic_mw:.2f}")
# print(f"  p-value: {pvalue_mw:.6f}")

# if pvalue_mw < 0.001:
#     print("  ✅✅✅ HIGHLY SIGNIFICANT (p < 0.001)")
#     verdict = "STRONG GREEN LIGHT"
# elif pvalue_mw < 0.01:
#     print("  ✅✅ VERY SIGNIFICANT (p < 0.01)")
#     verdict = "STRONG GREEN LIGHT"
# elif pvalue_mw < 0.05:
#     print("  ✅ SIGNIFICANT (p < 0.05)")
#     verdict = "GREEN LIGHT"
# elif pvalue_mw < 0.10:
#     print("  ⚠️ MARGINALLY SIGNIFICANT (p < 0.10)")
#     verdict = "YELLOW LIGHT"
# else:
#     print("  ❌ NOT SIGNIFICANT (p >= 0.10)")
#     verdict = "RED LIGHT"

# print(f"\nT-test (for reference):")
# print(f"  p-value: {pvalue_t:.6f}")

# # Effect size (Cohen's d)
# pooled_std = np.sqrt((int_std**2 + comp_std**2) / 2)
# cohens_d = (int_mean - comp_mean) / pooled_std if pooled_std > 0 else 0
# print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")

# if cohens_d > 0.8:
#     print("  ✅✅✅ LARGE EFFECT")
#     effect_verdict = "EXCELLENT"
# elif cohens_d > 0.5:
#     print("  ✅✅ MEDIUM EFFECT")
#     effect_verdict = "GOOD"
# elif cohens_d > 0.2:
#     print("  ✅ SMALL EFFECT")
#     effect_verdict = "ACCEPTABLE"
# elif cohens_d > 0:
#     print("  ⚠️ VERY SMALL EFFECT")
#     effect_verdict = "WEAK"
# else:
#     print("  ❌ NEGATIVE EFFECT (completions higher!)")
#     effect_verdict = "FAILED"

# # Overlap coefficient
# from scipy.stats import gaussian_kde

# # Create KDEs
# int_kde = gaussian_kde(int_similarities)
# comp_kde = gaussian_kde(comp_similarities)

# # Calculate overlap
# x_range = np.linspace(
#     min(min(int_similarities), min(comp_similarities)),
#     max(max(int_similarities), max(comp_similarities)),
#     1000
# )

# overlap = np.trapz(np.minimum(int_kde(x_range), comp_kde(x_range)), x_range)
# print(f"\nDistribution overlap: {overlap:.3f} (lower is better, <0.7 is good)")

# # Visualization
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# # 1. Histogram with KDE
# axes[0, 0].hist(int_similarities, bins=30, alpha=0.5, label='Interceptions', 
#                 color='red', density=True, edgecolor='darkred')
# axes[0, 0].hist(comp_similarities, bins=30, alpha=0.5, label='Completions', 
#                 color='blue', density=True, edgecolor='darkblue')

# # Add KDE lines
# axes[0, 0].plot(x_range, int_kde(x_range), 'r-', linewidth=2, alpha=0.8)
# axes[0, 0].plot(x_range, comp_kde(x_range), 'b-', linewidth=2, alpha=0.8)

# axes[0, 0].axvline(int_mean, color='red', linestyle='--', linewidth=2, 
#                    label=f'INT Mean: {int_mean:.3f}')
# axes[0, 0].axvline(comp_mean, color='blue', linestyle='--', linewidth=2, 
#                    label=f'Comp Mean: {comp_mean:.3f}')
# axes[0, 0].set_xlabel('Max Template Similarity')
# axes[0, 0].set_ylabel('Density')
# axes[0, 0].set_title('Template Similarity Distribution')
# axes[0, 0].legend(fontsize=9)
# axes[0, 0].grid(True, alpha=0.3)

# # 2. Box plot
# bp = axes[0, 1].boxplot([int_similarities, comp_similarities], 
#                         labels=['Interceptions\n(n={})'.format(len(int_similarities)), 
#                                 'Completions\n(n={})'.format(len(comp_similarities))],
#                         showmeans=True,
#                         patch_artist=True,
#                         widths=0.6)
# bp['boxes'][0].set_facecolor('red')
# bp['boxes'][1].set_facecolor('blue')
# bp['boxes'][0].set_alpha(0.5)
# bp['boxes'][1].set_alpha(0.5)
# axes[0, 1].set_ylabel('Max Template Similarity')
# axes[0, 1].set_title('Similarity Comparison')
# axes[0, 1].grid(True, alpha=0.3, axis='y')

# # Add text with statistics
# stats_text = f'Difference: {difference:.4f}\np-value: {pvalue_mw:.4f}\nCohen\'s d: {cohens_d:.3f}'
# axes[0, 1].text(0.02, 0.98, stats_text, transform=axes[0, 1].transAxes,
#                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
#                 fontsize=9)

# # 3. Cumulative distribution
# int_sorted = np.sort(int_similarities)
# comp_sorted = np.sort(comp_similarities)
# int_cdf = np.arange(1, len(int_sorted) + 1) / len(int_sorted)
# comp_cdf = np.arange(1, len(comp_sorted) + 1) / len(comp_sorted)

# axes[1, 0].plot(int_sorted, int_cdf, 'r-', linewidth=2.5, label='Interceptions', alpha=0.8)
# axes[1, 0].plot(comp_sorted, comp_cdf, 'b-', linewidth=2.5, label='Completions', alpha=0.8)
# axes[1, 0].set_xlabel('Max Template Similarity')
# axes[1, 0].set_ylabel('Cumulative Probability')
# axes[1, 0].set_title('Cumulative Distribution Function')
# axes[1, 0].legend()
# axes[1, 0].grid(True, alpha=0.3)

# # Add shaded area showing separation
# axes[1, 0].fill_betweenx([0, 1], int_mean, comp_mean, alpha=0.2, color='green')

# # 4. Violin plot
# parts = axes[1, 1].violinplot([int_similarities, comp_similarities], 
#                                positions=[1, 2], 
#                                showmeans=True, 
#                                showmedians=True,
#                                widths=0.7)

# for pc, color in zip(parts['bodies'], ['red', 'blue']):
#     pc.set_facecolor(color)
#     pc.set_alpha(0.5)

# axes[1, 1].set_xticks([1, 2])
# axes[1, 1].set_xticklabels(['Interceptions', 'Completions'])
# axes[1, 1].set_ylabel('Max Template Similarity')
# axes[1, 1].set_title('Violin Plot Comparison')
# axes[1, 1].grid(True, alpha=0.3, axis='y')

# plt.tight_layout()
# plt.savefig('int_vs_completion_similarity.png', dpi=150, bbox_inches='tight')
# # plt.show()

# print("\n" + "="*60)
# print("FINAL DECISION CRITERIA:")
# print("="*60)
# print(f"✅ Silhouette Score: 0.387 (target: >0.2) - EXCELLENT")
# print(f"✅ Sample sizes: INT={len(int_similarities)}, COMP={len(comp_similarities)} - GOOD")
# print(f"{'✅' if pvalue_mw < 0.05 else '⚠️' if pvalue_mw < 0.10 else '❌'} p-value: {pvalue_mw:.6f} (target: <0.05) - {verdict}")
# print(f"{'✅' if cohens_d > 0.2 else '❌'} Effect Size: {cohens_d:.3f} (target: >0.2) - {effect_verdict}")
# print(f"{'✅' if int_mean > comp_mean else '❌'} INT > Completion: {int_mean > comp_mean} - {'PASS' if int_mean > comp_mean else 'FAIL'}")
# print(f"{'✅' if overlap < 0.7 else '⚠️' if overlap < 0.8 else '❌'} Distribution overlap: {overlap:.3f} (target: <0.7) - {'GOOD' if overlap < 0.7 else 'ACCEPTABLE' if overlap < 0.8 else 'POOR'}")

# print("\n" + "="*60)
# print("OVERALL VERDICT:")
# print("="*60)

# if pvalue_mw < 0.05 and cohens_d > 0.3 and int_mean > comp_mean:
#     print("🚀🚀🚀 STRONG GREEN LIGHT - PROCEED WITH FULL PROJECT!")
#     print("\nExpected outcomes:")
#     print(f"  - Final model AUC: {0.70 + min(cohens_d * 0.1, 0.15):.2f} - {0.75 + min(cohens_d * 0.1, 0.15):.2f}")
#     print(f"  - Average Precision: {0.12 + cohens_d * 0.05:.2f} - {0.18 + cohens_d * 0.05:.2f}")
#     print(f"  - Template interpretability: HIGH")
#     print(f"  - Project success probability: >85%")
#     print("\n✅ NEXT STEPS:")
#     print("  1. Add pass breakups to positive class (~1500-2000 more examples)")
#     print("  2. Engineer contextual features (closing speed, distance, coverage type)")
#     print("  3. Train LightGBM calibration model")
#     print("  4. Build trajectory comparison animations")
#     print("  5. Generate defender rankings and insights")
# elif pvalue_mw < 0.10 and int_mean > comp_mean:
#     print("⚠️ YELLOW LIGHT - Proceed with modifications")
#     print("\nRecommendations:")
#     print("  - Add pass breakups to boost signal (CRITICAL)")
#     print("  - Include more contextual features")
#     print("  - Consider 6-7 clusters")
#     print("  - Expected AUC: 0.63-0.68")
# else:
#     print("🛑 RED LIGHT - Template similarity doesn't distinguish INT from completions")
#     print("\nConsider alternative approaches")
# print("="*60)




# # print("\n" + "="*60)
# # print("DAY 2 AFTERNOON: INT vs COMPLETION TEST (CRITICAL)")
# # print("="*60)

# # # IMPORTANT: Determine the target length from your kmeans model
# # expected_flat_length = kmeans.cluster_centers_.shape[1]
# # expected_traj_length = expected_flat_length // 2  # x and y coordinates

# # print(f"\nCluster centers shape: {kmeans.cluster_centers_.shape}")
# # print(f"Expected trajectory length: {expected_traj_length} frames")

# # # Get completions from the same games
# # available_games = set(tracking_data_output['game_id'].unique())
# # completions = supplementary_data[
# #     (supplementary_data['pass_result'] == 'C') & 
# #     (supplementary_data['game_id'].isin(available_games))
# # ]

# # # Sample up to 200 completions (or all if fewer)
# # n_completions_to_sample = min(200, len(completions))
# # completions_sample = completions.sample(n=n_completions_to_sample, random_state=42)

# # print(f"\nExtracting completion trajectories (n={len(completions_sample)})...")

# # completion_trajectories = []
# # failed_completions = 0

# # for idx, play in completions_sample.iterrows():
# #     try:
# #         # Get output for this play
# #         output_play = tracking_data_output[
# #             (tracking_data_output['game_id'] == play['game_id']) & 
# #             (tracking_data_output['play_id'] == play['play_id'])
# #         ]
        
# #         # Get ball_land from input
# #         input_play = tracking_data_input[
# #             (tracking_data_input['game_id'] == play['game_id']) & 
# #             (tracking_data_input['play_id'] == play['play_id'])
# #         ]
        
# #         if input_play.empty or output_play.empty:
# #             failed_completions += 1
# #             continue
        
# #         ball_land_x = input_play['ball_land_x'].iloc[0]
# #         ball_land_y = input_play['ball_land_y'].iloc[0]
        
# #         # Get nearest defender to ball (who failed to INT)
# #         final_frame = output_play[output_play['frame_id'] == output_play['frame_id'].max()]
# #         defenders = final_frame[final_frame['player_side'] == 'Defense'].copy()
        
# #         if defenders.empty:
# #             failed_completions += 1
# #             continue
        
# #         defenders['dist_to_ball'] = np.sqrt(
# #             (defenders['x'] - ball_land_x)**2 + 
# #             (defenders['y'] - ball_land_y)**2
# #         )
        
# #         nearest_defender_id = defenders.loc[defenders['dist_to_ball'].idxmin(), 'nfl_id']
        
# #         # Extract trajectory
# #         traj = output_play[output_play['nfl_id'] == nearest_defender_id][['frame_id', 'x', 'y']].values
        
# #         if len(traj) >= 5:
# #             # Use quick_normalize_fixed with the expected length
# #             normalized = quick_normalize_fixed(traj, target_length=expected_traj_length)
# #             if normalized is not None:
# #                 completion_trajectories.append(normalized)
# #             else:
# #                 failed_completions += 1
# #         else:
# #             failed_completions += 1
            
# #     except Exception as e:
# #         failed_completions += 1
# #         continue

# # print(f"✅ Extracted {len(completion_trajectories)} completion trajectories")
# # print(f"❌ Failed {failed_completions} extractions")

# # # Verify shapes match
# # if len(completion_trajectories) > 0 and len(trajectories) > 0:
# #     print(f"\nShape verification:")
# #     print(f"  INT trajectory shape: {trajectories[0].shape}")
# #     print(f"  Completion trajectory shape: {completion_trajectories[0].shape}")
# #     print(f"  INT flattened length: {trajectories[0].flatten().shape[0]}")
# #     print(f"  Completion flattened length: {completion_trajectories[0].flatten().shape[0]}")
# #     print(f"  Cluster center length: {kmeans.cluster_centers_.shape[1]}")
    
# #     # Check if shapes match
# #     if trajectories[0].flatten().shape[0] != kmeans.cluster_centers_.shape[1]:
# #         print("\n⚠️ WARNING: Shape mismatch detected!")
# #         print("Re-normalizing all trajectories to match cluster centers...")
        
# #         # Re-normalize all interception trajectories
# #         # This happens if you changed target_length after clustering
# #         all_trajectories_fixed = []
# #         for traj_orig in trajectories:
# #             # Need to go back to original unnormalized trajectory
# #             # Since we don't have that, we'll skip this batch
# #             pass
        
# #         print("❌ Cannot proceed - trajectories were normalized with different target_length")
# #         print("Solution: Re-run clustering with target_length matching current setting")
# # else:
# #     print("❌ No trajectories available for comparison")

# # # Compute similarity to templates
# # def compute_max_similarity(traj, cluster_centers):
# #     """Compute max similarity to any cluster center"""
# #     traj_flat = traj.flatten()
    
# #     # Verify shape matches
# #     if traj_flat.shape[0] != cluster_centers.shape[1]:
# #         raise ValueError(
# #             f"Shape mismatch: trajectory has {traj_flat.shape[0]} elements "
# #             f"but cluster centers expect {cluster_centers.shape[1]} elements. "
# #             f"Trajectory shape: {traj.shape}, expected: ({cluster_centers.shape[1]//2}, 2)"
# #         )
    
# #     similarities = []
# #     for center in cluster_centers:
# #         distance = np.linalg.norm(traj_flat - center)
# #         similarity = 1 / (1 + distance)
# #         similarities.append(similarity)
    
# #     return max(similarities)

# # # Only proceed if we have both sets of trajectories
# # if len(completion_trajectories) > 0 and len(trajectories) > 0:
# #     # Check shape consistency
# #     int_flat_len = trajectories[0].flatten().shape[0]
# #     comp_flat_len = completion_trajectories[0].flatten().shape[0]
# #     center_len = kmeans.cluster_centers_.shape[1]
    
# #     if int_flat_len != center_len or comp_flat_len != center_len:
# #         print("\n" + "="*60)
# #         print("❌ CRITICAL ERROR: Shape Mismatch")
# #         print("="*60)
# #         print(f"INT trajectories: {int_flat_len} elements")
# #         print(f"Completion trajectories: {comp_flat_len} elements")
# #         print(f"Cluster centers: {center_len} elements")
# #         print("\nYou need to re-run the clustering with the same target_length")
# #         print(f"Current target_length should be: {center_len // 2}")
# #     else:
# #         # Compute similarities for both groups
# #         print("\nComputing template similarities...")
        
# #         int_similarities = []
# #         for traj in trajectories:
# #             try:
# #                 sim = compute_max_similarity(traj, kmeans.cluster_centers_)
# #                 int_similarities.append(sim)
# #             except Exception as e:
# #                 print(f"Error computing INT similarity: {e}")
        
# #         comp_similarities = []
# #         for traj in completion_trajectories:
# #             try:
# #                 sim = compute_max_similarity(traj, kmeans.cluster_centers_)
# #                 comp_similarities.append(sim)
# #             except Exception as e:
# #                 print(f"Error computing completion similarity: {e}")
        
# #         print(f"Successfully computed {len(int_similarities)} INT similarities")
# #         print(f"Successfully computed {len(comp_similarities)} completion similarities")
        
# #         # Statistics
# #         int_mean = np.mean(int_similarities)
# #         int_std = np.std(int_similarities)
# #         int_median = np.median(int_similarities)
        
# #         comp_mean = np.mean(comp_similarities)
# #         comp_std = np.std(comp_similarities)
# #         comp_median = np.median(comp_similarities)
        
# #         print(f"\nInterceptions (n={len(int_similarities)}):")
# #         print(f"  Mean similarity:   {int_mean:.4f}")
# #         print(f"  Median similarity: {int_median:.4f}")
# #         print(f"  Std similarity:    {int_std:.4f}")
        
# #         print(f"\nCompletions (n={len(comp_similarities)}):")
# #         print(f"  Mean similarity:   {comp_mean:.4f}")
# #         print(f"  Median similarity: {comp_median:.4f}")
# #         print(f"  Std similarity:    {comp_std:.4f}")
        
# #         difference = int_mean - comp_mean
# #         pct_increase = (difference / comp_mean * 100) if comp_mean > 0 else 0
        
# #         print(f"\nDifference: {difference:.4f} ({pct_increase:.1f}% increase)")
        
# #         # Statistical test
# #         from scipy.stats import mannwhitneyu
# #         statistic, pvalue = mannwhitneyu(int_similarities, comp_similarities, alternative='greater')
# #         print(f"\nMann-Whitney U test:")
# #         print(f"  Statistic: {statistic:.2f}")
# #         print(f"  p-value: {pvalue:.6f}")
        
# #         if pvalue < 0.001:
# #             print("  ✅✅✅ HIGHLY SIGNIFICANT (p < 0.001)")
# #             verdict = "STRONG GREEN LIGHT"
# #         elif pvalue < 0.01:
# #             print("  ✅✅ VERY SIGNIFICANT (p < 0.01)")
# #             verdict = "STRONG GREEN LIGHT"
# #         elif pvalue < 0.05:
# #             print("  ✅ SIGNIFICANT (p < 0.05)")
# #             verdict = "GREEN LIGHT"
# #         elif pvalue < 0.10:
# #             print("  ⚠️ MARGINALLY SIGNIFICANT (p < 0.10)")
# #             verdict = "YELLOW LIGHT"
# #         else:
# #             print("  ❌ NOT SIGNIFICANT (p >= 0.10)")
# #             verdict = "RED LIGHT"
        
# #         # Effect size (Cohen's d)
# #         pooled_std = np.sqrt((int_std**2 + comp_std**2) / 2)
# #         cohens_d = (int_mean - comp_mean) / pooled_std if pooled_std > 0 else 0
# #         print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
        
# #         if cohens_d > 0.8:
# #             print("  ✅✅✅ LARGE EFFECT")
# #             effect_verdict = "EXCELLENT"
# #         elif cohens_d > 0.5:
# #             print("  ✅✅ MEDIUM EFFECT")
# #             effect_verdict = "GOOD"
# #         elif cohens_d > 0.2:
# #             print("  ✅ SMALL EFFECT")
# #             effect_verdict = "ACCEPTABLE"
# #         else:
# #             print("  ⚠️ NEGLIGIBLE EFFECT")
# #             effect_verdict = "WEAK"
        
# #         # Visualization
# #         import matplotlib.pyplot as plt
        
# #         fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
# #         # Histogram
# #         axes[0].hist(int_similarities, bins=30, alpha=0.6, label='Interceptions', color='red', density=True)
# #         axes[0].hist(comp_similarities, bins=30, alpha=0.6, label='Completions', color='blue', density=True)
# #         axes[0].axvline(int_mean, color='red', linestyle='--', linewidth=2, label=f'INT Mean: {int_mean:.3f}')
# #         axes[0].axvline(comp_mean, color='blue', linestyle='--', linewidth=2, label=f'Comp Mean: {comp_mean:.3f}')
# #         axes[0].set_xlabel('Max Template Similarity')
# #         axes[0].set_ylabel('Density')
# #         axes[0].set_title('Template Similarity Distribution')
# #         axes[0].legend(fontsize=8)
# #         axes[0].grid(True, alpha=0.3)
        
# #         # Box plot
# #         bp = axes[1].boxplot([int_similarities, comp_similarities], 
# #                               labels=['Interceptions', 'Completions'],
# #                               showmeans=True,
# #                               patch_artist=True)
# #         bp['boxes'][0].set_facecolor('red')
# #         bp['boxes'][1].set_facecolor('blue')
# #         bp['boxes'][0].set_alpha(0.5)
# #         bp['boxes'][1].set_alpha(0.5)
# #         axes[1].set_ylabel('Max Template Similarity')
# #         axes[1].set_title('Similarity Comparison')
# #         axes[1].grid(True, alpha=0.3)
        
# #         # Cumulative distribution
# #         int_sorted = np.sort(int_similarities)
# #         comp_sorted = np.sort(comp_similarities)
# #         int_cdf = np.arange(1, len(int_sorted) + 1) / len(int_sorted)
# #         comp_cdf = np.arange(1, len(comp_sorted) + 1) / len(comp_sorted)
        
# #         axes[2].plot(int_sorted, int_cdf, 'r-', linewidth=2, label='Interceptions')
# #         axes[2].plot(comp_sorted, comp_cdf, 'b-', linewidth=2, label='Completions')
# #         axes[2].set_xlabel('Max Template Similarity')
# #         axes[2].set_ylabel('Cumulative Probability')
# #         axes[2].set_title('Cumulative Distribution')
# #         axes[2].legend()
# #         axes[2].grid(True, alpha=0.3)
        
# #         plt.tight_layout()
# #         plt.savefig('int_vs_completion_similarity.png', dpi=150)
# #         # plt.show()
        
# #         print("\n" + "="*60)
# #         print("FINAL DECISION CRITERIA:")
# #         print("="*60)
# #         print(f"✅ Silhouette Score: 0.387 (target: >0.2) - EXCELLENT")
# #         print(f"{'✅' if pvalue < 0.05 else '⚠️' if pvalue < 0.10 else '❌'} p-value: {pvalue:.6f} (target: <0.05) - {verdict}")
# #         print(f"{'✅' if cohens_d > 0.2 else '❌'} Effect Size: {cohens_d:.3f} (target: >0.2) - {effect_verdict}")
# #         print(f"{'✅' if int_mean > comp_mean else '❌'} INT > Completion: {int_mean > comp_mean} - {'PASS' if int_mean > comp_mean else 'FAIL'}")
        
# #         print("\n" + "="*60)
# #         print("OVERALL VERDICT:")
# #         print("="*60)
        
# #         if pvalue < 0.05 and cohens_d > 0.3 and int_mean > comp_mean:
# #             print("🚀🚀🚀 STRONG GREEN LIGHT - PROCEED WITH FULL PROJECT!")
# #             print("\nExpected outcomes:")
# #             print(f"  - Final model AUC: {0.70 + min(cohens_d * 0.1, 0.15):.2f} - {0.75 + min(cohens_d * 0.1, 0.15):.2f}")
# #             print(f"  - Template interpretability: HIGH")
# #             print(f"  - Project success probability: >85%")
# #             print("\nNext steps:")
# #             print("  1. Add pass breakups to increase sample size")
# #             print("  2. Engineer contextual features")
# #             print("  3. Train LightGBM calibration model")
# #         elif pvalue < 0.05 and cohens_d > 0.2:
# #             print("✅ GREEN LIGHT - Proceed with confidence")
# #             print("\nExpected AUC: 0.68-0.73")
# #             print("Project success probability: >75%")
# #         elif pvalue < 0.10 and int_mean > comp_mean:
# #             print("⚠️ YELLOW LIGHT - Proceed with modifications")
# #             print("\nRecommendations:")
# #             print("  - Add pass breakups to boost positive sample")
# #             print("  - Include more contextual features (closing speed, distance to ball, etc.)")
# #             print("  - Consider 6-7 clusters instead of 5")
# #             print("  - Expected AUC: 0.63-0.68")
# #         else:
# #             print("🛑 RED LIGHT - Consider pivoting")
# #             print("\nThe template similarity doesn't distinguish INT from completions")
# #             print("Alternative approaches:")
# #             print("  - Pivot to catch probability heatmap")
# #             print("  - Focus on defender pursuit angles instead of full trajectories")
# #             print("  - Analyze defensive coordination patterns")
# #         print("="*60)
# # else:
# #     print("\n❌ Cannot proceed - insufficient trajectories for comparison")