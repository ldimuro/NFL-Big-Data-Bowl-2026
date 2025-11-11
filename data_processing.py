import pandas as pd
from pandas import DataFrame
import constants
import pickle
import math
import numpy as np


def normalize_field_direction(tracking_data):
    df = tracking_data.copy()
    left_mask = df['play_direction'] == 'left'
    df.loc[left_mask, 'x'] = constants.FIELD_LENGTH - df.loc[left_mask, 'x']
    df.loc[left_mask, 'absolute_yardline_number'] = constants.FIELD_LENGTH - df.loc[left_mask, 'absolute_yardline_number']
    df.loc[left_mask, 'ball_land_x'] = constants.FIELD_LENGTH - df.loc[left_mask, 'ball_land_x']
    df.loc[left_mask, 'o'] = (360 - df.loc[left_mask, 'o']) % 360
    df.loc[left_mask, 'dir'] = (360 - df.loc[left_mask, 'dir']) % 360
    df.loc[left_mask, 'play_direction'] = 'right_norm'

    return df


def normalize_field_direction_output(tracking_data):
    normalized_tracking_data = []

    # Flip spatial features so that offense is always going from left-to-right
    for week_df in tracking_data:
        week_df = week_df.copy()
        
        left_mask = week_df['play_direction'] == 'left'
        week_df.loc[left_mask, 'x'] = constants.FIELD_LENGTH - week_df.loc[left_mask, 'x']
        week_df.loc[left_mask, 'absolute_yardline_number'] = constants.FIELD_LENGTH - week_df.loc[left_mask, 'absolute_yardline_number']
        week_df.loc[left_mask, 'ball_land_x'] = constants.FIELD_LENGTH - week_df.loc[left_mask, 'ball_land_x']
        week_df.loc[left_mask, 'o'] = (360 - week_df.loc[left_mask, 'o']) % 360
        week_df.loc[left_mask, 'dir'] = (360 - week_df.loc[left_mask, 'dir']) % 360
        week_df.loc[left_mask, 'play_direction'] = 'right_norm'

        normalized_tracking_data.append(week_df)

    return normalized_tracking_data


def normalize_to_center(tracking_data: DataFrame):
    normalized_weeks = []
    
    for week_df in tracking_data:
        week_df = week_df.copy()
        normalized_plays = []

        # Group by each play (gameId + playId)
        for (game_id, play_id), play_df in week_df.groupby(['game_id', 'play_id']):
            ball_rows = play_df[play_df['team' if 'team' in week_df.columns else 'club'] == 'football']
            if ball_rows.empty:
                normalized_plays.append(play_df)
                continue

            # Calculate shift to move ball x to 60
            ball_x = ball_rows.iloc[0]['x']
            shift_x = 60 - ball_x

            play_df['x'] = play_df['x'] + shift_x
            normalized_plays.append(play_df)

        # Combine all normalized plays back into one DataFrame for the week
        normalized_weeks.append(pd.concat(normalized_plays, ignore_index=True))

    return normalized_weeks



def combine_input_and_output(input_df, output_df):
    metadata_cols = [
        'game_id', 'play_id', 'nfl_id', 'player_to_predict', 
        'play_direction', 'absolute_yardline_number', 'player_name',
        'player_height', 'player_weight', 'player_birth_date',
        'player_position', 'player_side', 'player_role',
        'ball_land_x', 'ball_land_y'
    ]

    player_metadata = input_df[metadata_cols].drop_duplicates(
        subset=['game_id', 'play_id', 'nfl_id']
    )

    input_selected = input_df[
        ['game_id', 'play_id', 'nfl_id', 'frame_id', 
         'x', 'y', 'o', 'dir', 'data_type']
    ].copy()

    output_selected = output_df[
        ['game_id', 'play_id', 'nfl_id', 'frame_id', 
         'x', 'y', 'data_type']
    ].copy()

    output_selected['o'] = -1
    output_selected['dir'] = -1

    combined_positions = pd.concat([input_selected, output_selected], ignore_index=True)

    combined_df = combined_positions.merge(
        player_metadata,
        on=['game_id', 'play_id', 'nfl_id'],
        how='left'
    )

    final_cols = ['game_id', 'play_id', 'nfl_id', 'player_to_predict', 
                'play_direction', 'absolute_yardline_number', 'player_name',
                'player_height', 'player_weight', 'player_birth_date',
                'player_position', 'player_side', 'player_role', 
                'frame_id', 'x', 'y', 'o', 'dir', 'ball_land_x', 'ball_land_y', 
                'data_type']

    combined_df = combined_df[final_cols]

    combined_df = combined_df.sort_values(by=['game_id', 'play_id', 'nfl_id'])

    return combined_df




def save_data(data, file_name):
    with open(f"{file_name}.pkl", 'wb') as f:
        pickle.dump(data, f)


def get_data(file_name):
    with open(f"{file_name}.pkl", 'rb') as f:
        data = pickle.load(f)
    return data


def _angle_wrap(a):
    """Wrap angle (deg) to [-180, 180]."""
    a = ((a + 180) % 360) - 180
    # Handle -180 edge to be consistent
    if a <= -180: a += 360
    return a

def _angle_deg(x1, y1, x2, y2):
    """Angle (deg) from (x1,y1) -> (x2,y2)."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def _angle_diff_deg(a, b):
    """Smallest signed difference a - b in degrees in [-180, 180]."""
    return _angle_wrap(a - b)

def _rotate(point_uv, theta_deg):
    """Rotate (dx,dy) by -theta around origin (deg). Returns (u,v)."""
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = point_uv
    # rotate by -theta -> multiply by [[cos, sin], [-sin, cos]]
    return (dx * c + dy * s, -dx * s + dy * c)

def _bin_pass_length(pl):
    if pd.isna(pl):
        return "unk"
    if pl <= 5: return 0 # 0-5
    if pl <= 10: return 1 # "6-10"
    if pl <= 20: return 2 # "11-20"
    return 3 # "21+"

def _encode_route(route_str: str) -> int:
    """
    Encode route string into numeric code based on frequency rank.
    Unseen or missing routes return 0.
    """
    route_map = {
        "HITCH": 1,
        "OUT": 2,
        "FLAT": 3,
        "CROSS": 4,
        "GO": 5,
        "IN": 6,
        "SLANT": 7,
        "POST": 8,
        "ANGLE": 9,
        "CORNER": 10,
        "SCREEN": 11,
        "WHEEL": 12,
    }
    if not isinstance(route_str, str):
        return 0
    route_str = route_str.strip().upper()
    return route_map.get(route_str, 0)

def extract_input_features(frame: pd.DataFrame, play_data: pd.Series | dict):
    # Identify targeted WR and coverage defenders
    wr_row = frame.loc[frame['player_role'] == 'Targeted Receiver']
    def_rows = frame.loc[frame['player_role'] == 'Defensive Coverage']

    if wr_row.empty or def_rows.empty:
        # Return minimal safe defaults if something is missing
        output = {
            'ok': 0,
            'depth_to_land_throw': np.nan,
            'sideline_prox': np.nan,
            'sep_d1_throw': np.nan,
            'sep_mean_k3_throw': np.nan,
            'count_r3_throw': 0,
            'count_r5_throw': 0,
            'cone_count_throw': 0,
            'density_10_throw': 0.0,
            'ang_spread_throw': np.nan,
            'wr_align_to_land_throw': np.nan,
            'db1_align_to_land_throw': np.nan,
            'land_gap_throw': np.nan,
            'down': play_data.get('down', np.nan),
            'yards_to_go': play_data.get('yards_to_go', np.nan),
            'pass_length_bin': _bin_pass_length(play_data.get('pass_length', np.nan)),
            'absolute_yardline_number': play_data.get('absolute_yardline_number', np.nan),
            'route_of_targeted_receiver': play_data.get('route_of_targeted_receiver', 'other'),
            'result': play_data.get('pass_result', np.nan)
        }

        output = pd.DataFrame([output])
        return output

    wr_row = wr_row.iloc[0]
    wr_id = wr_row['nfl_id']

    # Coordinates
    x_wr, y_wr = float(wr_row['x']), float(wr_row['y'])
    x_land, y_land = float(frame.iloc[0]['ball_land_x']), float(frame.iloc[0]['ball_land_y'])

    # Depth to landing & sideline proximity
    depth_to_land = math.dist((x_wr, y_wr), (x_land, y_land))
    sideline_prox = min(y_wr, 53.3 - y_wr)

    # Receiver orientation/direction (deg)
    dir_wr = float(wr_row['dir']) if not pd.isna(wr_row['dir']) else np.nan
    # Line from WR to landing
    ang_wr_to_land = _angle_deg(x_wr, y_wr, x_land, y_land)
    wr_align_to_land = abs(_angle_diff_deg(dir_wr, ang_wr_to_land)) if not np.isnan(dir_wr) else np.nan

    # Build WR-centric, landing-aligned coordinates for defenders
    # Translate each defender by WR, then rotate by -ang_wr_to_land
    def_list = []
    for _, d in def_rows.iterrows():
        dx = float(d['x']) - x_wr
        dy = float(d['y']) - y_wr
        u, v = _rotate((dx, dy), ang_wr_to_land)  # +u points toward landing
        r = math.hypot(u, v)
        dir_db = float(d['dir']) if not pd.isna(d['dir']) else np.nan
        ang_db_to_land = _angle_deg(float(d['x']), float(d['y']), x_land, y_land)
        db_align_to_land = abs(_angle_diff_deg(dir_db, ang_db_to_land)) if not np.isnan(dir_db) else np.nan
        dist_db_to_land = math.dist((float(d['x']), float(d['y'])), (x_land, y_land))
        def_list.append({
            'nfl_id': int(d['nfl_id']),
            'u': u, 'v': v, 'r': r,
            'phi': math.degrees(math.atan2(v, u)) if r > 0 else 0.0,
            'dir': dir_db,
            'db_align_to_land': db_align_to_land,
            'dist_db_to_land': dist_db_to_land
        })

    # Sort by radial distance and take K=3
    def_list.sort(key=lambda z: z['r'])
    K = 3
    nearest = def_list[:K]

    # Basic separations & group stats
    dists_all = [d['r'] for d in def_list]
    sep_d1 = nearest[0]['r'] if nearest else np.nan
    sep_mean_k3 = np.mean([d['r'] for d in nearest]) if nearest else np.nan
    count_r3 = sum(r <= 3.0 for r in dists_all)
    count_r5 = sum(r <= 5.0 for r in dists_all)
    density_10 = sum(1.0 / r for r in dists_all if r > 0 and r <= 10.0) if dists_all else 0.0

    # Cone occupancy toward landing (±40°)
    alpha = 40.0
    cone_count = sum(abs(d['phi']) <= alpha for d in def_list)

    # Angular spread (circular std of phi in radians) — robust crowding shape
    if len(def_list) >= 2:
        phis_rad = np.radians([d['phi'] for d in def_list])
        C = np.mean(np.cos(phis_rad)); S = np.mean(np.sin(phis_rad))
        R = np.hypot(C, S)
        ang_spread = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-9)))))
    else:
        ang_spread = np.nan

    # Alignment for the nearest defender (if present)
    db1_align_to_land = nearest[0]['db_align_to_land'] if nearest else np.nan

    # Landing distance gap (WR closer vs. nearest DB)
    dist_wr_to_land = depth_to_land
    min_db_dist_to_land = min(d['dist_db_to_land'] for d in nearest) if nearest else np.nan
    land_gap = (min_db_dist_to_land - dist_wr_to_land) if nearest else np.nan

    # Contextuals from play_data
    down = play_data.get('down', np.nan)
    ytg = play_data.get('yards_to_go', np.nan)
    pass_length_bin = _bin_pass_length(play_data.get('pass_length', np.nan))
    abs_yard = frame['absolute_yardline_number'].iloc[0] #play_data.get('absolute_yardline_number', np.nan)
    route = play_data.get('route_of_targeted_receiver', 'other')
    result = play_data.get('pass_result', np.nan)

    output = {
        'ok': 1,
        # Geometry
        'depth_to_land_throw': depth_to_land,
        'sideline_prox': sideline_prox,
        'sep_d1_throw': sep_d1,
        'sep_mean_k3_throw': sep_mean_k3,
        'count_r3_throw': count_r3,
        'count_r5_throw': count_r5,
        'cone_count_throw': cone_count,
        'density_10_throw': density_10,
        'ang_spread_throw': ang_spread,
        # Orientation / leverage
        'wr_align_to_land_throw': wr_align_to_land,
        'db1_align_to_land_throw': db1_align_to_land,
        # Distance advantage
        'land_gap_throw': land_gap,
        # Optional per-defender WR-centric positions (helps trees a lot)
        'u1_throw': nearest[0]['u'] if len(nearest) >= 1 else np.nan,
        'v1_throw': nearest[0]['v'] if len(nearest) >= 1 else np.nan,
        'u2_throw': nearest[1]['u'] if len(nearest) >= 2 else np.nan,
        'v2_throw': nearest[1]['v'] if len(nearest) >= 2 else np.nan,
        'u3_throw': nearest[2]['u'] if len(nearest) >= 3 else np.nan,
        'v3_throw': nearest[2]['v'] if len(nearest) >= 3 else np.nan,
        # Context
        'down': down,
        'yards_to_go': ytg,
        'pass_length_bin': pass_length_bin,
        'absolute_yardline_number': abs_yard,
        'route_of_targeted_receiver': route,
        'result': 1 if result == 'C' else 0
    }

    output = pd.DataFrame([output])

    output['route_of_targeted_receiver'] = output['route_of_targeted_receiver'].apply(_encode_route)

    output = output.drop(columns=['ok'])

    return output



