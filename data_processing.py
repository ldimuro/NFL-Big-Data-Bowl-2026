import pandas as pd
from pandas import DataFrame
import constants
import pickle


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



