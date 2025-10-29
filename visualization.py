import matplotlib.pyplot as plt
import constants
import math
from pandas import DataFrame
import os
import imageio.v2 as imageio
import shutil
import numpy as np
import subprocess
import imageio_ffmpeg
from matplotlib.patches import Patch
# import data_processing
from matplotlib.ticker import PercentFormatter, MultipleLocator, FuncFormatter

def plot_frame(frame, play_data, file_name, data_type, zoom=False):
    fig, ax = plt.subplots(figsize=(12, 7.5 if zoom else 6.5))

    zoom_offset_x = 15
    zoom_offset_y = 8

    # Set green background for the field
    ax.set_facecolor('#0A7E0A')

    try:
        off_color = team_colors[play_data['possession_team']]
        def_color = team_colors[play_data['defensive_team']]
    except:
        off_color = team_colors['home']
        def_color = team_colors['away']

    # Draw red end zone (left) and blue end zone (right)
    ax.axvspan(0, 10, color=off_color, zorder=1)
    ax.axvspan(110, 120, color=def_color, zorder=1)

    # Draw yard lines every 10 yards
    for x in range(10, 111, 5):
        ax.axvline(x=x, color='white', linewidth=4 if zoom else 1, zorder=2)

    # Draw yard line numbers
    for x in range(20, 101, 10):
        field_val = str(x-10 if x < 60 else 110-x)

        # Bottom numbers
        ax.text(x=x, 
                y=constants.SIDELINE_TO_HASH/2, 
                s=f'{field_val[0]} {field_val[1]}', 
                fontsize=16, 
                ha='center', 
                va='center', 
                color='white', 
                fontname='Times New Roman',
                fontweight='bold')

        # Top numbers
        ax.text(x=x, 
                y=constants.FIELD_WIDTH - constants.SIDELINE_TO_HASH/2, 
                s=f'{field_val[0]} {field_val[1]}', 
                fontsize=16, 
                ha='center', 
                va='center', 
                color='white', 
                fontname='Times New Roman', 
                rotation=180,
                fontweight='bold')
        
        # Arrows next to yard labels
        if x != 60:
            ax.text(x=x-2.5 if x < 60 else x+2.5,
                    y=constants.SIDELINE_TO_HASH/2 + 0.4,
                    s='\u25B6',
                    fontsize=6,
                    ha='center', 
                    va='center', 
                    color='white',
                    rotation=180 if x < 60 else 0)
            
            ax.text(x=x-2.5 if x < 60 else x+2.5,
                    y=constants.FIELD_WIDTH - constants.SIDELINE_TO_HASH/2 - 0.4,
                    s='\u25B6',
                    fontsize=6,
                    ha='center', 
                    va='center', 
                    color='white',
                    rotation=180 if x < 60 else 0)

    line_color = '#DCDCDC'

    # Draw Center Field and Goalines
    ax.axvline(x=constants.CENTER_FIELD, color=line_color, linewidth=6 if zoom else 2, zorder=2.1)
    ax.axvline(x=constants.OFF_GOALLINE, color=line_color, linewidth=6 if zoom else 2, zorder=2.1)
    ax.axvline(x=constants.DEF_GOALLINE, color=line_color, linewidth=6 if zoom else 2, zorder=2.1)

    # Draw hash marks
    ax.axhline(y=constants.SIDELINE_TO_HASH, color=line_color, linestyle='dotted', linewidth=6 if zoom else 2, zorder=0)
    ax.axhline(y=constants.FIELD_WIDTH - constants.SIDELINE_TO_HASH, color=line_color, linestyle='dotted', linewidth=6 if zoom else 2, zorder=0)
    # ax.axhline(y=0.5, color='white', linestyle='dotted', linewidth=6 if zoom else 2, zorder=0)
    # ax.axhline(y=constants.FIELD_WIDTH-0.5, color='white', linestyle='dotted', linewidth=6 if zoom else 2, zorder=0)

    # Draw LoS and 1st-Down marker
    # los = constants.DEF_GOALLINE - frame.iloc[0]['absolute_yardline_number'] + constants.OFF_GOALLINE
    los = frame.iloc[0]['absolute_yardline_number']# + constants.OFF_GOALLINE
    ax.axvline(x=los, color='#26248f', linewidth=6 if zoom else 2, zorder=2.2)
    ax.axvline(x=los + play_data['yards_to_go'], color="#f2d627", linewidth=6 if zoom else 2, zorder=2.2)

    # Handle team colors
    teams = frame['player_side'].unique().tolist()# if ('club' not in frame.columns or frame['club'].isna().all()) else frame['club'].unique().tolist()
    # teams.remove('football')
    color_map = {teams[0]: team_colors[teams[0]], teams[1]: team_colors[teams[1]], 'football': '#dec000'}

    # Add ball
    # football = frame[frame['team'] == 'football'] if ('club' not in frame.columns or frame['club'].isna().all()) else frame[frame['club'] == 'football']
    # ax.scatter(football['x'], football['y'], c='#dec000', s=500 if zoom else 25, marker='o',zorder=3.1)

    # Add players
    players = frame#frame[frame['team'] != 'football'] if ('club' not in frame.columns or frame['club'].isna().all()) else frame[frame['club'] != 'football']
    ax.scatter(
        players['x'], 
        players['y'], 
        c=players['player_side'].map(color_map), 
        s=1000 if zoom else 70,
        edgecolors='black',
        zorder=3
    )

    # Add indicator around Targeted Receiver
    targeted_receiver_id = frame[frame['player_role'] == 'Targeted Receiver'].iloc[0]['nfl_id']
    receiver_row = frame[frame['nfl_id'] == targeted_receiver_id]
    if not receiver_row.empty:
        indicator_color = "#15FF00"

        ax.scatter(
            receiver_row['x'], 
            receiver_row['y'], 
            s=1100 if zoom else 100, 
            facecolors='none', 
            edgecolors=indicator_color, 
            linewidths=2, 
            zorder=4
        )


    if frame['data_type'].iloc[0] == 'input':

        # Convert angles to radians
        angles = np.deg2rad(players['o'].fillna(0))
        dx = np.sin(angles)   # X-component of direction
        dy = np.cos(angles)   # Y-component of direction

        # Offset distance: approximate radius of player circle
        marker_radius = 1.2 if zoom else 0.4

        # Apply offset to starting positions
        x_offset = players['x'] + dx * marker_radius
        y_offset = players['y'] + dy * marker_radius

        # Arrow length
        arrow_length = 0.8#1.0 if zoom else 0.5

        ax.quiver(
            x_offset, y_offset,   # Offset starting points
            dx, dy,               # Arrow directions
            angles='xy',
            scale_units='xy',
            scale=1 / arrow_length,
            color=players['player_side'].map(color_map),
            width=0.005,#0.01 if zoom else 0.0015,
            edgecolor='black',
            linewidth=1.0,
            zorder=2.9
        )

    # Add jersey numbers
    # for _, row in frame.iterrows():
    #     # Only plot the labels of players in frame
    #     if (row['x'] > ball_x-zoom_offset_x and row['x'] <= ball_x+zoom_offset_x) and (row['y'] > ball_y-zoom_offset_y and row['y'] <= ball_y+zoom_offset_y) or not zoom:
    #         label = '' if math.isnan(row['jerseyNumber']) else int(row['jerseyNumber'])
    #         ax.text(row['x'] + (0.6 if zoom else 0.5), row['y'], label, fontsize=16 if zoom else 8, zorder=4)

    # Field settings
    # if zoom:
    #     plt.xlim(ball_x - zoom_offset_x, ball_x + zoom_offset_x)
    #     plt.ylim(ball_y - zoom_offset_y, ball_y + zoom_offset_y)
    # else:
    #     plt.xlim(0, 120)
    #     plt.ylim(0, 53.3)
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)


    # Print ball land location
    ball_land_x = frame['ball_land_x'].iloc[0]
    ball_land_y = frame['ball_land_y'].iloc[0]
    pass_result = play_data['pass_result']
    match pass_result:
        case 'C':
            pass_icon = 'x'
            icon_color = "#5AFF4B"
        case 'I':
            pass_icon = 'x'
            icon_color = 'red'
        case 'IN':
            pass_icon = 'X'
            icon_color = 'red'
        case _:
            pass_icon = '.'
            icon_color = 'white'
    ax.plot(ball_land_x, ball_land_y, marker=pass_icon, markersize=8, color=icon_color)




    title = f"game: {play_data['game_id']}, play: {play_data['play_id']}, frame: {frame['frame_id'].iloc[0]}, pass_result: {play_data['pass_result']}, {frame['data_type'].iloc[0]}"
    # title = 'test'
    fig.suptitle(title, fontsize=18)

    suffixes = {1: 'st', 2: 'nd', 3: 'rd', 4: 'th'}

    try:
        possession_team = play_data['possession_team']
        defensive_team = play_data['defensive_team']
    except:
        possession_team = 'Team1'
        defensive_team = 'Team2'

    play_state = f"{possession_team} vs. {defensive_team}, Q{play_data['quarter']} {play_data['game_clock']}, {play_data['down']}{suffixes[play_data['down']]} & {play_data['yards_to_go']}"
    # play_state += f", yardsGained: {yards_gained}, {spsp_prob*100:.2f}% SPSP ({spsp_rolling_avg*100:.2f}% rolling)"
    # play_state = 'test_state'
    fig.text(0.5, 0.90, play_state, ha='center', fontsize=16)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    # plt.savefig(f"play_frames/{file_name}/{file_name}_{frame['frameId'].iloc[0]:04d}{'_zoomed' if zoom else ''}.png")
    plt.savefig(f"play_frames/{file_name}/{file_name}_{data_type}_{frame['frame_id'].iloc[0]:04d}_{'_zoomed' if zoom else ''}.png")
    plt.close()


def create_play_gif(play_data, frames: DataFrame, file_name, zoom=False, loop=True, delete_frame_plots=False):

    # Create new folder for frame plots
    folder_name = f'play_frames/{file_name}'
    os.makedirs(folder_name, exist_ok=True)

    print('creating gif...')

    # Create a plot for every frame in the range, first before the pass (input) and then after the pass (output)
    for data_type in ['input', 'output']:
        frames_data_type = frames[frames['data_type'] == data_type]
        frame_start = frames_data_type['frame_id'].min()
        frame_end = frames_data_type['frame_id'].max()

        print(f'frame_start {data_type}: {frame_start}')
        print(f'frame_end {data_type}: {frame_end}')

        for frame_id in range(frame_start, frame_end+1):
            frame = frames_data_type[frames_data_type['frame_id'] == frame_id]
            plot_frame(frame, play_data, file_name, data_type)
    
    frames_folder = f"play_frames/{file_name}"
    gif_output_path = f"play_gifs/{file_name}.gif"

    # Get list of image filenames in sorted order
    frame_files = sorted([
        os.path.join(frames_folder, fname)
        for fname in os.listdir(frames_folder)
        if fname.endswith('.png')
    ])

    # Load all frames as images
    images = [imageio.imread(f) for f in frame_files]

    # Build per-frame durations, add a pause at the end
    normal_duration = 100   # 0.1s for regular frames
    pause_duration = 3000   # 3s hold on last frame
    durations = [normal_duration] * (len(images) - 1) + [pause_duration]

    # Save GIF with per-frame durations
    loops = 0 if loop else 1
    imageio.mimsave(
        gif_output_path,
        images,
        duration=durations,
        loop=loops 
    )

    # Delete individual frame plots when completed
    if delete_frame_plots:
        if os.path.exists(frames_folder):
            shutil.rmtree(frames_folder)
            print(f'Deleted folder: {frames_folder}')
        else:
            print(f'Folder not found: {frames_folder}')

    print('gif created:', file_name)


def convert_gif_to_mp4(gif_path, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-i", gif_path,
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Converted: {gif_path} → {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")


def get_rolling_avg(prob_list, window_size=3):
    rolling_avg = []
    for i in range(len(prob_list)):
        if i < window_size - 1:
            rolling_avg.append(None)
        else:
            avg = sum(prob_list[i - window_size + 1:i + 1]) / window_size
            rolling_avg.append(avg)

    return rolling_avg


def get_yards_needed_for_success(play_data):
    down = play_data['down']
    yards_to_go = play_data['yardsToGo']

    # Play succeeds if:
    #   40% of yardsToGo gained on 1st down
    #   60% of yardsToGo gained on 2nd down
    #   100% of yardsToGo gained on 3rd/4th down
    yards_for_success = 0
    if down == 1:
        yards_for_success = np.ceil(yards_to_go * 0.4)
    elif down == 2:
        yards_for_success = np.ceil(yards_to_go * 0.6)
    else:
        yards_for_success = yards_to_go
    
    return yards_for_success


def rotate_frame_90ccw(frame: DataFrame):
    # Swap the coordinates (no vertical flip)
    rot = frame.copy()
    rot['x'] = frame['y']           # new horizontal coordinate
    rot['y'] = frame['x']           # new vertical coordinate

    # Rotate the heading
    if 'o' in rot.columns:
        # rotate the heading in the opposite direction.
        rot['o'] = (90 - frame['o'].fillna(0)) % 360

    return rot



team_colors = {
    'ARI': '#97233F',
    'ATL': '#A71930',
    'BAL': '#241773',
    'BUF': '#00338D',
    'CAR': '#0085CA',
    'CHI': '#0B162A',
    'CIN': '#FB4F14',
    'CLE': '#311D00',
    'DAL': '#003594',
    'DEN': '#FB4F14',
    'DET': '#0076B6',
    'GB':  '#203731',
    'HOU': '#03202F',
    'IND': '#002C5F',
    'JAX': '#006778',
    'KC':  '#E31837',
    'LV':  '#000000',
    'LAC': '#2472ca',
    'LA': '#003594',
    'MIA': '#008E97',
    'MIN': '#4F2683',
    'NE':  '#002244',
    'NO':  '#D3BC8D',
    'NYG': '#0B2265',
    'NYJ': '#125740',
    'PHI': '#004a50',
    'PIT': '#FFB612',
    'SEA': '#69BE28',
    'SF':  '#AA0000',
    'TB':  '#D50A0A',
    'TEN': '#4B92DB',
    'WAS': '#773141',
    'home': '#000000',
    'away': "#FF1F1F",
    'Offense': '#000000',
    'Defense': "#FFFFFF"
}