import get_data
import visualization
import data_processing
import pandas as pd

tracking_data_input = get_data.get_tracking_data(week_start=1, week_end=1, input=True)
tracking_data_input['data_type'] = 'input'
print('INPUT:', tracking_data_input.columns)

tracking_data_output = get_data.get_tracking_data(week_start=1, week_end=1, input=False)
tracking_data_output['data_type'] = 'output'
print('OUTPUT:', tracking_data_output.columns)

tracking_data = data_processing.combine_input_and_output(tracking_data_input, tracking_data_output)
print('COMBINED:', tracking_data.columns)

tracking_data = data_processing.normalize_field_direction(tracking_data)

supplementary_data = get_data.get_supplementary_data()

print(supplementary_data['pass_result'].value_counts())

test_game_id = 2023091000#2023091011#2023091010#2023091003#2023091003#2023091001#2023091006#2023090700
test_play_id = 661#2388#1558#529#358#2351#575#101
play_data = supplementary_data[(supplementary_data['game_id'] == test_game_id) & (supplementary_data['play_id'] == test_play_id)].iloc[0]
print(play_data)
test_frames_input = tracking_data[(tracking_data['game_id'] == test_game_id) & (tracking_data['play_id'] == test_play_id)]
                                                                          
visualization.create_play_gif(play_data, test_frames_input, f'{test_game_id}_{test_play_id}')





