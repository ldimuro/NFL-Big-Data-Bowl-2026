import pandas as pd

def get_tracking_data(week_start, week_end, input):
    #TODO: Add dynamic way to obtain data if repo is copied from Github

    tracking_data = []

    for week in range(week_start, week_end+1):
        # file_path = f'/Volumes/T7/Machine_Learning/Datasets/NFL/tracking_data/{year}_tracking_week_{week}.csv'
        file_path = f"/Volumes/T7/Machine_Learning/Datasets/NFL/big_data_bowl_2026/train/{'input' if input else 'output'}_2023_w{week:02}.csv"
        tracking_data.append(pd.read_csv(file_path))
        print(f'loaded {file_path}')

    return pd.concat(tracking_data, ignore_index=True)


def get_supplementary_data():
    file_path = f'/Volumes/T7/Machine_Learning/Datasets/NFL/big_data_bowl_2026/supplementary_data.csv'
    data = pd.read_csv(file_path)
    return data