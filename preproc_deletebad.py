import pandas as pd
import os
from paths import *



if __name__ == "__main__":
    threshold = 1

    df = pd.read_csv(os.path.join(phone_seg_anno_log_path, "log.csv"))
    filtered_df = df[~(abs(df["n_frames"] - df["duration"] * 16000) > threshold)]
    filtered_df.to_csv(os.path.join(phone_seg_anno_log_path, "log.csv"), index=False)

    df = pd.read_csv(os.path.join(phone_seg_random_log_path, "log.csv"))
    filtered_df = df[~(abs(df["n_frames"] - df["duration"] * 16000) > threshold)]
    filtered_df.to_csv(os.path.join(phone_seg_random_log_path, "log.csv"), index=False)