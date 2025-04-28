import pandas as pd
import os

# 1) load the full Club4Paws user table
master_csv = r"D:\Diploma\data_april\user_table_all_club4paws_users.csv"
master_df = pd.read_csv(master_csv, sep=";", encoding="utf-8-sig")

# 2) define which columns to grab and how to rename them
metrics = [
    "Days since last visit",
    "Orders Count",
    "Total Spent",
    "User LT Month"
]
april_cols = {
    col: f"{col}_april" 
    for col in metrics
}

# 3) list of your April files
files = [
    r"D:\Diploma\data_april\users_no_purchase_clusters_offer_test_group_filtered_march_with_metrics.xlsx",
    r"D:\Diploma\data_april\users_purchased_churned_clusters_offer_test_group_filtered_march_with_metrics.xlsx",
    r"D:\Diploma\data_april\users_no_purchase_clusters_offer_control_group_filtered_march_with_metrics.xlsx",
    r"D:\Diploma\data_april\users_purchased_churned_clusters_offer_control_group_filtered_march_with_metrics.xlsx"
]

for path in files:
    # read the pre-existing April workbook
    df = pd.read_excel(path)

    # prepare the April metrics from master_df
    april_df = (
        master_df[["User Id"] + metrics]
        .rename(columns=april_cols)
    )

    # merge in the April metrics
    merged = df.merge(april_df, on="User Id", how="left")

    # build new filename with "_april" before the extension
    base, ext = os.path.splitext(path)
    out_path = f"{base}_april{ext}"

    # save back out
    merged.to_excel(out_path, index=False)
    print(f"✔️  {os.path.basename(out_path)} saved.")
