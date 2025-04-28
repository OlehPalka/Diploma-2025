import pandas as pd

# List of your merged April files
files = [
    r"D:\Diploma\data_april\users_no_purchase_clusters_offer_test_group_filtered_march_with_metrics_april.xlsx",
    r"D:\Diploma\data_april\users_no_purchase_clusters_offer_control_group_filtered_march_with_metrics_april.xlsx",
    r"D:\Diploma\data_april\users_purchased_churned_clusters_offer_test_group_filtered_march_with_metrics_april.xlsx",
    r"D:\Diploma\data_april\users_purchased_churned_clusters_offer_control_group_filtered_march_with_metrics_april.xlsx"
]

purchased_list = []

for path in files:
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print(f"Warning: file not found: {path}")
        continue

    # ensure numeric comparison
    df["Orders Count"] = pd.to_numeric(df["Orders Count"], errors="coerce").fillna(0)
    df["Orders Count_april"] = pd.to_numeric(df["Orders Count_april"], errors="coerce").fillna(0)

    # filter for purchased users
    purchased = df[df["Orders Count_april"] > df["Orders Count"]]
    purchased_list.append(purchased)

# concatenate all purchased users
if purchased_list:
    all_purchased = pd.concat(purchased_list, ignore_index=True)
    # write to a new Excel file
    output_path = r"D:\Diploma\data_april\all_purchased_users_april.xlsx"
    all_purchased.to_excel(output_path, index=False)
    print(f"Saved {len(all_purchased)} purchased users to {output_path}")
else:
    print("No purchased users found in any file.")