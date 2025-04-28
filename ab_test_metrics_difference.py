import pandas as pd

# List of your merged April files
files = [
    r"D:\Diploma\data_april\users_no_purchase_clusters_offer_test_group_filtered_march_with_metrics_april.xlsx",
    r"D:\Diploma\data_april\users_no_purchase_clusters_offer_control_group_filtered_march_with_metrics_april.xlsx",
    r"D:\Diploma\data_april\users_purchased_churned_clusters_offer_test_group_filtered_march_with_metrics_april.xlsx",
    r"D:\Diploma\data_april\users_purchased_churned_clusters_offer_control_group_filtered_march_with_metrics_april.xlsx"
]

for path in files:
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print(f"Warning: file not found: {path}")
        continue

    # 0) Coerce all relevant columns to numeric, filling non-parsable as 0
    to_numeric = [
        "Orders Count", "Orders Count_april",
        "Total Spent",  "Total Spent_april",
        "Days since last visit_april",
        "User LT Month", "User LT Month_april"
    ]
    for col in to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1) Orders count difference
    orders_diff = (df["Orders Count_april"] - df["Orders Count"]).sum()
    # 2) Total spend difference
    spend_diff = (df["Total Spent_april"] - df["Total Spent"]).sum()
    total_spent_m = df["Total Spent"].sum()
    # 3) % active users (Days since last visit_april < 30)
    active_pct = (df["Days since last visit_april"] < 30).sum()
    # 4) Average User LT Month before and in April
    avg_lt_before = df["User LT Month"].mean()
    avg_lt_april  = df["User LT Month_april"].mean()
    user_count = len(df)
    purchased_count   = (df["Orders Count_april"] > df["Orders Count"]).sum() 
    total_spent_april   = df["Total Spent_april"].sum()
    total_orders_april  = df["Orders Count_april"].sum()
    avg_pv_april = (total_spent_april / total_orders_april
                    if total_orders_april > 0 else 0)
    # Print results
    print(f"=== Stats for {path} ===")
    print(f"Number of users:               {user_count}")
    print(f"Number of purchased users (April): {purchased_count}")
    print(f"Total orders difference:       {orders_diff:.0f}")
    print(f"Total spend difference:        {spend_diff:.2f}")
    print(f"Total spend march:             {total_spent_m:.2f}")
    print(f"active users (<30 days):  {active_pct:.2f}")
    print(f"Avg User LT Month (before):    {avg_lt_before:.2f}")
    print(f"Avg User LT Month (in April):  {avg_lt_april:.2f}")
    print(f"Avg purchase value (in April):   {avg_pv_april:.2f}")
    print()
