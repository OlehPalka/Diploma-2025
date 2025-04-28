import json

def filter_users_by_days(input_file, output_file, threshold):
    """
    Load JSON data from input_file, filter users based on "Days since last visit"
    being less than the provided threshold, and save the filtered data to output_file.
    """
    with open(input_file, 'r', encoding="utf-8-sig") as f:
        data = json.load(f)
    
    filtered_data = {}
    for user_id, details in data.items():
        days = details.get("Days since last visit")
        if days is not None:
            try:
                days_numeric = float(days)
            except ValueError:
                continue
            if days_numeric < threshold:
                filtered_data[user_id] = details
    
    with open(output_file, 'w', encoding="utf-8-sig") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
    print(f"Filtered data saved to {output_file}. Total users: {len(filtered_data)}")

# -------------------------------------------------------------------
# For users without purchase: threshold is 35 days.
# -------------------------------------------------------------------
input_file_no_purchase = r"D:\Diploma\data_march\users_no_purchase_clusters_offer_test_group_march.json"
output_file_no_purchase = r"D:\Diploma\data_march\users_no_purchase_clusters_offer_test_group_filtered_march.json"
filter_users_by_days(input_file_no_purchase, output_file_no_purchase, threshold=35)

# -------------------------------------------------------------------
# For users with purchase/churned: threshold is 100 days.
# -------------------------------------------------------------------
input_file_purchased = r"D:\Diploma\data_march\users_purchased_churned_clusters_offer_test_group_march.json"
output_file_purchased = r"D:\Diploma\data_march\users_purchased_churned_clusters_offer_test_group_filtered_march.json"
filter_users_by_days(input_file_purchased, output_file_purchased, threshold=100)
