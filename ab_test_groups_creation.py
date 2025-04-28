import json
import random

def split_json_file(input_file, output_file_A, output_file_B):
    # Load the JSON data from the input file.
    with open(input_file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    
    # Get the list of user IDs.
    keys = list(data.keys())
    
    # Shuffle the keys randomly.
    random.shuffle(keys)
    
    # Calculate the split index (if odd, one group may have one extra user).
    n = len(keys)
    half = n // 2
    
    # Group A uses the first half; Group B uses the remaining keys.
    group_A_keys = keys[:half]
    group_B_keys = keys[half:]
    
    # Build dictionaries for each group.
    data_A = {user_id: data[user_id] for user_id in group_A_keys}
    data_B = {user_id: data[user_id] for user_id in group_B_keys}
    
    # Save the two groups as separate JSON files.
    with open(output_file_A, "w", encoding="utf-8-sig") as f:
        json.dump(data_A, f, indent=4, ensure_ascii=False)
    with open(output_file_B, "w", encoding="utf-8-sig") as f:
        json.dump(data_B, f, indent=4, ensure_ascii=False)
    
    print(f"Split '{input_file}' into:\n  - {output_file_A} (group A)\n  - {output_file_B} (group B)")

# ----------------------------------------------------------------
# Split users_no_purchase_clusters_offer.json into two random groups.
# ----------------------------------------------------------------
input_file_no_purchase = r"D:\Diploma\data_march\users_no_purchase_clusters_offer.json"
output_file_no_purchase_A = r"D:\Diploma\data_march\users_no_purchase_clusters_offer_A.json"
output_file_no_purchase_B = r"D:\Diploma\data_march\users_no_purchase_clusters_offer_B.json"
split_json_file(input_file_no_purchase, output_file_no_purchase_A, output_file_no_purchase_B)

# ----------------------------------------------------------------
# Split users_purchased_churned_clusters_offer.json into two random groups.
# ----------------------------------------------------------------
input_file_purchased = r"D:\Diploma\data_march\users_purchased_churned_clusters_offer.json"
output_file_purchased_A = r"D:\Diploma\data_march\users_purchased_churned_clusters_offer_A.json"
output_file_purchased_B = r"D:\Diploma\data_march\users_purchased_churned_clusters_offer_B.json"
split_json_file(input_file_purchased, output_file_purchased_A, output_file_purchased_B)
