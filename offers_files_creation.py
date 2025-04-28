import pandas as pd
import json

# ----------------------------------------------
# Load the projected centroids from the original training.
# We assume that "projected_centroids.csv" was saved with "Cluster" as the index.
# It should have at least these columns: 
# 'AVG Purchase Size', 'AVG Purchase value', 'Purchase Freq.' (note: column names must match exactly).
centroids_file = r"D:\Diploma\data_march\projected_centroids.csv"
centroids_df = pd.read_csv(centroids_file, sep=";", decimal=",", index_col="cluster")

# ----------------------------------------------
# 1. Process Users Without Purchase
# ----------------------------------------------
# Load the full dataset for users without purchase.
# This file should contain all columns (including User Id, day_of_week, hour_of_day, etc.).
file_no_purchase = r"D:\Diploma\data_march\user_table_no_purchase_with_clusters.csv"
df_no_purchase = pd.read_csv(file_no_purchase, sep=";", decimal=",", encoding="utf-8-sig")

df_no_purchase = df_no_purchase.dropna(subset=["Pet"])
df_no_purchase = df_no_purchase.dropna(subset=["day_of_week"])
df_no_purchase = df_no_purchase.dropna(subset=["hour_of_day"])

# For each row in df_no_purchase, we will form a dictionary entry.
# For these users, the three clustering fields (AVG Purchase Size, AVG Purchase value, Purchase Freq)
# are taken from the corresponding cluster’s centroid.
result_no_purchase = {}

for idx, row in df_no_purchase.iterrows():
    # Assume 'User Id' uniquely identifies the user.
    user_id = row["User Id"]
    cluster = row["cluster"]
    
    # Look up the centroid row for that cluster.
    # centroids_df.index are the cluster numbers.
    centroid = centroids_df.loc[int(cluster)]
    
    # Build the dictionary for the user.
    result_no_purchase[user_id] = {
        "day_of_week": row["day_of_week"],
        "hour_of_day": row["hour_of_day"],
        "Days since last visit": row["Days since last visit"],
        "AVG Purchase Size": centroid["AVG Purchase Size"],
        "AVG Purchase value": centroid["AVG Purchase value"],
        "Purchase Freq": centroid["Purchase Freq."],
        "Is interested in actions": row["Is interested in actions"],
        "Pet": row["Pet"],
        "cluster": cluster
    }

# Save the result as JSON.
output_no_purchase = r"D:\Diploma\data_march\users_no_purchase_clusters_offer.json"
with open(output_no_purchase, "w", encoding="utf-8-sig") as f:
    json.dump(result_no_purchase, f, indent=4, ensure_ascii=False)
print(f"Users without purchase JSON saved as '{output_no_purchase}'.")


# ----------------------------------------------
# 2. Process Users With Purchase (Purchased/Churned)
# ----------------------------------------------
# Load the full dataset for purchased/churned users.
file_purchased = r"D:\Diploma\data_march\user_table_purchased_churned_with_clusters.csv"
df_purchased = pd.read_csv(file_purchased, sep=";", decimal=",", encoding="utf-8-sig")

df_purchased = df_purchased.dropna()

result_purchased = {}

for idx, row in df_purchased.iterrows():
    user_id = row["User Id"]
    cluster = row["cluster"]
    
    # For churned users, if Purchase Freq. equals 0, then take it from the centroid; otherwise, use the original.
    original_purchase_freq = row["Purchase Freq."]
    if float(original_purchase_freq) == 0:
        purchase_freq = centroids_df.loc[int(cluster), "Purchase Freq."]
    else:
        purchase_freq = original_purchase_freq
        
    result_purchased[user_id] = {
        "day_of_week": row["day_of_week"],
        "hour_of_day": row["hour_of_day"],
        "Days since last visit": row["Days since last visit"],
        "AVG Purchase Size": row["AVG Purchase Size"],
        "AVG Purchase value": row["AVG Purchase value"],
        "Purchase Freq": purchase_freq,
        "Is interested in actions": row["Is interested in actions"],
        "Pet": row["Pet"],
        "cluster": cluster
    }

# Save the purchased/churned users dictionary as JSON.
output_purchased = r"D:\Diploma\data_march\users_purchased_churned_clusters_offer.json"
with open(output_purchased, "w", encoding="utf-8-sig") as f:
    json.dump(result_purchased, f, indent=4, ensure_ascii=False)
print(f"Users with purchase/churned JSON saved as '{output_purchased}'.")


import json
import pandas as pd

# ---------------------------
# File paths (update accordingly)
# ---------------------------
# Existing JSON file with cluster assignments for users without purchase
input_json = r"D:\Diploma\data_march\users_no_purchase_clusters_offer.json"
# CSV file with user categories; expected columns: "User Id", "Category of product"
categories_file = r"D:\Diploma\data_march\user_table_no_purchase_with_categories.csv"
# Output JSON file that will contain the updated information
output_json = r"D:\Diploma\data_march\users_no_purchase_clusters_offer.json"

# ---------------------------
# Load the JSON file with user cluster assignments.
# It is assumed that the JSON is a dictionary keyed by "User Id"
# with the user's details as the value (a nested dictionary).
# ---------------------------
with open(input_json, "r", encoding="utf-8-sig") as f:
    users_data = json.load(f)

# ---------------------------
# Load the CSV file that contains user categories.
# There may be multiple rows per user.
# ---------------------------
df_categories = pd.read_csv(categories_file, sep=";", decimal=",", encoding="utf-8-sig")

# ---------------------------
# Group the categories by "User Id" to create a dictionary
# where keys are User Id's and values are lists of categories.
# ---------------------------
categories_by_user = df_categories.groupby("User Id")["Category of product"].apply(list).to_dict()

# ---------------------------
# Update the JSON data.
# For each user in the JSON, add a new key "categories" with the list
# of categories from the CSV (or an empty list if the user has no entry).
# ---------------------------
for user_id, details in users_data.items():
    # Use .get() so that if the user_id is not in the categories dictionary, an empty list is returned.
    details["categories"] = categories_by_user.get(user_id, [])

# ---------------------------
# Save the updated JSON data to a new file.
# ---------------------------
with open(output_json, "w", encoding="utf-8-sig") as f:
    json.dump(users_data, f, indent=4, ensure_ascii=False)

print(f"Updated JSON with categories saved as '{output_json}'")


# ---------------------------
# File paths (update as needed)
# ---------------------------
# Existing JSON file for purchased/churned users (previously created)
json_input = r"D:\Diploma\data_march\users_purchased_churned_clusters_offer.json"
# CSV file with orders; expected columns: "User Id", "Idaction Name", "Quantity"
orders_file = r"D:\Diploma\data_march\user_table_purchased_churned_with_orders.csv"
# Output JSON file (updated)
json_output = r"D:\Diploma\data_march\users_purchased_churned_clusters_offer.json"

# ---------------------------
# Load the purchased/churned users JSON file
# ---------------------------
with open(json_input, "r", encoding="utf-8-sig") as f:
    purchased_data = json.load(f)

# ---------------------------
# Load the orders CSV file
# ---------------------------
df_orders = pd.read_csv(orders_file, sep=";", decimal=",")

# ---------------------------
# Group orders by "User Id" and "Idaction Name"
# Sum quantities for each action per user (if there are duplicates)
# ---------------------------
orders_grouped = df_orders.groupby(["User Id", "idaction_sku"])["Quantity"].sum().reset_index()

# Build a dictionary mapping each User Id to a dictionary of {Idaction Name: Quantity}
orders_dict = {}
for _, row in orders_grouped.iterrows():
    user_id = row["User Id"]
    sku = row["idaction_sku"]
    quantity = row["Quantity"]
    if user_id not in orders_dict:
        orders_dict[user_id] = {}
    orders_dict[user_id][sku] = quantity

# ---------------------------
# Update the JSON data for purchased/churned users:
# Add a new key "previous_purchase" for each user, populated with the orders dictionary.
# If a user does not have any orders, assign an empty dictionary.
# ---------------------------
for user_id, details in purchased_data.items():
    details["previous_purchase"] = orders_dict.get(user_id, {})

# ---------------------------
# Save the updated JSON with proper encoding and ensure_ascii=False so that Unicode characters are preserved.
# ---------------------------
with open(json_output, "w", encoding="utf-8-sig") as f:
    json.dump(purchased_data, f, indent=4, ensure_ascii=False)

print(f"Updated JSON with previous purchases saved as '{json_output}'.")



###
#CREATE RECOMENDATIONS
###

# ----------------------------------------------------------------
# Step 0: Set file paths (update these as needed)
# ----------------------------------------------------------------
# JSON files produced previously
json_no_purchase_file = r"D:\Diploma\data_march\users_no_purchase_clusters_offer.json"
json_purchased_file = r"D:\Diploma\data_march\users_purchased_churned_clusters_offer.json"

# New JSON output files with offers
json_no_purchase_offer = r"D:\Diploma\data_march\users_no_purchase_clusters_offer.json"
json_purchased_offer = r"D:\Diploma\data_march\users_purchased_churned_clusters_offer.json"

# Product table file (assumed to have columns including "ProductSku", "ProductPopularity", 
# "Category", "Pet" and "OftenBoughtWith")
product_table_file = r"D:\Diploma\data_march\Product_table_club4paws.csv"

# ----------------------------------------------------------------
# Step 1: Load the product table
# ----------------------------------------------------------------
# Adjust separator and decimal as needed
df_products = pd.read_csv(product_table_file, sep=";", decimal=",")

# It is assumed that:
# - "ProductSku" is the product identifier.
# - "ProductPopularity" is numeric, used to measure popularity.
# - "Category" represents product category.
# - "Pet" indicates for which pet the product is suitable.
# - "OftenBoughtWith" contains a value (could be a product SKU or a recommendation string).
#
# If your column names differ, please modify accordingly.
    
# ----------------------------------------------------------------
# Step 2: Process users without purchase for offers
# ----------------------------------------------------------------
# Load the existing JSON for users without purchase.
with open(json_no_purchase_file, "r", encoding="utf-8-sig") as f:
    users_no_purchase = json.load(f)

# For each user, use their "Pet" value and "categories" (a list) to find top-2 products.
# We assume that the JSON contains keys: "Pet" and "categories".
#
# For each category in user's "categories", filter the product table to rows where:
# df_products["Pet"] matches the user["Pet"] and df_products["Category"] equals that category.
# Then sort by "ProductPopularity" (descending) and take the top 2 "ProductSku".
#
# Combine the results (unique values) into a list and add as new key "you_may_like".

for user_id, info in users_no_purchase.items():
    user_pet = info.get("Pet")
    if user_pet == 1:
        user_pet = "Коти"
    elif user_pet == 2:
        user_pet = "Собаки"
    elif user_pet == 3:
        user_pet = "Мікс"
    # Expecting "categories" to be a list in the JSON
    user_categories = info.get("categories", [])
    recommendations = set()
    
    # For each category, find top2 products matching user's pet and the category.
    if user_pet in ("Коти", "Собаки"):
        for cat in user_categories:
            matched_products = df_products[
                (df_products["Product Cat/Dog"] == user_pet) & (df_products["ProductCategory"] == cat)
            ]
            # Sort descending by popularity
            matched_products = matched_products.sort_values(by="ProductPopularity", ascending=False)
            # Take top 2 SKUs (if available)
            top_products = matched_products["ProductSku"].head(2).tolist()
            recommendations.update(top_products)
    else:
        for cat in user_categories:
            matched_products = df_products[
                (df_products["ProductCategory"] == cat)
            ]
            # Sort descending by popularity
            matched_products = matched_products.sort_values(by="ProductPopularity", ascending=False)
            # Take top 2 SKUs (if available)
            top_products = matched_products["ProductSku"].head(2).tolist()
            recommendations.update(top_products)
    
    # Convert recommendations to list
    info["you_may_like"] = list(recommendations)

# Save the updated JSON for non-purchase users with offers.
with open(json_no_purchase_offer, "w", encoding="utf-8-sig") as f:
    json.dump(users_no_purchase, f, indent=4, ensure_ascii=False)
print(f"Users without purchase offers saved to '{json_no_purchase_offer}'.")


# ----------------------------------------------------------------
# Step 3: Process users with purchase/churned for offers
# ----------------------------------------------------------------
with open(json_purchased_file, "r", encoding="utf-8-sig") as f:
    users_purchased = json.load(f)

# For each user, use their "previous_purchase" dictionary.
# For each key in previous_purchase (assumed to be the product SKU or similar),
# look up in the product table the corresponding row where "ProductSku" matches this key.
# Then, retrieve the "OftenBoughtWith" value.
# If a match is found, add that "OftenBoughtWith" value to a recommendation set.
# Finally, add the unique list as the new key "you_may_like" in the JSON.

for user_id, info in users_purchased.items():
    prev_purchase = info.get("previous_purchase", {})
    recommendations = set()
    # Iterate through each purchased product (the key in the dictionary)
    for prod in prev_purchase.keys():
        # Filter product table where ProductSku equals the prod value.
        matched = df_products[df_products["ProductSku"] == int(prod)]
        if not matched.empty:
            # There might be multiple rows; we take the first's "OftenBoughtWith" value.
            often_bought_with = matched.iloc[0]["OftenBoughtWith"]
            # Optionally, check if the field is not null.
            if pd.notna(often_bought_with):
                often_bought_with = str(matched.iloc[0]["OftenBoughtWith"]).replace(".0", "")
                recommendations.add(int(often_bought_with))
                
    info["you_may_like"] = list(recommendations)

# Save the updated JSON for purchased/churned users with offers.
with open(json_purchased_offer, "w", encoding="utf-8-sig") as f:
    json.dump(users_purchased, f, indent=4, ensure_ascii=False)
print(f"Users with purchase/churned offers saved to '{json_purchased_offer}'.")