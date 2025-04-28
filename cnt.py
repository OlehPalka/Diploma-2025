import json
import pandas as pd
import os

def parse_key(key):
    """
    Given a key in the format:
        "20308985 - Юлія Кіселик - +380668804374"
    split and return (first_name, surname, phone).

    We split the key on " - " and assume the middle segment contains a first name and a surname.
    """
    try:
        parts = key.split(" - ")
        if len(parts) < 3:
            raise ValueError("Key does not have the expected format: 'ID - Firstname Lastname - Phone'")
        # parts[1] will be something like "Юлія Кіселик"
        name_parts = parts[1].split()
        if len(name_parts) < 2:
            raise ValueError("Name part does not contain both first name and surname.")
        first_name = name_parts[0]
        surname = name_parts[1]
        phone = parts[2]
        return first_name, surname, phone
    except Exception as e:
        print(f"Error parsing key '{key}': {e}")
        return None, None, None

def process_json_to_excel(file_path):
    # Open JSON file with proper encoding
    with open(file_path, 'r', encoding="utf-8-sig") as f:
        data = json.load(f)

    results = []
    
    # Process each key in the JSON dictionary
    for key in data:
        first_name, surname, phone = parse_key(key)
        if first_name and surname and phone:
            results.append({
                "first_name": first_name,
                "surname": surname,
                "phone": phone
            })

    # Create a pandas DataFrame from the parsed data
    df = pd.DataFrame(results)
    
    # Create Excel filename by using the same base name as the input file
    base_name, _ = os.path.splitext(file_path)
    output_file = base_name + ".xlsx"
    
    # Save the DataFrame to an Excel file (without DataFrame index)
    df.to_excel(output_file, index=False)
    print(f"Excel file saved as: {output_file}")

if __name__ == "__main__":
    # List of JSON file paths to process
    file_paths = [
        r"D:\Diploma\data_march\users_no_purchase_clusters_offer_test_group_filtered_march.json",
        r"D:\Diploma\data_march\users_purchased_churned_clusters_offer_test_group_filtered_march.json"
    ]
    
    # Process each file in the list
    for file_path in file_paths:
        process_json_to_excel(file_path)
