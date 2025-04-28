import pandas as pd

def process_csv(file_path, necessary_columns, output_file):
    """
    Reads a CSV file, keeps only the necessary columns, removes unnecessary ones,
    and calculates the mode of the day of the week and the hour of the day for 'visit_first_action_time' for each User Id.
    
    :param file_path: str, path to the CSV file
    :param necessary_columns: list, columns to keep
    :param output_file: str, path to save the filtered CSV file
    :return: DataFrame with only the necessary columns
    """
    # Read the CSV file
    df = pd.read_csv(file_path, sep=";")
    
    # Print all column names
    print("All columns in the CSV:", df.columns.tolist())
    
    df = df[necessary_columns]
    
    # Convert visit_first_action_time to datetime and extract day of the week and hour
    df['visit_first_action_time'] = pd.to_datetime(df['visit_first_action_time'], errors='coerce')
    df['day_of_week'] = df['visit_first_action_time'].dt.dayofweek  # Monday=0, Sunday=6
    df['hour_of_day'] = df['visit_first_action_time'].dt.hour  # 0-23 hours
    
    # Find mode of day of the week for each User Id
    mode_day = df.groupby('user_id')['day_of_week'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    
    # Find mode of hour of the day for each User Id
    mode_hour = df.groupby('user_id')['hour_of_day'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    
    # Merge mode_day and mode_hour back into df
    df = df.drop(columns=['visit_first_action_time', 'day_of_week', 'hour_of_day']).drop_duplicates()
    df = df.merge(mode_day, on='user_id', how='left')
    df = df.merge(mode_hour, on='user_id', how='left', suffixes=('_day', '_hour'))
    
    # Save the filtered DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")
    
    return df

# Example usage
if __name__ == "__main__":
    file_path = "\Diploma\data_march\log_visit.csv"  # Replace with actual file path
    output_file = "\Diploma\data_march\weekday_hour_log_visit.csv"  # Replace with desired output file path
    necessary_columns = ['user_id', 'visit_first_action_time']
    
    filtered_df = process_csv(file_path, necessary_columns, output_file)
    print(filtered_df.head())
