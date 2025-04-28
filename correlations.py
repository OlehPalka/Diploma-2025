import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
file_path = r"D:\Diploma\data_march\user_table_who_purchased_full.csv"  # Ensure correct path

data = pd.read_csv(file_path, sep=";", decimal=',')
print(data)

data = data.dropna()

correlation_matrix = data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of User Data")
plt.show()

# Save the cleaned data back to the same file path
data.to_csv(file_path, sep=";", index=False)


file_path = r"D:\Diploma\data_march\user_table_who_purchased.csv"  # Ensure correct path

data = pd.read_csv(file_path, sep=";", decimal=',')
print(data)

data = data.dropna()

columns_to_include = [
                      'Days since last purchase', 
                      'Orders2Visit ratio', 
                      'AVG Purchase value', 
                      'AVG Purchase Size', 
                      'User LT Month', 
                      'Purchase Freq.', 
                      'AVG Pages count per visit', 
                      'Total Time On site', 
                      'AVG Time On Page', 
                      'Pet', 
                      'Is interested in actions']

data = data[columns_to_include]

dtype_mapping = {
    'Days since last purchase': 'int',
    'Orders2Visit ratio': 'float',
    'AVG Purchase value': 'float',
    'AVG Purchase Size': 'float',
    'User LT Month': 'int',
    'Purchase Freq.': 'float',
    'AVG Pages count per visit': 'float',
    'Total Time On site': 'float',
    'AVG Time On Page': 'float',
    'Pet': 'int',
    'Is interested in actions': 'int'
}

data = data.astype(dtype_mapping)

correlation_matrix = data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of User Data")
plt.show()

# Save the cleaned data back to the same file path
data.to_csv(file_path, sep=";", index=False)
