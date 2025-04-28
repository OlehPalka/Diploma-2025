import pandas as pd
from datetime import datetime

def create_product_table(file_path):
    df = pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
    df['Date'] = pd.to_datetime(df['Date'], format="%d.%m.%Y %H:%M")
    latest_prices = df.sort_values('Date').groupby('Idaction Name').last()['Price']
    product_popularity = df['Idaction Name'].value_counts()

    order_groups = df.groupby('OrderId')['Idaction Name'].apply(list)
    product_pairs = []

    for products in order_groups:
        if len(products) > 1:
            for product in set(products):
                co_purchased = [co for co in products if co != product]
                product_pairs.extend([(product, co_product) for co_product in co_purchased])

    pairs_df = pd.DataFrame(product_pairs, columns=['idaction_sku', 'OftenBoughtWith'])
    often_bought_with = pairs_df.groupby('idaction_sku')['OftenBoughtWith'] \
                            .agg(lambda x: int(x.value_counts().idxmax()))



    product_table = pd.DataFrame({
        'Idaction Name': product_popularity.index,
        'FullName': df.groupby('Idaction Name').last()['Name'],
        'ProductSku': df.groupby('Idaction Name').last()['idaction_sku'],
        'Price': latest_prices,
        'ProductPopularity': product_popularity.astype(int),
        'OftenBoughtWith': often_bought_with.astype(int),
        'Product Cat/Dog': df.groupby('Idaction Name').last()['Product Cat\Dog'],
        'ProductCategory': df.groupby('Idaction Name').last()['Product category'],
    }).reset_index(drop=True)

    product_table.to_csv(r'D:\Diploma\data_march\Product_table_club4paws.csv', index=False, sep=';', encoding="utf-8-sig")
    return r'D:\Diploma\data_march\Product_table_club4paws.csv'


create_product_table(r"D:\Diploma\data_march\Club4Paws_orders_products.csv")

