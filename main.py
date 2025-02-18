import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Download latest version
path = kagglehub.dataset_download("lakshmi25npathi/online-retail-dataset")

# Load the dataset
# Assuming the dataset is in CSV format and located in the same directory
df = pd.read_excel(path+'/online_retail_II.xlsx', engine='openpyxl')

# Data Preprocessing
def preprocess_data(df):
    # Remove rows with missing values
    df_clean = df.dropna()
    
    # Calculate total amount for each transaction
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']
    
    # Convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    return df_clean

# Customer Segmentation Analysis
def create_customer_segments(df):
    # Calculate key metrics for each customer
    customer_metrics = df.groupby('Customer ID').agg({
        'Invoice': 'count',
        'TotalAmount': 'sum',
        'Quantity': 'sum'
    }).rename(columns={
        'Invoice': 'PurchaseFrequency',
        'TotalAmount': 'TotalSpent',
        'Quantity': 'TotalItems'
    })
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(customer_metrics)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    customer_metrics['Segment'] = kmeans.fit_predict(features_scaled)
    
    return customer_metrics

# Marketing Analysis
def analyze_purchase_patterns(df):
    # Analyze monthly sales trends
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['TotalAmount'].sum()
    
    # Analyze country-wise distribution
    country_sales = df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False)
    
    return monthly_sales, country_sales

def main():
    # Preprocess data
    df_clean = preprocess_data(df)
    
    # Create customer segments
    customer_segments = create_customer_segments(df_clean)
    
    # Analyze patterns
    monthly_sales, country_sales = analyze_purchase_patterns(df_clean)
    
    # Visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot customer segments
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=customer_segments, x='TotalSpent', y='PurchaseFrequency', hue='Segment')
    plt.title('Customer Segments')
    
    # Plot monthly sales trends
    plt.subplot(1, 2, 2)
    monthly_sales.plot()
    plt.title('Monthly Sales Trends')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis_results.png')
    
    # Print segment characteristics
    print("\nCustomer Segment Characteristics:")
    print(customer_segments.groupby('Segment').mean())
    
    # Print top countries by sales
    print("\nTop 5 Countries by Sales:")
    print(country_sales.head())

if __name__ == "__main__":
    main()