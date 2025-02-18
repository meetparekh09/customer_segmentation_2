import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import kagglehub

st.set_page_config(page_title="E-commerce Analytics Dashboard", layout="wide")

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("lakshmi25npathi/online-retail-dataset")
    df = pd.read_excel(path+'/online_retail_II.xlsx', engine='openpyxl')
    return df

def preprocess_data(df):
    df_clean = df.copy()
    # Remove rows with missing values
    df_clean = df_clean.dropna()
    
    # Calculate total amount for each transaction
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']
    
    # Convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    return df_clean

def create_customer_segments(df, n_clusters):
    customer_metrics = df.groupby('Customer ID').agg({
        'Invoice': 'count',
        'TotalAmount': 'sum',
        'Quantity': 'sum'
    }).rename(columns={
        'Invoice': 'PurchaseFrequency',
        'TotalAmount': 'TotalSpent',
        'Quantity': 'TotalItems'
    })
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(customer_metrics)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_metrics['Segment'] = kmeans.fit_predict(features_scaled)
    
    return customer_metrics

def main():
    st.title("E-commerce Analytics Dashboard")
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
        df_clean = preprocess_data(df)
    
    # Sidebar controls
    st.sidebar.header("Controls")
    n_clusters = st.sidebar.slider("Number of Customer Segments", 2, 6, 4)
    date_range = st.sidebar.date_input(
        "Date Range",
        [df_clean['InvoiceDate'].min(), df_clean['InvoiceDate'].max()]
    )
    
    # Filter data by date
    mask = (df_clean['InvoiceDate'].dt.date >= date_range[0]) & (df_clean['InvoiceDate'].dt.date <= date_range[1])
    df_filtered = df_clean[mask]
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Customer Segments", "Sales Analysis", "Product Analysis"])
    
    with tab1:
        st.header("Customer Segmentation")
        
        # Create customer segments
        customer_segments = create_customer_segments(df_filtered, n_clusters)
        
        # Scatter plot of customer segments
        fig_segments = px.scatter(
            customer_segments,
            x='TotalSpent',
            y='PurchaseFrequency',
            color='Segment',
            title='Customer Segments by Spending and Purchase Frequency'
        )
        st.plotly_chart(fig_segments, use_container_width=True)
        
        # Segment characteristics
        st.subheader("Segment Characteristics")
        segment_stats = customer_segments.groupby('Segment').mean().round(2)
        st.dataframe(segment_stats)
    
    with tab2:
        st.header("Sales Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly sales trend
            monthly_sales = df_filtered.groupby(df_filtered['InvoiceDate'].dt.to_period('M'))['TotalAmount'].sum()
            fig_monthly = px.line(
                x=monthly_sales.index.astype(str),
                y=monthly_sales.values,
                title='Monthly Sales Trend'
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            # Country-wise sales
            country_sales = df_filtered.groupby('Country')['TotalAmount'].sum().sort_values(ascending=True)
            fig_country = px.bar(
                x=country_sales.values,
                y=country_sales.index,
                orientation='h',
                title='Sales by Country'
            )
            st.plotly_chart(fig_country, use_container_width=True)
    
    with tab3:
        st.header("Product Analysis")
        
        # Top products
        top_products = df_filtered.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        fig_products = px.bar(
            x=top_products.index,
            y=top_products.values,
            title='Top 10 Products by Quantity Sold'
        )
        fig_products.update_xaxes(tickangle=45)
        st.plotly_chart(fig_products, use_container_width=True)
        
        # Product search
        search_term = st.text_input("Search for a product:")
        if search_term:
            product_results = df_filtered[df_filtered['Description'].str.contains(search_term, case=False)]
            if not product_results.empty:
                st.dataframe(product_results[['Description', 'Quantity', 'Price', 'Country']].head())
            else:
                st.write("No products found matching your search.")

if __name__ == "__main__":
    main() 