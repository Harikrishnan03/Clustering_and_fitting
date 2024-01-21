# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:57:54 2024

@author: Harikrishnan Marimuthu
"""

# Importing packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.cluster import KMeans

# Define a polynomial function:


def polynomial_function(x, a, b, c):
    """
    Returns the value of a quadratic polynomial function.

    Parameters:
    - x (float): Input value.
    - a, b, c (float): Coefficients of the quadratic polynomial function.

    Returns:
    float: Result of the quadratic polynomial function.
    """
    return a * x ** 2 + b * x + c

# Function to calculate the standard deviation of residuals:


def calculate_residual_stddev(x, y, degree):
    """
    Calculates the standard deviation of residuals for a polynomial fit.

    Parameters:
    - x (array-like): Independent variable data.
    - y (array-like): Dependent variable data.
    - degree (int): Degree of the polynomial fit.

    Returns:
    float: Standard deviation of residuals.
    """
    coefficients = np.polyfit(x, y, degree)
    y_estimate = np.polyval(coefficients, x)
    residuals = y - y_estimate
    return np.std(residuals)

# Function to read data from an Excel file:


def read_excel_data(file_path, skip_rows):
    """
    Reads data from an Excel file and transposes the data.

    Parameters:
    - file_path (str): Path to the Excel file.
    - skip_rows (int): Number of rows to skip while reading the file.

    Returns:
    tuple: A tuple containing the original data and its transpose.
    """
    data = pd.read_excel(file_path, skiprows=skip_rows)
    return data, data.transpose()

# Function to prepare methane emissions and population data:


def prepare_methane_and_population_data(file_path, skip_rows):
    """
    Prepares methane emissions and population data for the years 1990 and 2020.

    Parameters:
    - file_path (str): Path to the Excel file containing the data.
    - skip_rows (int): Number of rows to skip while reading the file.

    Returns:
    tuple: Two DataFrames containing methane emissions per head, GDP 
    per capita,
    and relevant data for the years 1990 and 2020.
    """
    data, data_transpose = read_excel_data(file_path, skip_rows)

    methane_data = data[data['Indicator Name'] ==
                        'Methane emissions (kt of CO2 equivalent)']
    methane_data = methane_data.drop(
        ['Country Code', 'Indicator Code', 'Indicator Name'], axis=1)
    required_columns = ['Country Name', '1990', '2020']
    methane_data = methane_data[required_columns]
    methane_data = methane_data.dropna(how='any')
    methane_data = methane_data.set_index('Country Name')

    population_data = data[data['Indicator Name'] == 'Population, total']
    population_data = population_data[required_columns]
    population_data = population_data[population_data['Country Name'].isin(
        methane_data.index)]
    population_data = population_data.set_index('Country Name')

    methane_data['1990'] = methane_data['1990'] / population_data['1990']
    methane_data['2020'] = methane_data['2020'] / population_data['2020']

    gdp_data, gdp_data_transpose = read_excel_data(
        'API_NY.GDP.PCAP.KD.ZG_DS2_en_excel_v2_6298376.xls', 3)

    gdp_data = gdp_data[required_columns]
    gdp_data = gdp_data[gdp_data['Country Name'].isin(methane_data.index)]
    gdp_data = gdp_data.set_index('Country Name')

    data_1990 = methane_data.drop(['2020'], axis=1)
    data_1990 = data_1990.rename(
        columns={'1990': 'Methane emissions per head'})
    data_1990['GDP per capita'] = gdp_data['1990']
    data_1990 = data_1990.dropna(how='any')

    data_2020 = methane_data.drop(['1990'], axis=1)
    data_2020 = data_2020.rename(
        columns={'2020': 'Methane emissions per head'})
    data_2020['GDP per capita'] = gdp_data['2020']
    data_2020 = data_2020.dropna(how='any')

    return data_1990, data_2020

# Function to visualize data:


def visualize_data(data_1990, data_2020):
    """
    Creates a scatter plot to visualize the relationship between methane
    emissions per head
    and GDP per capita for the years 1990 and 2020.

    Parameters:
    - data_1990 (pd.DataFrame): DataFrame containing data for the year 1990.
    - data_2020 (pd.DataFrame): DataFrame containing data for the year 2020.

    Returns:
    None
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(data_1990['Methane emissions per head'],
                   data_1990['GDP per capita'], c='chocolate')
    axs[0].set_title('1990 Clusters: Methane Emissions vs GDP per Capita',
                     fontweight='bold', fontsize=10)
    axs[0].set_xlabel('Methane emissions per head', fontweight='bold')
    axs[0].set_ylabel('GDP per capita', fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    axs[1].scatter(data_2020['Methane emissions per head'],
                   data_2020['GDP per capita'], c='darkslateblue')
    axs[1].set_title('2000 Clusters: Methane Emissions vs GDP per Capita',
                     fontweight='bold', fontsize=10)
    axs[1].set_xlabel('Methane emissions per head', fontweight='bold')
    axs[1].set_ylabel('GDP per capita', fontweight='bold')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    for ax in axs:
        ax.set_facecolor('azure')
    plt.tight_layout()
    plt.show()

# Function to normalize data:


def normalize_data(data_1990, data_2020):
    """
    Normalizes the data for clustering.

    Parameters:
    - data_1990 (pd.DataFrame): DataFrame containing data for the year 1990.
    - data_2020 (pd.DataFrame): DataFrame containing data for the year 2020.

    Returns:
    tuple: Two NumPy arrays containing normalized data for the years 1990 and
    2020.
    """
    p1 = data_1990[['Methane emissions per head', 'GDP per capita']].values
    p2 = data_2020[['Methane emissions per head', 'GDP per capita']].values

    max_value1 = p1.max()
    min_value1 = p1.min()

    max_value2 = p2.max()
    min_value2 = p2.min()

    p1_normalized = (p1 - min_value1) / (max_value1 - min_value1)
    p2_normalized = (p2 - min_value2) / (max_value2 - min_value2)

    return p1_normalized, p2_normalized

# Function to plot the elbow method:


def plot_elbow_method(p1_normalized, p2_normalized):
    """
    Plots the elbow method to determine the optimal number of clusters for
    k-means clustering.

    Parameters:
    - p1_normalized (np.ndarray): Normalized data for the year 1990.
    - p2_normalized (np.ndarray): Normalized data for the year 2020.

    Returns:
    None
    """
    cluster_1990 = []
    cluster_2000 = []

    for i in range(1, 11):
        kmeans1 = KMeans(n_clusters=i, init='k-means++',
                         max_iter=300, n_init=10, random_state=0)
        kmeans1.fit(p1_normalized)
        cluster_1990.append(kmeans1.inertia_)

        kmeans2 = KMeans(n_clusters=i, init='k-means++',
                         max_iter=300, n_init=10, random_state=0)
        kmeans2.fit(p2_normalized)
        cluster_2000.append(kmeans2.inertia_)

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, 11), cluster_1990, label='1990',
            color='steelblue', marker='o')
    ax.plot(range(1, 11), cluster_2000, label='2020',
            color='crimson', marker='o')
    ax.set_title('Elbow Method for Clustering (1990 vs 2020)',
                 fontweight='bold')
    ax.set_xlabel('Number of Clusters', fontweight='bold')
    ax.set_ylabel('Sum of Squared Errors (SSE)', fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('azure')
    plt.tight_layout()
    plt.show()

# Function to cluster data and plot clusters:


def cluster_and_plot(p1_normalized, p2_normalized):
    """
    Performs k-means clustering on normalized data and visualizes the clusters.

    Parameters:
    - p1_normalized (np.ndarray): Normalized data for the year 1990.
    - p2_normalized (np.ndarray): Normalized data for the year 2020.

    Returns:
    None
    """
    n_clusters = 5
    kmeans1 = KMeans(n_clusters=n_clusters, init='k-means++',
                     max_iter=300, n_init=10, random_state=0)
    kmeans2 = KMeans(n_clusters=n_clusters, init='k-means++',
                     max_iter=300, n_init=10, random_state=0)
    cluster_labels1 = kmeans1.fit_predict(p1_normalized)
    cluster_labels2 = kmeans2.fit_predict(p2_normalized)

    data_1990['cluster'] = cluster_labels1
    data_2020['cluster'] = cluster_labels2

    cluster_centers1 = kmeans1.cluster_centers_
    cluster_centers2 = kmeans2.cluster_centers_

    plt.style.use('default')

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=140)
    axs[0].scatter(p1_normalized[cluster_labels1 == 0, 0],
                   p1_normalized[cluster_labels1 == 0, 1], s=50,
                   c='lightcoral', label='cluster 0')
    axs[0].scatter(p1_normalized[cluster_labels1 == 1, 0],
                   p1_normalized[cluster_labels1 == 1, 1], s=50,
                   c='tan', label='cluster 1')
    axs[0].scatter(p1_normalized[cluster_labels1 == 2, 0],
                   p1_normalized[cluster_labels1 == 2, 1], s=50,
                   c='orange', label='cluster 2')
    axs[0].scatter(p1_normalized[cluster_labels1 == 3, 0],
                   p1_normalized[cluster_labels1 == 3, 1], s=50,
                   c='red', label='cluster 3')
    axs[0].scatter(p1_normalized[cluster_labels1 == 4, 0],
                   p1_normalized[cluster_labels1 == 4, 1], s=50,
                   c='green', label='cluster 4')
    axs[0].scatter(cluster_centers1[:, 0], cluster_centers1[:, 1],
                   s=50, c='black', label='Centroids')
    axs[0].set_title(
        '1990 Clusters: Methane Emissions vs GDP per Capita',
        fontweight='bold', fontsize=10)
    axs[0].set_xlabel('Methane emissions per head', fontweight='bold')
    axs[0].set_ylabel('GDP per capita', fontweight='bold')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)

    axs[1].scatter(p2_normalized[cluster_labels2 == 0, 0],
                   p2_normalized[cluster_labels2 == 0, 1], s=50,
                   c='lightcoral', label='cluster 0')
    axs[1].scatter(p2_normalized[cluster_labels2 == 1, 0],
                   p2_normalized[cluster_labels2 == 1, 1], s=50,
                   c='tan', label='cluster 1')
    axs[1].scatter(p2_normalized[cluster_labels2 == 2, 0],
                   p2_normalized[cluster_labels2 == 2, 1], s=50,
                   c='orange', label='cluster 2')
    axs[1].scatter(p2_normalized[cluster_labels2 == 3, 0],
                   p2_normalized[cluster_labels2 == 3, 1], s=50,
                   c='red', label='cluster 3')
    axs[1].scatter(p2_normalized[cluster_labels2 == 4, 0],
                   p2_normalized[cluster_labels2 == 4, 1], s=50,
                   c='green', label='cluster 4')
    axs[1].scatter(cluster_centers2[:, 0], cluster_centers2[:, 1],
                   s=50, c='black', label='Centroids')
    axs[1].set_title(
        '2000 Clusters: Methane Emissions vs GDP per Capita',
        fontweight='bold', fontsize=10)
    axs[1].set_xlabel('Methane emissions per head', fontweight='bold')
    axs[1].set_ylabel('GDP per capita', fontweight='bold')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    for ax in axs:
        ax.set_facecolor('azure')
    plt.tight_layout()
    plt.show()


# Function to analyze GDP growth for specific countries:
def analyze_gdp_growth(countries, GDP_transpose):
    """
    Analyzes and visualizes GDP growth for specific countries, including
    forecasting for 2025.

    Parameters:
    - countries (list): List of country names to analyze.
    - GDP_transpose (pd.DataFrame): Transposed GDP data.

    Returns:
    None
    """
    years = np.arange(1960, 2025)
    plt.style.use('default')
    plt.figure(dpi=600)

    for country in countries:
        plt.figure(figsize=(6, 4))
        GDP_country_data = GDP_transpose.drop(
            ['Country Code', 'Indicator Name', 'Indicator Code'])
        GDP_country_data.columns = GDP_country_data.iloc[0, :]
        GDP_country_data = GDP_country_data[1:]
        GDP_country_data = GDP_country_data[GDP_country_data.columns[
            GDP_country_data.columns.isin([country])]]
        GDP_country_data = GDP_country_data.dropna(how='any')
        GDP_country_data['Year'] = GDP_country_data.index

        GDP_fit_country_data = GDP_country_data[['Year', country]].apply(
            pd.to_numeric, errors='coerce')
        params_country, covariance_country = opt.curve_fit(
            polynomial_function, GDP_fit_country_data['Year'],
            GDP_fit_country_data[country])

        fit_and_forecast_curve = polynomial_function(years, *params_country)

        error_country = calculate_residual_stddev(
            GDP_fit_country_data['Year'], GDP_fit_country_data[country], 2)

        upper_bound = fit_and_forecast_curve + error_country
        lower_bound = fit_and_forecast_curve - error_country

        plt.plot(GDP_fit_country_data["Year"], GDP_fit_country_data[country],
                 label=f'GDP/Capita growth - {country}', color='crimson')
        plt.plot(years, fit_and_forecast_curve,
                 label="Forecast", color='dodgerblue')
        plt.fill_between(years, lower_bound, upper_bound,
                         color='bisque', alpha=0.7, label='Error range')

        prediction_for_2025 = polynomial_function(2025, *params_country)
        plt.scatter(2025, prediction_for_2025, color='midnightblue',
                    label=f'Prediction for 2025: {prediction_for_2025:.2f}')
        plt.annotate(f'{prediction_for_2025:.2f}', (2025, prediction_for_2025),
                     textcoords="offset points", xytext=(0, 10), ha='center')

        plt.xlabel("Year", fontweight='bold', fontsize=10)
        plt.xticks(np.arange(min(years), 2031, step=10))
        plt.ylabel("GDP per capita", fontweight='bold', fontsize=10)
        plt.gca().set_facecolor('azure')
        plt.legend()
        plt.title(
            f'GDP per Capita Comparison - {country}', fontweight='bold',
            fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


# Main execution:
if __name__ == "__main__":
    data_1990, data_2020 = prepare_methane_and_population_data(
        'API_19_DS2_en_excel_v2_6300761.xls', 3)
    visualize_data(data_1990, data_2020)
    p1_normalized, p2_normalized = normalize_data(data_1990, data_2020)
    plot_elbow_method(p1_normalized, p2_normalized)
    cluster_and_plot(p1_normalized, p2_normalized)
    countries_to_analyze = ['United Kingdom', 'Spain', 'Belgium']
    GDP, GDP_transpose_data = read_excel_data(
        'API_NY.GDP.PCAP.KD.ZG_DS2_en_excel_v2_6298376.xls', 3)
    analyze_gdp_growth(countries_to_analyze, GDP_transpose_data)
