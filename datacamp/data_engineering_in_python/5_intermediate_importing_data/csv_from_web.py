# Import packages
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import pandas as pd

# Assign url of file: url
url = 'https://assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'

def read_and_save(url):
    # Save file locally
    urlretrieve(url, filename='datasets/winequality-red.csv')

    # Read file into a DataFrame and print its head
    df = pd.read_csv('datasets/winequality-red.csv', sep=';')
    print(df.head())

def read_from_web(url):
    # Read file into a DataFrame: df
    df = pd.read_csv(url, sep = ';')

    # Print the head of the DataFrame
    print(df.head())

    # Plot first column of df
    df.iloc[:, 0].hist()
    plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
    plt.ylabel('count')
    plt.show()

read_from_web(url)