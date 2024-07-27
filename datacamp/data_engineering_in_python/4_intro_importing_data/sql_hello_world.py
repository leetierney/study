# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///datasets/Chinook.sqlite')

# Perform query and save results to DataFrame: df
df = pd.read_sql_query("SELECT * FROM album", engine)

# Print head of DataFrame df
print(df.head())