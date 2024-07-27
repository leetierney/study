from sqlalchemy import create_engine, inspect

# Create engine: engine
engine = create_engine('sqlite:///datasets/Chinook.sqlite')

# Use inspector to get the table names
inspector = inspect(engine)
table_names = inspector.get_table_names()

# Print the table names to the shell
print(table_names)