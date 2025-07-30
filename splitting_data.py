import pandas as pd 

# Load the original metadata 
path = "C:\\Users\\DELL\\Downloads\\Data_Entry_2017_v2020.csv"
df = pd.read_csv(path)

# Filter rows 
filtered_df = df[df['Image Index'].str.endswith('_001.png')]

# Keep only the columns i care about 
cleaned_df = filtered_df[['Image Index', 'Finding Labels']]

# Save the cleaned DataFrameto a new CSV file 
path = "C:\\Users\\DELL\\Downloads\\cleaned_data.csv"
cleaned_df.to_csv(path, index=False)
