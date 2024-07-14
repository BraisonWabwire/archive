import pandas as pd

# Function to convert CSV to Excel
def convert_csv_to_excel(csv_file, excel_file):
    # Read the CSV file
    df = pd.read_csv('C:/Users/braisonW/Desktop/machine Learning/archive/AAPL_stock_data.csv')
    
    # Save to Excel file
    df.to_excel(excel_file, index=False)

# Example usage
csv_file = 'example.csv'
excel_file = 'example.xlsx'
convert_csv_to_excel(csv_file, excel_file)

print(f"CSV file '{csv_file}' has been converted to Excel file '{excel_file}'.")
