#!/usr/bin/env python3
import sys
import pandas as pd

def excel_to_markdown(filename, sheet_name=0):
    # Read the specified sheet from the Excel file
    df = pd.read_excel(filename, sheet_name=sheet_name)
    
    # Convert dataset IDs into Markdown links
    # Format: [dataset_id](https://nemar.org/dataexplorer/detail?dataset_id=dataset_id)
    df['DatasetID'] = df['DatasetID'].astype(str).apply(
        lambda x: f"[{x}](https://nemar.org/dataexplorer/detail?dataset_id={x})"
    )
    
    # Replace "Schizophrenia/Psychosis" with "Psychosis" in the entire DataFrame
    df = df.replace("Schizophrenia/Psychosis", "Psychosis")
    
    # Convert the DataFrame to a Markdown table (excluding the index)
    markdown = df.to_markdown(index=False)
    return markdown

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <excel_filename> [sheet_name]")
        sys.exit(1)
    
    excel_filename = sys.argv[1]
    sheet = sys.argv[2] if len(sys.argv) > 2 else 0

    markdown_table = excel_to_markdown(excel_filename, sheet)
    print(markdown_table)
