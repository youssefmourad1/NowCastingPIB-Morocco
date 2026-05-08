import json
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_masi_data(json_path, output_path):
    try:
        logger.info(f"Loading JSON from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)

        items = data.get('data', [])
        logger.info(f"Found {len(items)} items in JSON")

        records = []
        for item in items:
            attrs = item.get('attributes', {})
            date_str = attrs.get('field_seance_date')
            value_str = attrs.get('field_index_value')
            
            if date_str and value_str:
                records.append({
                    'Date': pd.to_datetime(date_str),
                    'Masi_construction': float(value_str)
                })

        df = pd.DataFrame(records)
        
        if df.empty:
            logger.warning("No data found to extract.")
            return

        # Sort by date
        df = df.sort_values('Date')

        # To get the first day of each month:
        # 1. Create a Month column
        df['Month'] = df['Date'].dt.to_period('M')
        
        # 2. Group by Month and take the first record (minimum Date)
        # This naturally picks the first available trading day if the 1st is missing.
        monthly_first_days = df.groupby('Month').first().reset_index()
        
        logger.info(f"Filtered to {len(monthly_first_days)} monthly records")

        # Select the required columns
        result_df = monthly_first_days[['Date', 'Masi_construction']]

        # Format date for better readability in Excel if needed
        # result_df['Date'] = result_df['Date'].dt.date 

        # Save to XLSX
        result_df.to_excel(output_path, index=False)
        logger.info(f"Successfully saved to {output_path}")
        
    except Exception as e:
        logger.error(f"An error occurred during extraction: {e}")
        raise

def merge_excel_files(file_list, output_path):
    try:
        logger.info(f"Merging files: {file_list}")
        dfs = []
        for file in file_list:
            logger.info(f"Reading {file}")
            df = pd.read_excel(file)
            dfs.append(df)
        
        # Combine all DataFrames
        final_df = pd.concat(dfs, ignore_index=True)
        
        # Convert Date to datetime for reliable sorting
        final_df['Date'] = pd.to_datetime(final_df['Date'])
        
        # Sort by date and drop duplicates
        final_df = final_df.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
        
        # Save to final Excel
        final_df.to_excel(output_path, index=False)
        logger.info(f"Successfully merged into {output_path}")
        logger.info(f"Final record count: {len(final_df)}")
        
    except Exception as e:
        logger.error(f"An error occurred during merging: {e}")
        raise

if __name__ == "__main__":
    base_path = '/Users/Apple/Desktop/projects/Tennis AI v2.0/Tennis AI Analysis/'
    
    # 1. Extraction (already done, but keeping the logic here for reference/re-run if needed)
    # files_to_process = [
    #     ('extract.json', 'masi_construction.xlsx'),
    #     ('extract2.json', 'masi_construction2.xlsx'),
    #     ('extract3.json', 'masi_construction3.xlsx')
    # ]
    # for in_f, out_f in files_to_process:
    #     extract_masi_data(base_path + in_f, base_path + out_f)

    # 2. Merge
    excel_files = [
        base_path + 'masi_construction.xlsx',
        base_path + 'masi_construction2.xlsx',
        base_path + 'masi_construction3.xlsx'
    ]
    merge_excel_files(excel_files, base_path + 'masi_construction_final.xlsx')
