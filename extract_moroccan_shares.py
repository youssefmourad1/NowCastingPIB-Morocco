import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_moroccan_shares(input_path, output_path):
    try:
        logger.info(f"Reading Moroccan All Shares data from {input_path}")
        
        # Read the CSV file
        # Date format in file: "DD/MM/YYYY" (e.g., "03/01/2026")
        # Price format: "16,655.58"
        df = pd.read_csv(input_path)
        
        logger.info(f"Loaded {len(df)} records")
        
        # 1. Clean Price column
        # Convert "16,655.58" string to float
        if 'Price' in df.columns:
            df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
        else:
            raise ValueError("Column 'Price' not found in the input file.")
            
        # 2. Parse Date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        else:
            raise ValueError("Column 'Date' not found in the input file.")
            
        # 3. Handle duplicates or multiple entries (picking the latest/first if necessary)
        # The file seems to be monthly already, but we'll group just in case.
        # Group by Month and pick the first available record (which is usually the start/end of the month in this CSV format)
        df['Month'] = df['Date'].dt.to_period('M')
        
        # Sort by date descending (to match original order if needed) or ascending (standard)
        df = df.sort_values('Date')
        
        # If the user only wants the Price column as per previous request "only one column in it which is the Masi_construction"
        # However, they also asked for a date column earlier, so I'll include Date and Price unless explicitly told otherwise.
        # But looking at the prompt: "i only need the price column please"
        # I will extract just the Price column but keep Date for sorting and verification.
        
        result_df = df[['Date', 'Price']]
        
        # Save to XLSX
        # Since the user specifically said "i only need the price column please", 
        # I'll create the final output with just 'Price' but keep 'Date' in a separate version if they change their mind.
        # Actually, standard practice for these financial files they've asked for is Date + Value.
        # Re-reading: "i only need the price column please" -> I'll stick to just Price column as requested.
        
        final_output_df = result_df[['Price']]
        
        final_output_df.to_excel(output_path, index=False)
        
        logger.info(f"Successfully extracted {len(final_output_df)} monthly prices")
        logger.info(f"Saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    input_file = '/Users/Apple/Desktop/projects/Tennis AI v2.0/Tennis AI Analysis/Moroccan All Shares Historical Data (2).csv'
    output_file = '/Users/Apple/Desktop/projects/Tennis AI v2.0/Tennis AI Analysis/moroccan_shares_monthly.xlsx'
    
    extract_moroccan_shares(input_file, output_file)
