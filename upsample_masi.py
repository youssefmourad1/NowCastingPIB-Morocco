import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upsample_trimester_to_monthly(input_path, output_path):
    try:
        logger.info(f"Reading trimester data from {input_path}")
        
        # Read the tab-delimited file. No header in the original file.
        # Column 0: Year:Trimester, Column 1: Value
        df = pd.read_csv(input_path, sep='\t', header=None, names=['Period', 'Value'])
        
        logger.info(f"Loaded {len(df)} trimester records")
        
        monthly_records = []
        
        for index, row in df.iterrows():
            period = str(row['Period'])
            value = row['Value']
            
            # Split "2014:1" into Year and Trimester
            if ':' in period:
                year, trimester = period.split(':')
                year = int(year)
                trimester = int(trimester)
                
                # Each trimester has 3 months
                # T1 -> 1, 2, 3
                # T2 -> 4, 5, 6
                # T3 -> 7, 8, 9
                # T4 -> 10, 11, 12
                start_month = (trimester - 1) * 3 + 1
                
                for i in range(3):
                    current_month = start_month + i
                    # Create a date object (using the 1st of the month)
                    date_obj = pd.to_datetime(f"{year}-{current_month:02d}-01")
                    
                    monthly_records.append({
                        'Date': date_obj,
                        'Value': value
                    })
            else:
                logger.warning(f"Skipping malformed period: {period}")

        result_df = pd.DataFrame(monthly_records)
        
        # Sort by date
        result_df = result_df.sort_values('Date').reset_index(drop=True)
        
        # Save to XLSX
        result_df.to_excel(output_path, index=False)
        logger.info(f"Successfully upsampled to {len(result_df)} monthly records")
        logger.info(f"Saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    input_file = '/Users/Apple/Desktop/projects/Tennis AI v2.0/Tennis AI Analysis/extract3.csv'
    output_file = '/Users/Apple/Desktop/projects/Tennis AI v2.0/Tennis AI Analysis/extract3_monthly.xlsx'
    
    upsample_trimester_to_monthly(input_file, output_file)
