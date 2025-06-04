import os
import pandas as pd
import glob
import re

def combine_solid_csv_files(input_dir, output_file='combined_solid_data.csv'):
    """
    Combine all solid CSV files into a single CSV file with reordered columns.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the solid CSV files (time_*.csv)
    output_file : str
        Output file path for the combined CSV
    """
    print(f"Looking for solid CSV files in: {input_dir}")
    
    # Find all CSV files with pattern time_*.csv
    file_pattern = os.path.join(input_dir, 'time_*.csv')
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        print(f"No CSV files found with pattern 'time_*.csv' in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Initialize a list to store DataFrames
    all_data = []
    
    # Process each CSV file
    for file_path in csv_files:
        # Extract time step from filename (e.g., time_8.csv -> 8)
        file_name = os.path.basename(file_path)
        match = re.search(r'time_(\d+)\.csv', file_name)
        
        if not match:
            print(f"Skipping file with invalid name format: {file_name}")
            continue
        
        time_index = int(match.group(1))
        time_value = time_index * 0.01
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Verify that the file appears to be a solid file by checking for expected columns
            solid_columns = ['x_0', 'x_1', 'u_0', 'u_1']
            if not all(col in df.columns for col in solid_columns):
                print(f"Skipping {file_name} as it doesn't have all required solid columns")
                continue
            
            # Add time column based on file name
            df['time'] = time_value
            
            # Check if pressure column 'p_f' exists, otherwise use a placeholder
            if 'p_f' not in df.columns:
                df['p_f'] = 0.0
                print(f"Note: Created placeholder 'p_f' column for {file_name}")
            
            # Reorder and rename columns
            column_mapping = {
                'time': 'time',
                'x_0': 'x',
                'x_1': 'y',
                'u_0': 'u',
                'u_1': 'v',
                'p_f': 'p',
                'f_0': 'fx',
                'f_1': 'fy'
            }
            
            # Check if force columns exist
            if 'f_0' not in df.columns:
                df['f_0'] = 0.0
                print(f"Note: Created placeholder 'f_0' column for {file_name}")
            
            if 'f_1' not in df.columns:
                df['f_1'] = 0.0
                print(f"Note: Created placeholder 'f_1' column for {file_name}")
            
            # Select columns for reordering and renaming
            cols_to_keep = ['time', 'x_0', 'x_1', 'u_0', 'u_1', 'p_f', 'f_0', 'f_1']
            df_reordered = df[cols_to_keep]
            
            # Rename columns using the mapping
            df_renamed = df_reordered.rename(columns=column_mapping)
            
            # Append to the list
            all_data.append(df_renamed)
            
            # Print progress every 10 files
            if len(all_data) % 10 == 0:
                print(f"Processed {len(all_data)} files...")
                
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    if not all_data:
        print("No valid solid data files were processed.")
        return
    
    # Combine all DataFrames
    print("Combining all data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    print(f"Saving combined data to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    
    print(f"Combined data saved successfully!")
    print(f"Total number of rows: {len(combined_df)}")
    print(f"Time steps included: {combined_df['time'].nunique()}")
    print(f"Columns: {', '.join(combined_df.columns)}")


if __name__ == "__main__":
    # Get input directory from user
    input_dir = "./data/2D_FSI_Cavity_Data/Solid-newset"
    output_file = "./data/processed_dataset/combined_solid_data.csv"
    
    combine_solid_csv_files(input_dir, output_file)