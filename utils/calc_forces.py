import os
import pandas as pd
import numpy as np
import glob
import re
from pathlib import Path

def calculate_forces_from_stress(df):
    """
    Calculate forces using the stress tensor components.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the solid interface data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added force columns
    """
    # Calculate approximate center and radius
    center_x = np.mean(df['x_0'])
    center_y = np.mean(df['x_1'])
    
    # Get all points for this time step
    n_points = len(df)
    
    # Calculate radius for each point
    radii = np.sqrt((df['x_0'] - center_x)**2 + (df['x_1'] - center_y)**2)
    radius = np.mean(radii)
    
    # Calculate normal vectors
    nx = (df['x_0'] - center_x) / radii
    ny = (df['x_1'] - center_y) / radii
    
    # Calculate differential surface element
    perimeter = 2 * np.pi * radius
    ds = perimeter / n_points
    
    # Calculate total stress tensor
    sigma_00 = df['sigma_dev_00'] + df['sigma_dil_00']
    sigma_01 = df['sigma_dev_01'] + df['sigma_dil_01']
    sigma_10 = df['sigma_dev_10'] + df['sigma_dil_10']
    sigma_11 = df['sigma_dev_11'] + df['sigma_dil_11']
    
    # Calculate traction vector
    traction_x = sigma_00 * nx + sigma_01 * ny
    traction_y = sigma_10 * nx + sigma_11 * ny
    
    # Calculate force by integrating traction
    force_x = traction_x * ds
    force_y = traction_y * ds
    
    # Add calculated columns to DataFrame
    df['f_x_calc'] = force_x
    df['f_y_calc'] = force_y
    
    return df

def combine_solid_csv_files_with_forces(input_dir, output_file='combined_solid_data_with_forces.csv'):
    """
    Combine all solid CSV files into a single CSV file with calculated forces.
    
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
    csv_files = sorted(glob.glob(file_pattern))
    
    if not csv_files:
        print(f"No CSV files found with pattern 'time_*.csv' in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Initialize a list to store DataFrames
    all_data = []
    
    # Track any missing required columns
    missing_columns_warning = False
    
    # Process each CSV file
    for i, file_path in enumerate(csv_files):
        # Print progress every 10 files or for the first and last file
        if i % 10 == 0 or i == len(csv_files) - 1:
            print(f"Processing file {i+1}/{len(csv_files)}: {os.path.basename(file_path)}")
        
        # Extract time step from filename (e.g., time_8.csv -> 8)
        file_name = os.path.basename(file_path)
        match = re.search(r'time_(\d+)\.csv', file_name)
        
        if not match:
            print(f"Skipping file with invalid name format: {file_name}")
            continue
        
        time_index = int(match.group(1))
        time_value = time_index * 0.01  # Assuming time step is 0.01
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Verify that the file appears to be a solid file by checking for expected columns
            solid_columns = ['x_0', 'x_1', 'u_0', 'u_1']
            if not all(col in df.columns for col in solid_columns):
                print(f"Skipping {file_name} as it doesn't have all required solid columns")
                continue
            
            # Check for stress tensor columns
            stress_columns = ['sigma_dev_00', 'sigma_dev_01', 'sigma_dev_10', 'sigma_dev_11',
                              'sigma_dil_00', 'sigma_dil_01', 'sigma_dil_10', 'sigma_dil_11']
            
            # If any stress columns are missing, create placeholder warning
            if not all(col in df.columns for col in stress_columns):
                missing_cols = [col for col in stress_columns if col not in df.columns]
                if not missing_columns_warning:
                    print(f"Warning: Some files are missing stress tensor columns: {missing_cols}")
                    missing_columns_warning = True
                
                # Create placeholder columns for missing stress components
                for col in missing_cols:
                    if col.startswith('sigma_dev'):
                        df[col] = 0.0  # Zero deviatoric stress
                    elif col.startswith('sigma_dil'):
                        # Use negative pressure for dilatational stress diagonal components
                        if col in ['sigma_dil_00', 'sigma_dil_11']:
                            if 'p_f' in df.columns:
                                df[col] = -df['p_f'] * 3  # p = -trace(sigma_dil)/3
                            else:
                                df[col] = 0.0
                        else:
                            df[col] = 0.0  # Zero off-diagonal components
            
            # Add time column based on file name
            df['time'] = time_value
            
            # Check if pressure column 'p_f' exists, otherwise use a placeholder
            if 'p_f' not in df.columns:
                df['p_f'] = 0.0
                print(f"Note: Created placeholder 'p_f' column for {file_name}")
            
            # Calculate forces using stress tensor approach
            try:
                df = calculate_forces_from_stress(df)
            except Exception as calc_error:
                print(f"Error calculating forces for {file_name}: {str(calc_error)}")
                # Create placeholder force columns
                df['f_x_calc'] = np.nan
                df['f_y_calc'] = np.nan
            
            # Check if original force columns exist
            if 'f_0' not in df.columns:
                df['f_0'] = np.nan
                print(f"Note: Created placeholder 'f_0' column for {file_name}")
            
            if 'f_1' not in df.columns:
                df['f_1'] = np.nan
                print(f"Note: Created placeholder 'f_1' column for {file_name}")
            
            # Reorder and rename columns
            column_mapping = {
                'time': 'time',
                'x_0': 'x',
                'x_1': 'y',
                'u_0': 'u',
                'u_1': 'v',
                'p_f': 'p',
                'f_0': 'fx_ibamr',
                'f_1': 'fy_ibamr',
                'f_x_calc': 'fx_calc',
                'f_y_calc': 'fy_calc'
            }
            
            # Select and reorder columns (include all original columns plus calculated ones)
            cols_to_keep = ['time', 'x_0', 'x_1', 'u_0', 'u_1', 'p_f', 'f_0', 'f_1', 'f_x_calc', 'f_y_calc']
            
            # Make sure all columns exist
            for col in cols_to_keep:
                if col not in df.columns:
                    df[col] = np.nan
            
            df_reordered = df[cols_to_keep]
            
            # Rename columns using the mapping
            df_renamed = df_reordered.rename(columns=column_mapping)
            
            # Append to the list
            all_data.append(df_renamed)
            
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
    
    return combined_df

def batch_process_with_forces(input_dir, output_dir, batch_size=10):
    """
    Process CSV files in batches, calculate forces, and save combined results.
    Useful for large datasets that may not fit in memory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the solid CSV files
    output_dir : str
        Directory to save output files
    batch_size : int
        Number of time steps to process in each batch
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files with pattern time_*.csv
    file_pattern = os.path.join(input_dir, 'time_*.csv')
    csv_files = sorted(glob.glob(file_pattern))
    
    if not csv_files:
        print(f"No CSV files found with pattern 'time_*.csv' in {input_dir}")
        return
    
    # Group files into batches
    batches = [csv_files[i:i+batch_size] for i in range(0, len(csv_files), batch_size)]
    
    print(f"Processing {len(csv_files)} files in {len(batches)} batches")
    
    # Process each batch
    batch_dfs = []
    
    for batch_idx, batch_files in enumerate(batches):
        print(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch_files)} files")
        
        # Process files in this batch
        batch_data = []
        
        for file_path in batch_files:
            # Extract time from filename
            file_name = os.path.basename(file_path)
            match = re.search(r'time_(\d+)\.csv', file_name)
            
            if not match:
                continue
                
            time_index = int(match.group(1))
            time_value = time_index * 0.01
            
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Add time column
                df['time'] = time_value
                
                # Calculate forces
                df = calculate_forces_from_stress(df)
                
                # Rename columns
                column_mapping = {
                    'time': 'time',
                    'x_0': 'x',
                    'x_1': 'y',
                    'u_0': 'u',
                    'u_1': 'v',
                    'p_f': 'p',
                    'f_0': 'fx_ibamr',
                    'f_1': 'fy_ibamr',
                    'f_x_calc': 'fx_calc',
                    'f_y_calc': 'fy_calc'
                }
                
                # Select required columns
                cols_to_keep = ['time', 'x_0', 'x_1', 'u_0', 'u_1', 'p_f', 'f_0', 'f_1', 'f_x_calc', 'f_y_calc']
                for col in cols_to_keep:
                    if col not in df.columns:
                        df[col] = np.nan
                
                df_reordered = df[cols_to_keep]
                df_renamed = df_reordered.rename(columns=column_mapping)
                
                batch_data.append(df_renamed)
            
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
        
        if not batch_data:
            continue
            
        # Combine batch data
        batch_df = pd.concat(batch_data, ignore_index=True)
        
        # Save batch to CSV
        batch_output = os.path.join(output_dir, f"solid_data_batch_{batch_idx+1:03d}.csv")
        batch_df.to_csv(batch_output, index=False)
        print(f"Saved batch {batch_idx+1} to {batch_output}")
        
        # Store batch DataFrame reference
        batch_dfs.append(batch_output)
    
    # Create a summary file with batch information
    summary_file = os.path.join(output_dir, "batch_processing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Total files processed: {len(csv_files)}\n")
        f.write(f"Number of batches: {len(batches)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write("\nBatch files:\n")
        for idx, batch_file in enumerate(batch_dfs):
            f.write(f"Batch {idx+1}: {os.path.basename(batch_file)}\n")
    
    print(f"Batch processing complete. Summary saved to {summary_file}")
    return batch_dfs

if __name__ == "__main__":
    # Get input and output directories
    input_dir = "./data/2D_FSI_Cavity_Data/Solid-newset"
    output_file = "./data/processed_dataset/combined_solid_data_with_forces.csv"
    
    # For small datasets, use the combined approach
    combine_solid_csv_files_with_forces(input_dir, output_file)
    
    # For large datasets that may not fit in memory, uncomment this:
    # output_dir = "./data/processed_dataset/batched_data"
    # batch_process_with_forces(input_dir, output_dir, batch_size=10)