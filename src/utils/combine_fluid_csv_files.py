import os
import pandas as pd
import glob
import re

def combine_fluid_csv_files(input_dir, output_file='combined_fluid_data.csv'):
    """
    Combine all fluid CSV files into a single CSV file with reordered columns.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the fluid CSV files (time_*.csv)
    output_file : str
        Output file path for the combined CSV
    """
    print(f"Looking for fluid CSV files in: {input_dir}")
    
    # Find all CSV files with pattern time_*.csv
    file_pattern = os.path.join(input_dir, 'time*.csv')
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        print(f"No CSV files found with pattern 'time*.csv' in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Initialize a list to store DataFrames
    all_data = []
    
    # Process each CSV file
    for file_path in csv_files:
        # Extract time step from filename (e.g., time_21.csv -> 21)
        file_name = os.path.basename(file_path)
        match = re.search(r'time(\d+)\.csv', file_name)
        
        if not match:
            print(f"Skipping file with invalid name format: {file_name}")
            continue
        
        time_index = int(match.group(1))
        time_value = time_index * 0.01
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Verify that the file appears to be a fluid file by checking for expected columns
            required_columns = ['PointLocations:0', 'PointLocations:1', 'U_x', 'U_y', 'P', 'F_x', 'F_y']
            if not all(col in df.columns for col in required_columns):
                print(f"Skipping {file_name} as it doesn't have all required fluid columns")
                continue
            
            # Add time column based on file name
            df['time'] = time_value
            
            # Reorder and rename columns as specified
            df_reordered = df[['time', 'PointLocations:0', 'PointLocations:1', 'U_x', 'U_y', 'P', 'F_x', 'F_y']]
            df_renamed = df_reordered.rename(columns={
                'PointLocations:0': 'x',
                'PointLocations:1': 'y',
                'U_x': 'u',
                'U_y': 'v',
                'P': 'p',
                'F_x': 'fx',
                'F_y': 'fy'
            })
            
            # Append to the list
            all_data.append(df_renamed)
            
            # Print progress every 10 files
            if len(all_data) % 10 == 0:
                print(f"Processed {len(all_data)} files...")
                
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    if not all_data:
        print("No valid fluid data files were processed.")
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
    input_dir = "./data/2D_FSI_Cavity_Data/Fluid"
    output_file = "./data/processed_dataset/combined_fluid_data.csv"
    combine_fluid_csv_files(input_dir, output_file)
    
    
    
# Fluid_data = scipy.io.loadmat("./data/Fluid_trainingData.mat")

# fluid = Fluid_data["Fluid_training"]
# interface = Fluid_data["Solid_interface"]
# solid = Fluid_data["Solid_points"]

# import pandas as pd
# interface_df = pd.DataFrame(interface, columns=["time", "x", "y", "u", "v", "p", "fx", "fy"])
# interface_df["is_interface"] = True  # Add the new column

# header = "time,x,y,u,v,p,fx,fy"
# header_interface = "time,x,y,u,v,p,fx,fy,is_interface"

# np.savetxt(os.path.join(training_data_path, "fluid.csv"), fluid, delimiter=",", comments='', header=header)
# np.savetxt(os.path.join(training_data_path, "solid.csv"), solid, delimiter=",", comments='', header=header)

# interface_df.to_csv(os.path.join(training_data_path, "interface.csv"), index=False)
