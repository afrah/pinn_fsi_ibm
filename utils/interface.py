import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os
from sklearn.covariance import EllipticEnvelope
from scipy import optimize
from matplotlib.patches import Ellipse

def fit_ellipse(x, y):
    """
    Fit an ellipse to the given x, y coordinates
    
    Parameters:
    -----------
    x, y : numpy arrays
        Coordinates of solid points
        
    Returns:
    --------
    tuple
        (center_x, center_y, semi-major axis, semi-minor axis, angle)
    """
    # Stack coordinates
    coords = np.vstack([x, y])
    
    # Calculate mean (center of ellipse)
    center_x, center_y = np.mean(coords, axis=1)
    
    # Center data
    coords_centered = coords - np.array([[center_x], [center_y]])
    
    # Get covariance matrix
    cov = np.cov(coords_centered)
    
    # Eigenvalues give the squared semi-axes lengths
    evals, evecs = np.linalg.eig(cov)
    
    # Sort eigenvalues in decreasing order
    sort_indices = np.argsort(evals)[::-1]
    semi_major = np.sqrt(evals[sort_indices[0]])
    semi_minor = np.sqrt(evals[sort_indices[1]])
    
    # Angle of the ellipse (in radians)
    angle = np.arctan2(evecs[1, sort_indices[0]], evecs[0, sort_indices[0]])
    
    return center_x, center_y, semi_major, semi_minor, angle


def is_point_near_ellipse(x, y, ellipse_params, threshold=0.02):
    """
    Determine if a point is near an ellipse boundary
    
    Parameters:
    -----------
    x, y : float
        Coordinates of the point to check
    ellipse_params : tuple
        (center_x, center_y, semi-major, semi-minor, angle)
    threshold : float
        Distance threshold from ellipse boundary
        
    Returns:
    --------
    bool
        True if point is near ellipse boundary
    """
    center_x, center_y, a, b, angle = ellipse_params
    
    # Translate and rotate point to the ellipse coordinate system
    x_translated = x - center_x
    y_translated = y - center_y
    
    x_rotated = x_translated * np.cos(-angle) - y_translated * np.sin(-angle)
    y_rotated = x_translated * np.sin(-angle) + y_translated * np.cos(-angle)
    
    # Calculate normalized distance from center (1.0 means exactly on the ellipse)
    normalized_distance = np.sqrt((x_rotated/a)**2 + (y_rotated/b)**2)
    
    # Point is near ellipse if its normalized distance is close to 1.0
    return abs(normalized_distance - 1.0) <= threshold / min(a, b)


def fit_circle(x, y):
    """
    Fit a circle to the data using least squares
    
    Parameters:
    -----------
    x, y : numpy arrays
        Coordinates of solid points
        
    Returns:
    --------
    tuple
        (center_x, center_y, radius)
    """
    def calc_r(xc, yc):
        """ Calculate the distance of each data point from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    
    def f_residuals(c):
        """ Calculate the algebraic distance between the data points and the circle defined by c """
        r = calc_r(*c)
        return r - r.mean()
    
    # Initial guess: mean of data
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    
    # Fit using Scipy's least squares optimizer
    center, _ = optimize.leastsq(f_residuals, center_estimate)
    
    # Calculate radius
    radius = calc_r(*center).mean()
    
    return center[0], center[1], radius


def identify_interface_points_ellipse(fluid_data, solid_data, radius=0.2, distance_threshold=0.02, timesteps=None):
    """
    Identify interface points by fitting an ellipse/circle to the solid points
    
    Parameters:
    -----------
    fluid_data : pandas.DataFrame
        Combined fluid data with columns [time, x, y, ...]
    solid_data : pandas.DataFrame
        Combined solid data with columns [time, x, y, ...]
    radius : float
        Known radius of the disc (default: 0.2 as specified in the problem)
    distance_threshold : float
        Distance threshold from the ellipse boundary
    timesteps : list, optional
        Specific timesteps to process
        
    Returns:
    --------
    pandas.DataFrame
        Fluid data with an additional 'is_interface' column
    """
    # Get unique timesteps in both datasets
    fluid_times = fluid_data['time'].unique()
    solid_times = solid_data['time'].unique()
    
    # Find common timesteps
    common_timesteps = np.intersect1d(fluid_times, solid_times)
    
    if timesteps is not None:
        # Filter to only requested timesteps that exist in the data
        timesteps = np.intersect1d(timesteps, common_timesteps)
    else:
        timesteps = common_timesteps
    
    print(f"Processing {len(timesteps)} timesteps using ellipse/circle fitting")
    
    # Initialize a list to store interface points for each timestep
    all_interface_points = []
    
    # For rigidity check - store initial positions to compare
    initial_points = None
    
    # Process each timestep
    for t in timesteps:
        # Filter data for current timestep
        fluid_t = fluid_data[fluid_data['time'] == t]
        solid_t = solid_data[solid_data['time'] == t]
        
        if len(fluid_t) == 0 or len(solid_t) == 0:
            print(f"Skipping timestep {t}: Missing data")
            continue
        
        # Store initial configuration for first timestep
        if initial_points is None:
            initial_points = solid_t[['x', 'y']].values
        
        # Extract coordinates
        solid_coords_x = solid_t['x'].values
        solid_coords_y = solid_t['y'].values
        
        # Check if the disc retains its circular shape (rigid body)
        # For a rigid body, distances between points should remain constant
        current_points = solid_t[['x', 'y']].values
        
        # Check if shape is approximately circular by comparing distances
        is_circular = True
        
        # Fit an ellipse to the solid points
        center_x, center_y, semi_major, semi_minor, angle = fit_ellipse(solid_coords_x, solid_coords_y)
        
        # Calculate aspect ratio to check circularity
        aspect_ratio = semi_major / semi_minor
        
        if 0.95 <= aspect_ratio <= 1.05:
            # Nearly circular - use simpler circle fitting
            center_x, center_y, radius_fitted = fit_circle(solid_coords_x, solid_coords_y)
            print(f"Timestep {t}: Fitted circle with center ({center_x:.4f}, {center_y:.4f}), radius {radius_fitted:.4f}")
            
            # Identify interface points in fluid data
            interface_mask = []
            for _, row in fluid_t.iterrows():
                # Calculate distance from point to circle center
                d = np.sqrt((row['x'] - center_x)**2 + (row['y'] - center_y)**2)
                # Point is interface if it's close to the circle boundary
                interface_mask.append(abs(d - radius_fitted) <= distance_threshold)
        else:
            # Elliptical - use full ellipse parameters
            print(f"Timestep {t}: Fitted ellipse with center ({center_x:.4f}, {center_y:.4f}), "
                  f"semi-axes ({semi_major:.4f}, {semi_minor:.4f}), angle {np.degrees(angle):.2f}°")
            
            # Identify interface points in fluid data
            interface_mask = []
            ellipse_params = (center_x, center_y, semi_major, semi_minor, angle)
            for _, row in fluid_t.iterrows():
                interface_mask.append(is_point_near_ellipse(row['x'], row['y'], ellipse_params, distance_threshold))
        
        # Create a copy of the fluid data for this timestep
        fluid_t_copy = fluid_t.copy()
        
        # Add an interface flag column
        fluid_t_copy['is_interface'] = interface_mask
        
        # Add to the list of interface points
        all_interface_points.append(fluid_t_copy)
        
        print(f"Timestep {t}: Found {sum(interface_mask)} interface points")
    
    # Combine all interface points
    if all_interface_points:
        all_interface_df = pd.concat(all_interface_points, ignore_index=True)
        return all_interface_df
    else:
        print("No interface points found.")
        return None


def visualize_interface_with_ellipse(fluid_data, solid_data, interface_data, timestep, save_path=None):
    """
    Visualize the identified interface points with the fitted ellipse/circle
    
    Parameters:
    -----------
    fluid_data : pandas.DataFrame
        Combined fluid data
    solid_data : pandas.DataFrame
        Combined solid data
    interface_data : pandas.DataFrame
        Data with interface points identified
    timestep : float
        Specific timestep to visualize
    save_path : str, optional
        Path to save the visualization
    """
    # Filter data for the specified timestep
    fluid_t = fluid_data[fluid_data['time'] == timestep]
    solid_t = solid_data[solid_data['time'] == timestep]
    interface_t = interface_data[interface_data['time'] == timestep]
    
    if len(fluid_t) == 0 or len(solid_t) == 0:
        print(f"No data available for timestep {timestep}")
        return
    
    # Extract coordinates
    solid_coords_x = solid_t['x'].values
    solid_coords_y = solid_t['y'].values
    
    # Fit ellipse to solid points
    center_x, center_y, semi_major, semi_minor, angle = fit_ellipse(solid_coords_x, solid_coords_y)
    
    # Calculate aspect ratio to check circularity
    aspect_ratio = semi_major / semi_minor
    
    # Create the visualization
    plt.figure(figsize=(10, 10))
    
    # Plot all fluid points (blue)
    plt.scatter(fluid_t['x'], fluid_t['y'], s=1, color='blue', alpha=0.3, label='Fluid')
    
    # Plot solid points (red)
    plt.scatter(solid_t['x'], solid_t['y'], s=20, color='red', label='Solid')
    
    # Plot interface points (green)
    interface_points = interface_t[interface_t['is_interface']]
    plt.scatter(interface_points['x'], interface_points['y'], s=10, color='green', label='Interface')
    
    # Add velocity vectors for interface points
    plt.quiver(interface_points['x'], interface_points['y'],
               interface_points['u'], interface_points['v'],
               color='black', scale=50, width=0.002)
    
    # Draw the fitted ellipse/circle
    if 0.95 <= aspect_ratio <= 1.05:
        # Draw circle
        center_x, center_y, radius = fit_circle(solid_coords_x, solid_coords_y)
        circle = plt.Circle((center_x, center_y), radius, fill=False, color='purple', 
                          linestyle='--', linewidth=2, label='Fitted Circle')
        plt.gca().add_patch(circle)
        plt.title(f'Interface Points at t = {timestep:.2f} (Circular Fit, radius = {radius:.4f})')
    else:
        # Draw ellipse
        ellipse = Ellipse((center_x, center_y), 2*semi_major, 2*semi_minor, 
                           np.degrees(angle), fill=False, color='purple', 
                           linestyle='--', linewidth=2, label='Fitted Ellipse')
        plt.gca().add_patch(ellipse)
        plt.title(f'Interface Points at t = {timestep:.2f} (Elliptical Fit)')
    
    # Set axis limits with a small margin
    min_x, max_x = min(solid_coords_x) - 0.05, max(solid_coords_x) + 0.05
    min_y, max_y = min(solid_coords_y) - 0.05, max(solid_coords_y) + 0.05
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    
    # Add labels
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text with ellipse parameters
    plt.text(0.02, 0.98, f"Center: ({center_x:.4f}, {center_y:.4f})", 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    if 0.95 <= aspect_ratio <= 1.05:
        plt.text(0.02, 0.94, f"Radius: {radius:.4f}", 
                transform=plt.gca().transAxes, verticalalignment='top')
    else:
        plt.text(0.02, 0.94, f"Semi-major: {semi_major:.4f}, Semi-minor: {semi_minor:.4f}", 
                transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.02, 0.90, f"Angle: {np.degrees(angle):.2f}°", 
                transform=plt.gca().transAxes, verticalalignment='top')
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def analyze_ellipse_evolution(solid_data, timesteps=None, save_path=None):
    """
    Analyze the evolution of the ellipse/circle parameters over time
    
    Parameters:
    -----------
    solid_data : pandas.DataFrame
        Combined solid data
    timesteps : list, optional
        Specific timesteps to analyze
    save_path : str, optional
        Path to save the results
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with ellipse parameters for each timestep
    """
    # Get unique timesteps
    all_timesteps = solid_data['time'].unique()
    all_timesteps.sort()
    
    if timesteps is not None:
        timesteps = np.intersect1d(timesteps, all_timesteps)
    else:
        timesteps = all_timesteps
    
    # Initialize lists to store results
    times = []
    centers_x = []
    centers_y = []
    semi_majors = []
    semi_minors = []
    angles = []
    aspect_ratios = []
    radii = []
    
    # Process each timestep
    for t in timesteps:
        # Filter data for current timestep
        solid_t = solid_data[solid_data['time'] == t]
        
        if len(solid_t) < 3:  # Need at least 3 points to fit an ellipse
            print(f"Skipping timestep {t}: Insufficient data points")
            continue
        
        # Extract coordinates
        solid_coords_x = solid_t['x'].values
        solid_coords_y = solid_t['y'].values
        
        # Fit an ellipse to the solid points
        center_x, center_y, semi_major, semi_minor, angle = fit_ellipse(solid_coords_x, solid_coords_y)
        
        # Fit a circle as well
        cx, cy, radius = fit_circle(solid_coords_x, solid_coords_y)
        
        # Calculate aspect ratio
        aspect_ratio = semi_major / semi_minor
        
        # Store results
        times.append(t)
        centers_x.append(center_x)
        centers_y.append(center_y)
        semi_majors.append(semi_major)
        semi_minors.append(semi_minor)
        angles.append(np.degrees(angle))  # Convert to degrees
        aspect_ratios.append(aspect_ratio)
        radii.append(radius)
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'time': times,
        'center_x': centers_x,
        'center_y': centers_y,
        'semi_major': semi_majors,
        'semi_minor': semi_minors,
        'angle_deg': angles,
        'aspect_ratio': aspect_ratios,
        'radius': radii
    })
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot center position
    plt.subplot(2, 2, 1)
    plt.plot(results['time'], results['center_x'], 'b-', label='Center X')
    plt.plot(results['time'], results['center_y'], 'r-', label='Center Y')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Disc Center Position Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot semi-axes
    plt.subplot(2, 2, 2)
    plt.plot(results['time'], results['semi_major'], 'g-', label='Semi-major')
    plt.plot(results['time'], results['semi_minor'], 'm-', label='Semi-minor')
    plt.plot(results['time'], results['radius'], 'k--', label='Circle Radius')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Expected Radius (0.2)')
    plt.xlabel('Time')
    plt.ylabel('Length')
    plt.title('Disc Semi-axes Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot aspect ratio
    plt.subplot(2, 2, 3)
    plt.plot(results['time'], results['aspect_ratio'], 'b-')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Circle (1.0)')
    plt.xlabel('Time')
    plt.ylabel('Aspect Ratio')
    plt.title('Disc Aspect Ratio Over Time (Major/Minor)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot orientation
    plt.subplot(2, 2, 4)
    plt.plot(results['time'], results['angle_deg'], 'g-')
    plt.xlabel('Time')
    plt.ylabel('Angle (degrees)')
    plt.title('Disc Orientation Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    # Save results to CSV if a save path is provided
    if save_path:
        csv_path = save_path.replace('.png', '.csv')
        results.to_csv(csv_path, index=False)
        print(f"Saved ellipse parameters to {csv_path}")
    
    return results


def main():
    # Load the combined fluid and solid data
    fluid_file = 'data/processed_dataset/combined_fluid_data.csv'
    solid_file = 'data/processed_dataset/combined_solid_data.csv'
    
    if not os.path.exists(fluid_file) or not os.path.exists(solid_file):
        print(f"Error: Data files not found. Make sure {fluid_file} and {solid_file} exist.")
        return
    
    print(f"Loading fluid data from {fluid_file}...")
    fluid_data = pd.read_csv(fluid_file)
    
    print(f"Loading solid data from {solid_file}...")
    solid_data = pd.read_csv(solid_file)
    
    print(f"Loaded {len(fluid_data)} fluid points and {len(solid_data)} solid points.")
    
    # Create output directory for results
    os.makedirs('ellipse_results', exist_ok=True)
    
    # Analyze the evolution of the ellipse/circle over time
    print("\nAnalyzing disc shape evolution over time...")
    ellipse_results = analyze_ellipse_evolution(solid_data, save_path='ellipse_results/disc_evolution.png')
    
    # Identify interface points using ellipse/circle fitting
    print("\nIdentifying interface points using ellipse/circle fitting...")
    interface_data = identify_interface_points_ellipse(fluid_data, solid_data, radius=0.2, distance_threshold=0.02)
    
    if interface_data is not None:
        # Save the interface data
        interface_file = 'ellipse_interface_points.csv'
        interface_data.to_csv(interface_file, index=False)
        print(f"Saved interface points to {interface_file}")
        
        # Visualize selected timesteps
        timesteps_to_viz = ellipse_results['time'].tolist()[::10]  # Sample every 10th timestep
        if len(timesteps_to_viz) > 5:
            timesteps_to_viz = timesteps_to_viz[:5]  # Limit to 5 visualizations
        
        for t in timesteps_to_viz:
            if t in interface_data['time'].values:
                save_path = f"ellipse_results/interface_t{t:.2f}.png"
                visualize_interface_with_ellipse(fluid_data, solid_data, interface_data, t, save_path)
                print(f"Saved visualization for t={t:.2f}")


if __name__ == "__main__":
    main()