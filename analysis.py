#!/usr/bin/env python3
"""
Analysis and plotting utilities for the pneumatic hopper simulation.

This script provides functions for analyzing and visualizing simulation results,
including comparison plots for different controllers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from datetime import datetime

# Import SimulationLogger for static load_data method
from core.logger import SimulationLogger


def plot_simulation_results(data, title=None, save_path=None):
    """
    Plot simulation results from logged data.
    
    Args:
        data (dict): Simulation data dictionary
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot to
    """
    # Extract data
    physics = data['physics']
    kalman = data['kalman']
    controller = data['controller']
    sensors = data['sensors']
    
    # Create figure
    fig = plt.figure(figsize=(12, 15))
    
    # Set title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        timestamp = datetime.fromtimestamp(data.get('timestamp', time.time()))
        fig.suptitle(f'Pneumatic Hopper Simulation Results - {timestamp}', fontsize=16)
    
    # Position plot
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(physics['time'], physics['position'], 'b-', label='True Position')
    ax1.plot(kalman['time'], kalman['position'], 'r--', label='Kalman Estimate')
    ax1.plot(controller['time'], controller['target'], 'g-', label='Target')
    
    # Plot sensor readings
    lidar_time = np.array(sensors['time'])
    lidar_readings = np.array(sensors['lidar_readings'])
    valid_lidar = ~np.isnan(lidar_readings)
    if np.any(valid_lidar):
        ax1.scatter(lidar_time[valid_lidar], lidar_readings[valid_lidar], 
                   marker='.', color='g', s=10, alpha=0.5, label='LiDAR')
    
    ax1.set_ylabel('Position (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Velocity plot
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(physics['time'], physics['velocity'], 'b-', label='True Velocity')
    ax2.plot(kalman['time'], kalman['velocity'], 'r--', label='Kalman Estimate')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True)
    ax2.legend()
    
    # Acceleration plot
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(physics['time'], physics['acceleration'], 'b-', label='True Acceleration')
    ax3.plot(kalman['time'], kalman['acceleration'], 'r--', label='Kalman Estimate')
    
    # Plot sensor readings
    mpu_time = np.array(sensors['time'])
    mpu_readings = np.array(sensors['mpu_readings'])
    valid_mpu = ~np.isnan(mpu_readings)
    if np.any(valid_mpu):
        ax3.scatter(mpu_time[valid_mpu], mpu_readings[valid_mpu], 
                   marker='.', color='m', s=10, alpha=0.5, label='MPU6050')
    
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.grid(True)
    ax3.legend()
    
    # Control output plot
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(controller['time'], controller['output'], 'k-', label='Control Output')
    ax4.plot(physics['time'], physics['thrust'], 'b-', label='Applied Thrust')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Control / Thrust')
    ax4.grid(True)
    ax4.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    # Show plot
    plt.show()


def plot_simulation_results_comparison(results_dict, title=None, save_path=None):
    """
    Plot comparison of simulation results from multiple controllers.
    
    Args:
        results_dict (dict): Dictionary where keys are controller names and values are data dictionaries
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot to
    """
    if not results_dict:
        print("No data to plot")
        return
    
    # Create figure
    fig = plt.figure(figsize=(14, 16))
    
    # Set title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Controller Performance Comparison', fontsize=16)
    
    # Define a color cycle for different controllers
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    line_styles = ['-', '--', '-.', ':']
    
    # Position plot
    ax1 = plt.subplot(4, 1, 1)
    ax1.set_title('Position Tracking')
    
    # Velocity plot
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.set_title('Velocity')
    
    # Error plot
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.set_title('Position Error')
    
    # Control output plot
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.set_title('Control Output')
    
    # Plot data for each controller
    for i, (controller_name, data) in enumerate(results_dict.items()):
        color_idx = i % len(colors)
        style_idx = (i // len(colors)) % len(line_styles)
        
        color = colors[color_idx]
        style = line_styles[style_idx]
        
        # Extract data
        physics = data['physics']
        kalman = data['kalman']
        controller = data['controller']
        
        # Position plot
        ax1.plot(physics['time'], physics['position'], f'{color}{style}', 
                 label=f'{controller_name} - Actual')
        ax1.plot(controller['time'], controller['target'], f'{color}:', 
                 label=f'{controller_name} - Target')
        
        # Velocity plot
        ax2.plot(physics['time'], physics['velocity'], f'{color}{style}', 
                 label=controller_name)
        
        # Calculate position error
        error = np.array(controller['error'])
        ax3.plot(controller['time'], error, f'{color}{style}', 
                 label=controller_name)
        
        # Control output
        ax4.plot(controller['time'], controller['output'], f'{color}{style}', 
                 label=f'{controller_name} - Command')
        ax4.plot(physics['time'], physics['thrust'], f'{color}:', 
                 label=f'{controller_name} - Applied', alpha=0.5)
    
    # Add grid and legends
    ax1.set_ylabel('Position (m)')
    ax1.grid(True)
    ax1.legend(loc='best')
    
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True)
    ax2.legend(loc='best')
    
    ax3.set_ylabel('Error (m)')
    ax3.grid(True)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.legend(loc='best')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Control / Thrust')
    ax4.grid(True)
    ax4.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Comparison plot saved to {save_path}")
    
    # Show plot
    plt.show()


def plot_kalman_performance(data, save_path=None):
    """
    Plot Kalman filter performance analysis.
    
    Args:
        data (dict): Simulation data dictionary
        save_path (str, optional): Path to save the plot to
    """
    # Extract data
    physics = data['physics']
    kalman = data['kalman']
    
    # Interpolate physics data to match Kalman timestamps
    # This is necessary because physics and Kalman data might be logged at different rates
    from scipy.interpolate import interp1d
    
    # Make sure we have data to interpolate
    if len(physics['time']) < 2 or len(kalman['time']) < 2:
        print("Not enough data points for Kalman performance analysis")
        return
    
    # Create interpolation functions
    interp_pos = interp1d(physics['time'], physics['position'], bounds_error=False, fill_value='extrapolate')
    interp_vel = interp1d(physics['time'], physics['velocity'], bounds_error=False, fill_value='extrapolate')
    interp_acc = interp1d(physics['time'], physics['acceleration'], bounds_error=False, fill_value='extrapolate')
    
    # Interpolate physics data at Kalman timestamps
    true_pos = interp_pos(kalman['time'])
    true_vel = interp_vel(kalman['time'])
    true_acc = interp_acc(kalman['time'])
    
    # Calculate errors
    pos_error = np.array(kalman['position']) - true_pos
    vel_error = np.array(kalman['velocity']) - true_vel
    acc_error = np.array(kalman['acceleration']) - true_acc
    
    # Extract covariance diagonal (variances)
    covariance = np.array(kalman['covariance'])
    
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle('Kalman Filter Performance Analysis', fontsize=16)
    
    # Position error plot
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(kalman['time'], pos_error, 'b-', label='Position Error')
    if covariance.shape[1] >= 1:
        std_pos = np.sqrt(covariance[:, 0])
        ax1.fill_between(kalman['time'], -2*std_pos, 2*std_pos, color='b', alpha=0.2, label='±2σ')
    ax1.set_ylabel('Position Error (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Velocity error plot
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(kalman['time'], vel_error, 'g-', label='Velocity Error')
    if covariance.shape[1] >= 2:
        std_vel = np.sqrt(covariance[:, 1])
        ax2.fill_between(kalman['time'], -2*std_vel, 2*std_vel, color='g', alpha=0.2, label='±2σ')
    ax2.set_ylabel('Velocity Error (m/s)')
    ax2.grid(True)
    ax2.legend()
    
    # Acceleration error plot
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(kalman['time'], acc_error, 'r-', label='Acceleration Error')
    if covariance.shape[1] >= 3:
        std_acc = np.sqrt(covariance[:, 2])
        ax3.fill_between(kalman['time'], -2*std_acc, 2*std_acc, color='r', alpha=0.2, label='±2σ')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration Error (m/s²)')
    ax3.grid(True)
    ax3.legend()
    
    # Print error statistics
    rms_pos = np.sqrt(np.mean(pos_error**2))
    rms_vel = np.sqrt(np.mean(vel_error**2))
    rms_acc = np.sqrt(np.mean(acc_error**2))
    
    print(f"Kalman Filter Performance:")
    print(f"  Position RMS Error: {rms_pos:.4f} m")
    print(f"  Velocity RMS Error: {rms_vel:.4f} m/s")
    print(f"  Acceleration RMS Error: {rms_acc:.4f} m/s²")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Kalman performance plot saved to {save_path}")
    
    # Show plot
    plt.show()


def plot_controller_performance(data, save_path=None):
    """
    Plot controller performance analysis.
    
    Args:
        data (dict): Simulation data dictionary
        save_path (str, optional): Path to save the plot to
    """
    # Extract data
    physics = data['physics']
    controller = data['controller']
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Controller Performance Analysis', fontsize=16)
    
    # Position and target plot
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(physics['time'], physics['position'], 'b-', label='Position')
    ax1.plot(controller['time'], controller['target'], 'r--', label='Target')
    ax1.set_ylabel('Position (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Error plot
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(controller['time'], controller['error'], 'g-', label='Error')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Error (m)')
    ax2.grid(True)
    ax2.legend()
    
    # Control output and thrust plot
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(controller['time'], controller['output'], 'k-', label='Control Output')
    ax3.plot(physics['time'], physics['thrust'], 'b-', label='Applied Thrust')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control / Thrust')
    ax3.grid(True)
    ax3.legend()
    
    # Print performance statistics
    settling_time = None
    overshoot = None
    steady_state_error = None
    
    # Calculate settling time (time to reach within 5% of target)
    if len(controller['time']) > 0 and len(controller['error']) > 0:
        # Get final target
        final_target = controller['target'][-1]
        
        # Calculate 5% threshold
        threshold = 0.05 * final_target
        
        # Find settling time
        for i in range(len(controller['time'])):
            if abs(controller['error'][i]) <= threshold:
                # Check if it stays within threshold
                if all(abs(e) <= threshold for e in controller['error'][i:]):
                    settling_time = controller['time'][i] - controller['time'][0]
                    break
        
        # Calculate overshoot
        max_pos = max(physics['position'])
        if max_pos > final_target:
            overshoot = (max_pos - final_target) / final_target * 100
        else:
            overshoot = 0.0
        
        # Calculate steady-state error (average of last 10% of data)
        n = len(controller['error'])
        if n > 10:
            steady_state_error = np.mean(np.abs(controller['error'][int(0.9*n):]))
    
    print(f"Controller Performance:")
    if settling_time is not None:
        print(f"  Settling Time (5%): {settling_time:.2f} s")
    if overshoot is not None:
        print(f"  Overshoot: {overshoot:.2f}%")
    if steady_state_error is not None:
        print(f"  Steady-State Error: {steady_state_error:.4f} m")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Controller performance plot saved to {save_path}")
    
    # Show plot
    plt.show()


def plot_controller_performance_comparison(results_dict, title=None, save_path=None):
    """
    Plot comparison of controller performance metrics for multiple controllers.
    
    Args:
        results_dict (dict): Dictionary where keys are controller names and values are data dictionaries
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot to
    """
    if not results_dict:
        print("No data to plot")
        return
    
    # Calculate performance metrics for each controller
    controller_metrics = {}
    
    for controller_name, data in results_dict.items():
        # Extract data
        physics = data['physics']
        controller = data['controller']
        
        # Calculate metrics
        metrics = {}
        
        # Calculate RMS tracking error
        error = np.array(controller['error'])
        metrics['rms_error'] = np.sqrt(np.mean(error**2))
        
        # Calculate control effort (integral of absolute control)
        control_output = np.array(controller['output'])
        time = np.array(controller['time'])
        dt = np.diff(time)
        metrics['control_effort'] = np.sum(np.abs(control_output[1:]) * dt)
        
        # Calculate settling time (time to reach within 5% of target)
        settling_time = None
        if len(controller['time']) > 0 and len(controller['error']) > 0:
            # Get final target
            final_target = controller['target'][-1]
            
            # Calculate 5% threshold
            threshold = 0.05 * final_target
            
            # Find settling time
            for i in range(len(controller['time'])):
                if abs(controller['error'][i]) <= threshold:
                    # Check if it stays within threshold
                    remain_within = True
                    for j in range(i, len(controller['error'])):
                        if abs(controller['error'][j]) > threshold:
                            remain_within = False
                            break
                    
                    if remain_within:
                        settling_time = controller['time'][i] - controller['time'][0]
                        break
        
        metrics['settling_time'] = settling_time if settling_time is not None else np.nan
        
        # Calculate overshoot
        max_pos = max(physics['position'])
        final_target = controller['target'][-1]
        if max_pos > final_target:
            metrics['overshoot'] = (max_pos - final_target) / final_target * 100
        else:
            metrics['overshoot'] = 0.0
        
        # Calculate steady-state error (average of last 10% of data)
        n = len(controller['error'])
        if n > 10:
            metrics['steady_state_error'] = np.mean(np.abs(controller['error'][int(0.9*n):]))
        else:
            metrics['steady_state_error'] = np.nan
        
        controller_metrics[controller_name] = metrics
    
    # Create bar chart comparison
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title if title else 'Controller Performance Metrics Comparison', fontsize=16)
    
    # Get controller names and metrics
    controllers = list(controller_metrics.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(controllers)))
    
    # RMS Error
    rms_errors = [controller_metrics[c]['rms_error'] for c in controllers]
    axs[0, 0].bar(controllers, rms_errors, color=colors)
    axs[0, 0].set_title('RMS Tracking Error')
    axs[0, 0].set_ylabel('Error (m)')
    axs[0, 0].grid(axis='y')
    
    # Settling Time
    settling_times = [controller_metrics[c]['settling_time'] for c in controllers]
    axs[0, 1].bar(controllers, settling_times, color=colors)
    axs[0, 1].set_title('Settling Time (5%)')
    axs[0, 1].set_ylabel('Time (s)')
    axs[0, 1].grid(axis='y')
    
    # Overshoot
    overshoots = [controller_metrics[c]['overshoot'] for c in controllers]
    axs[1, 0].bar(controllers, overshoots, color=colors)
    axs[1, 0].set_title('Overshoot')
    axs[1, 0].set_ylabel('Percentage (%)')
    axs[1, 0].grid(axis='y')
    
    # Control Effort
    control_efforts = [controller_metrics[c]['control_effort'] for c in controllers]
    axs[1, 1].bar(controllers, control_efforts, color=colors)
    axs[1, 1].set_title('Control Effort')
    axs[1, 1].set_ylabel('Integrated Control Signal')
    axs[1, 1].grid(axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Controller metrics comparison plot saved to {save_path}")
    
    # Show plot
    plt.show()
    
    # Print metrics in a table
    print("\nController Performance Metrics:")
    print(f"{'Controller':<15} {'RMS Error':<12} {'Settling Time':<15} {'Overshoot':<12} {'SS Error':<12}")
    print("-" * 70)
    
    for controller in controllers:
        metrics = controller_metrics[controller]
        settling_time_str = f"{metrics['settling_time']:.2f}" if not np.isnan(metrics['settling_time']) else "N/A"
        ss_error_str = f"{metrics['steady_state_error']:.4f}" if not np.isnan(metrics['steady_state_error']) else "N/A"
        
        print(f"{controller:<15} {metrics['rms_error']:.4f} m     {settling_time_str:<15} {metrics['overshoot']:.2f} %      {ss_error_str}")
    
    return controller_metrics


# If run as a standalone script
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pneumatic Hopper Simulation Analysis')
    parser.add_argument('file', help='Simulation data file to analyze')
    parser.add_argument('--output', '-o', help='Output folder for plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    parser.add_argument('--compare', action='store_true', help='Compare multiple controllers (provide directory of log files)')
    args = parser.parse_args()
    
    if args.compare:
        # Load all log files in the provided directory
        if os.path.isdir(args.file):
            log_files = [f for f in os.listdir(args.file) if f.endswith('.pkl')]
            if not log_files:
                print(f"No log files found in {args.file}")
                exit(1)
            
            # Load all data files
            results_dict = {}
            for log_file in log_files:
                file_path = os.path.join(args.file, log_file)
                try:
                    # Try to extract controller name from filename
                    # Expected format: simulation_YYYYMMDD_HHMMSS_CONTROLLERNAME.pkl
                    parts = log_file.split('_')
                    if len(parts) >= 4:
                        controller_name = '_'.join(parts[3:]).replace('.pkl', '')
                    else:
                        controller_name = log_file.replace('.pkl', '')
                    
                    # Load data and add to results dictionary
                    data = SimulationLogger.load_data(file_path)
                    results_dict[controller_name] = data
                    print(f"Loaded data for controller: {controller_name}")
                except Exception as e:
                    print(f"Error loading {log_file}: {e}")
            
            if not results_dict:
                print("No valid data files could be loaded")
                exit(1)
            
            # Create output folder if needed
            if args.output and not os.path.exists(args.output):
                os.makedirs(args.output)
            
            # Plot comparison results
            if args.output:
                save_path = os.path.join(args.output, "controller_comparison.png")
                metrics_path = os.path.join(args.output, "controller_metrics.png")
            else:
                save_path = None
                metrics_path = None
            
            if not args.no_show or save_path:
                plot_simulation_results_comparison(results_dict, save_path=save_path)
                plot_controller_performance_comparison(results_dict, save_path=metrics_path)
        else:
            print(f"Error: {args.file} is not a directory")
            exit(1)
    else:
        # Single file analysis (original behavior)
        # Load data
        data = SimulationLogger.load_data(args.file)
        
        # Create output folder if needed
        if args.output and not os.path.exists(args.output):
            os.makedirs(args.output)
        
        # Generate plots
        if args.output:
            base_name = os.path.splitext(os.path.basename(args.file))[0]
            save_path = os.path.join(args.output, f"{base_name}_results.png")
            kalman_path = os.path.join(args.output, f"{base_name}_kalman.png")
            controller_path = os.path.join(args.output, f"{base_name}_controller.png")
        else:
            save_path = None
            kalman_path = None
            controller_path = None
        
        # Plot results
        if not args.no_show or save_path:
            plot_simulation_results(data, save_path=save_path)
        
        # Plot Kalman performance
        if not args.no_show or kalman_path:
            plot_kalman_performance(data, save_path=kalman_path)
        
        # Plot controller performance
        if not args.no_show or controller_path:
            plot_controller_performance(data, save_path=controller_path)
