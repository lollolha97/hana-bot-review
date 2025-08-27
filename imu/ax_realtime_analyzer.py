#!/usr/bin/env python3
"""
Accelerometer Real-time Data Analysis & Visualization System
Enhanced data science toolkit for ax, ay, az accelerometer analysis

Author: Data Science Team
Version: 2.0.0
License: MIT
"""

import socket
import csv
import io
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from collections import deque, defaultdict
import threading
import time
from datetime import datetime, timedelta
import warnings
import json

# Optional imports with fallbacks
try:
    from scipy import signal, stats
    from scipy.fft import fft, fftfreq
    HAS_SCIPY = True
except ImportError:
    # Fallback to numpy FFT
    from numpy.fft import fft, fftfreq
    HAS_SCIPY = False
    print("⚠️  Warning: scipy not available, using numpy fallbacks for advanced signal processing")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = False  # Not critical for core functionality
except ImportError:
    HAS_SKLEARN = False

try:
    import pandas as pd
    HAS_PANDAS = False  # Not critical for core functionality
except ImportError:
    HAS_PANDAS = False

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)

class AccelerometerConfig:
    """Configuration class for accelerometer analysis parameters"""
    
    # Network Configuration
    UDP_PORT = 2055
    BUFFER_SIZE = 65535
    
    # Data Processing
    MAX_SAMPLES = 2000          # Increased buffer for better analysis
    SAMPLING_RATE = 100         # Expected Hz, auto-detected
    WINDOW_SIZE = 512           # FFT window size
    OVERLAP_RATIO = 0.5         # Window overlap for smoother analysis
    
    # Signal Processing
    FILTER_ORDER = 4            # Butterworth filter order
    LOW_CUTOFF = 0.5           # Low-pass filter cutoff (Hz)
    HIGH_CUTOFF = 50           # High-pass filter cutoff (Hz)
    OUTLIER_THRESHOLD = 3.0    # Standard deviations for outlier detection
    
    # Quality Metrics
    MIN_SAMPLE_RATE = 10       # Minimum acceptable sampling rate
    MAX_NOISE_RATIO = 0.3      # Maximum noise-to-signal ratio
    MIN_SNR_DB = 10           # Minimum signal-to-noise ratio (dB)
    
    # Visualization
    UPDATE_INTERVAL = 50       # Milliseconds between updates
    PLOT_HISTORY = 1000       # Number of samples to display
    COLOR_SCHEME = {
        'ax': '#FF4444',       # Red for X-axis
        'ay': '#44FF44',       # Green for Y-axis  
        'az': '#4444FF',       # Blue for Z-axis
        'magnitude': '#FFFF44', # Yellow for magnitude
        'background': '#0A0A0A',
        'grid': '#333333',
        'text': '#CCCCCC'
    }

class DataReceiver:
    """Enhanced UDP data receiver with error handling and validation"""
    
    def __init__(self, port=AccelerometerConfig.UDP_PORT):
        self.port = port
        self.socket = None
        self.running = False
        self.data_callback = None
        self.error_callback = None
        self.stats = {
            'packets_received': 0,
            'packets_dropped': 0,
            'parse_errors': 0,
            'last_packet_time': 0
        }
        
    def setup_socket(self):
        """Initialize UDP socket with error handling"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(("", self.port))
            self.socket.settimeout(1.0)  # Non-blocking with timeout
            return True
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Socket setup failed: {e}")
            return False
    
    def start_receiving(self, data_callback, error_callback=None):
        """Start receiving data in separate thread"""
        self.data_callback = data_callback
        self.error_callback = error_callback
        
        if not self.setup_socket():
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        return True
    
    def stop_receiving(self):
        """Stop receiving data and cleanup"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
    def _receive_loop(self):
        """Main data receiving loop with robust error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running:
            try:
                # Receive data with timeout
                data, addr = self.socket.recvfrom(AccelerometerConfig.BUFFER_SIZE)
                
                # Reset error counter on successful receive
                consecutive_errors = 0
                
                # Process received data
                self._process_packet(data, addr)
                
            except socket.timeout:
                # Timeout is expected, continue
                continue
            except Exception as e:
                consecutive_errors += 1
                self.stats['packets_dropped'] += 1
                
                if self.error_callback:
                    self.error_callback(f"Receive error: {e}")
                
                # Stop if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    if self.error_callback:
                        self.error_callback("Too many consecutive errors, stopping receiver")
                    break
                
                # Brief pause before retry
                time.sleep(0.1)
    
    def _process_packet(self, data, addr):
        """Process received packet and extract accelerometer data"""
        try:
            # Decode packet
            packet_str = data.decode('utf-8', errors='ignore').strip()
            if packet_str.endswith('#'):
                packet_str = packet_str[:-1]
            
            # Parse CSV data
            row = next(csv.reader(io.StringIO(packet_str)))
            
            # Convert to float values
            values = []
            for val in row:
                if re.match(r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$', val.strip()):
                    values.append(float(val))
                else:
                    # Skip non-numeric values
                    continue
            
            # Validate data format (expect at least 3 values for ax, ay, az)
            if len(values) >= 3:
                timestamp = time.time()
                accel_data = {
                    'timestamp': timestamp,
                    'ax': values[0],
                    'ay': values[1],
                    'az': values[2],
                    'source': addr,
                    'packet_size': len(data)
                }
                
                # Add additional axes if available
                if len(values) > 3:
                    accel_data['extra'] = values[3:]
                
                self.stats['packets_received'] += 1
                self.stats['last_packet_time'] = timestamp
                
                if self.data_callback:
                    self.data_callback(accel_data)
            else:
                self.stats['parse_errors'] += 1
                if self.error_callback:
                    self.error_callback(f"Invalid data format: expected ≥3 values, got {len(values)}")
                    
        except Exception as e:
            self.stats['parse_errors'] += 1
            if self.error_callback:
                self.error_callback(f"Packet processing error: {e}")

class SignalProcessor:
    """Advanced signal processing and analysis tools"""
    
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        # self.scaler = StandardScaler()  # Remove sklearn dependency
        
        # Initialize filters
        self.setup_filters()
        
        # Analysis windows (with fallbacks)
        if HAS_SCIPY:
            self.windows = {
                'hanning': signal.windows.hann,
                'hamming': signal.windows.hamming,
                'blackman': signal.windows.blackman
            }
        else:
            # Use numpy fallbacks
            self.windows = {
                'hanning': np.hanning,
                'hamming': np.hamming,
                'blackman': np.blackman
            }
        
    def setup_filters(self):
        """Setup digital filters for noise reduction"""
        if not HAS_SCIPY:
            print("🔧 Filters disabled: scipy not available")
            self.lowpass_sos = None
            self.highpass_sos = None
            return
            
        nyquist = self.sampling_rate / 2
        
        try:
            # Low-pass filter for smoothing
            self.lowpass_sos = signal.butter(
                AccelerometerConfig.FILTER_ORDER,
                AccelerometerConfig.HIGH_CUTOFF / nyquist,
                btype='low', output='sos'
            )
            
            # High-pass filter for removing DC bias
            self.highpass_sos = signal.butter(
                AccelerometerConfig.FILTER_ORDER,
                AccelerometerConfig.LOW_CUTOFF / nyquist,
                btype='high', output='sos'
            )
        except Exception as e:
            print(f"⚠️ Filter setup failed: {e}")
            self.lowpass_sos = None
            self.highpass_sos = None
    
    def apply_filter(self, data, filter_type='bandpass'):
        """Apply digital filter to data"""
        if len(data) < 10 or not HAS_SCIPY or self.lowpass_sos is None:
            return data
            
        try:
            if filter_type == 'lowpass':
                return signal.sosfilt(self.lowpass_sos, data)
            elif filter_type == 'highpass':
                return signal.sosfilt(self.highpass_sos, data)
            elif filter_type == 'bandpass':
                # Apply both filters
                filtered = signal.sosfilt(self.highpass_sos, data)
                return signal.sosfilt(self.lowpass_sos, filtered)
            else:
                return data
        except Exception:
            return data  # Return original data if filtering fails
    
    def remove_outliers(self, data, threshold=None):
        """Remove outliers using Z-score method"""
        if threshold is None:
            threshold = AccelerometerConfig.OUTLIER_THRESHOLD
            
        if len(data) < 10:
            return data
        
        try:
            if HAS_SCIPY:
                z_scores = np.abs(stats.zscore(data))
                mask = z_scores < threshold
                return np.array(data)[mask]
            else:
                # Simple outlier removal without scipy
                mean_val = np.mean(data)
                std_val = np.std(data)
                mask = np.abs(data - mean_val) < threshold * std_val
                return np.array(data)[mask]
        except Exception:
            return data
    
    def compute_fft(self, data, window='hanning'):
        """Compute FFT with windowing"""
        if len(data) < AccelerometerConfig.WINDOW_SIZE:
            return None, None
        
        try:
            # Apply window
            if window in self.windows:
                windowed = data * self.windows[window](len(data))
            else:
                windowed = data
            
            # Compute FFT
            fft_vals = fft(windowed)
            freqs = fftfreq(len(windowed), 1/self.sampling_rate)
            
            # Return positive frequencies only
            n = len(freqs) // 2
            return freqs[:n], np.abs(fft_vals[:n])
        except Exception:
            return None, None
    
    def compute_psd(self, data):
        """Compute Power Spectral Density"""
        if len(data) < AccelerometerConfig.WINDOW_SIZE:
            return None, None
        
        if not HAS_SCIPY:
            return None, None  # PSD requires scipy
        
        try:
            freqs, psd = signal.welch(
                data, 
                fs=self.sampling_rate,
                window='hanning',
                nperseg=min(AccelerometerConfig.WINDOW_SIZE, len(data)//4),
                noverlap=None
            )
            return freqs, psd
        except Exception:
            return None, None
    
    def detect_motion_state(self, ax_data, ay_data, az_data):
        """Classify motion state based on accelerometer patterns"""
        if len(ax_data) < 50:  # Need sufficient data
            return "insufficient_data"
        
        try:
            # Compute statistical features
            magnitude = np.sqrt(np.array(ax_data)**2 + np.array(ay_data)**2 + np.array(az_data)**2)
            
            # Statistical measures
            mag_std = np.std(magnitude)
            mag_mean = np.mean(magnitude)
            
            # Classify based on patterns
            if mag_std < 0.5:  # Low variability
                if 9.0 < mag_mean < 10.5:  # Near 1g
                    return "static"
                else:
                    return "tilted_static"
            elif mag_std < 2.0:  # Moderate variability
                return "gentle_motion"
            elif mag_std < 5.0:  # High variability
                return "active_motion"
            else:  # Very high variability
                return "vigorous_motion"
                
        except Exception:
            return "unknown"
    
    def compute_signal_quality(self, data):
        """Compute signal quality metrics"""
        if len(data) < 100:
            return {'snr_db': 0, 'quality_score': 0}
        
        try:
            # Remove trend (simple detrending without scipy)
            if HAS_SCIPY:
                detrended = signal.detrend(data)
            else:
                # Simple linear detrending
                x = np.arange(len(data))
                p = np.polyfit(x, data, 1)
                detrended = data - np.polyval(p, x)
            
            # Compute SNR (simplified)
            signal_power = np.var(detrended)
            noise_estimate = np.var(np.diff(detrended))  # High-freq content as noise proxy
            
            if noise_estimate > 0:
                snr = signal_power / noise_estimate
                snr_db = 10 * np.log10(snr) if snr > 0 else -np.inf
            else:
                snr_db = np.inf
            
            # Quality score (0-100)
            quality_score = min(100, max(0, (snr_db + 20) * 2))  # Map -10dB to 100
            
            return {
                'snr_db': snr_db,
                'quality_score': quality_score,
                'signal_power': signal_power,
                'noise_power': noise_estimate
            }
            
        except Exception:
            return {'snr_db': 0, 'quality_score': 0}

class StatisticalAnalyzer:
    """Real-time statistical analysis and monitoring"""
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.reset_stats()
    
    def reset_stats(self):
        """Reset all statistical accumulators"""
        self.stats = {
            'ax': defaultdict(list),
            'ay': defaultdict(list), 
            'az': defaultdict(list),
            'magnitude': defaultdict(list)
        }
        self.correlation_buffer = deque(maxlen=self.window_size)
    
    def update_statistics(self, ax, ay, az):
        """Update running statistics with new data point"""
        magnitude = np.sqrt(ax**2 + ay**2 + az**2)
        
        # Store data point for correlation analysis
        self.correlation_buffer.append([ax, ay, az, magnitude])
        
        # Update per-axis statistics
        for axis, value in [('ax', ax), ('ay', ay), ('az', az), ('magnitude', magnitude)]:
            self.stats[axis]['values'].append(value)
            if len(self.stats[axis]['values']) > self.window_size:
                self.stats[axis]['values'].pop(0)
    
    def get_current_statistics(self):
        """Compute current statistical measures"""
        results = {}
        
        for axis in ['ax', 'ay', 'az', 'magnitude']:
            values = self.stats[axis]['values']
            if len(values) < 10:
                continue
                
            arr = np.array(values)
            
            # Basic statistics with fallbacks
            basic_stats = {
                'mean': np.mean(arr),
                'std': np.std(arr),
                'rms': np.sqrt(np.mean(arr**2)),
                'min': np.min(arr),
                'max': np.max(arr),
                'peak_to_peak': np.ptp(arr),
                'percentile_95': np.percentile(arr, 95),
                'percentile_05': np.percentile(arr, 5)
            }
            
            # Advanced statistics if scipy available
            if HAS_SCIPY and len(arr) > 3:
                basic_stats['skewness'] = stats.skew(arr)
                basic_stats['kurtosis'] = stats.kurtosis(arr)
            else:
                basic_stats['skewness'] = 0
                basic_stats['kurtosis'] = 0
                
            results[axis] = basic_stats
        
        return results
    
    def get_correlation_matrix(self):
        """Compute correlation matrix between axes"""
        if len(self.correlation_buffer) < 50:
            return None
            
        try:
            data = np.array(list(self.correlation_buffer))
            corr_matrix = np.corrcoef(data.T)
            return corr_matrix
        except Exception:
            return None
    
    def detect_anomalies(self, current_values):
        """Simple anomaly detection based on statistical thresholds"""
        anomalies = {}
        current_stats = self.get_current_statistics()
        
        for axis, value in current_values.items():
            if axis in current_stats:
                stats_data = current_stats[axis]
                threshold = stats_data['mean'] + 3 * stats_data['std']
                
                if abs(value) > threshold:
                    anomalies[axis] = {
                        'value': value,
                        'threshold': threshold,
                        'severity': abs(value) / threshold if threshold > 0 else 0
                    }
        
        return anomalies

class VisualizationDashboard:
    """Professional matplotlib-based visualization dashboard"""
    
    def __init__(self, config=None):
        self.config = config or AccelerometerConfig()
        self.setup_matplotlib()
        self.setup_layout()
        self.init_data_buffers()
        
    def setup_matplotlib(self):
        """Configure matplotlib for professional appearance"""
        plt.style.use('dark_background')
        plt.rcParams.update({
            'font.size': 10,
            'axes.linewidth': 0.8,
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'figure.facecolor': self.config.COLOR_SCHEME['background'],
            'axes.facecolor': '#0F0F0F',
            'text.color': self.config.COLOR_SCHEME['text'],
            'axes.edgecolor': '#444444'
        })
        
    def setup_layout(self):
        """Create professional dashboard layout"""
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.patch.set_facecolor(self.config.COLOR_SCHEME['background'])
        
        # Create grid layout (4 rows, 4 columns)
        gs = self.fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # Primary time series plot (large, top-left)
        self.ax_timeseries = self.fig.add_subplot(gs[0:2, 0:3])
        
        # 3D trajectory plot (top-right)
        self.ax_3d = self.fig.add_subplot(gs[0:2, 3], projection='3d')
        
        # Frequency analysis (middle row)
        self.ax_fft = self.fig.add_subplot(gs[2, 0])
        self.ax_psd = self.fig.add_subplot(gs[2, 1])
        self.ax_correlation = self.fig.add_subplot(gs[2, 2])
        self.ax_statistics = self.fig.add_subplot(gs[2, 3])
        
        # Individual axis plots (bottom row)
        self.ax_x = self.fig.add_subplot(gs[3, 0])
        self.ax_y = self.fig.add_subplot(gs[3, 1])
        self.ax_z = self.fig.add_subplot(gs[3, 2])
        self.ax_magnitude = self.fig.add_subplot(gs[3, 3])
        
        # Style all axes
        self.style_axes()
        
    def style_axes(self):
        """Apply consistent styling to all axes"""
        all_axes = [self.ax_timeseries, self.ax_fft, self.ax_psd, self.ax_correlation,
                   self.ax_statistics, self.ax_x, self.ax_y, self.ax_z, self.ax_magnitude]
        
        for ax in all_axes:
            ax.set_facecolor('#0F0F0F')
            ax.grid(True, alpha=0.3, color=self.config.COLOR_SCHEME['grid'])
            ax.tick_params(colors=self.config.COLOR_SCHEME['text'], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color('#444444')
                spine.set_linewidth(0.8)
        
        # 3D plot styling
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False
        
    def init_data_buffers(self):
        """Initialize data buffers for visualization"""
        self.time_buffer = deque(maxlen=self.config.PLOT_HISTORY)
        self.ax_buffer = deque(maxlen=self.config.PLOT_HISTORY)
        self.ay_buffer = deque(maxlen=self.config.PLOT_HISTORY)
        self.az_buffer = deque(maxlen=self.config.PLOT_HISTORY)
        self.magnitude_buffer = deque(maxlen=self.config.PLOT_HISTORY)
        
        # Performance tracking
        self.last_update = time.time()
        self.frame_times = deque(maxlen=100)
        
    def update_data(self, timestamp, ax, ay, az):
        """Update data buffers with new measurements"""
        self.time_buffer.append(timestamp)
        self.ax_buffer.append(ax)
        self.ay_buffer.append(ay)
        self.az_buffer.append(az)
        self.magnitude_buffer.append(np.sqrt(ax**2 + ay**2 + az**2))
    
    def update_plots(self, statistics=None, signal_quality=None, motion_state="unknown"):
        """Update all visualization panels"""
        start_time = time.time()
        
        if len(self.time_buffer) < 2:
            return
            
        # Convert buffers to arrays for plotting
        times = np.array(list(self.time_buffer))
        ax_data = np.array(list(self.ax_buffer))
        ay_data = np.array(list(self.ay_buffer))
        az_data = np.array(list(self.az_buffer))
        mag_data = np.array(list(self.magnitude_buffer))
        
        # Normalize time to relative seconds
        times = times - times[0]
        
        # Update primary time series
        self.update_timeseries(times, ax_data, ay_data, az_data)
        
        # Update 3D trajectory
        self.update_3d_trajectory(ax_data, ay_data, az_data)
        
        # Update frequency analysis
        self.update_frequency_plots(ax_data, ay_data, az_data)
        
        # Update statistics panel
        self.update_statistics_panel(statistics, signal_quality, motion_state)
        
        # Update individual axis plots
        self.update_individual_axes(times, ax_data, ay_data, az_data, mag_data)
        
        # Update main title with status
        self.update_main_title(len(self.time_buffer), signal_quality)
        
        # Track performance
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        
        plt.draw()
        plt.pause(0.001)  # Allow GUI updates
    
    def update_timeseries(self, times, ax_data, ay_data, az_data):
        """Update main time series plot"""
        self.ax_timeseries.clear()
        self.ax_timeseries.set_facecolor('#0F0F0F')
        
        # Plot all axes with distinct colors
        self.ax_timeseries.plot(times, ax_data, color=self.config.COLOR_SCHEME['ax'], 
                               linewidth=2, label='ax (X)', alpha=0.9, marker='.', markersize=1, markevery=10)
        self.ax_timeseries.plot(times, ay_data, color=self.config.COLOR_SCHEME['ay'], 
                               linewidth=2, label='ay (Y)', alpha=0.9, marker='.', markersize=1, markevery=10)
        self.ax_timeseries.plot(times, az_data, color=self.config.COLOR_SCHEME['az'], 
                               linewidth=2, label='az (Z)', alpha=0.9, marker='.', markersize=1, markevery=10)
        
        # Formatting
        self.ax_timeseries.set_title('🚀 Real-time Accelerometer Data (ax, ay, az)', 
                                   fontsize=16, fontweight='bold', color='#00FF88')
        self.ax_timeseries.set_xlabel('Time (seconds)', color=self.config.COLOR_SCHEME['text'])
        self.ax_timeseries.set_ylabel('Acceleration (m/s²)', color=self.config.COLOR_SCHEME['text'])
        self.ax_timeseries.legend(loc='upper right', fancybox=True, shadow=True, framealpha=0.8)
        self.ax_timeseries.grid(True, alpha=0.3, linestyle='--')
        
        # Add current values as text
        if len(ax_data) > 0:
            current_text = f'Current: ax={ax_data[-1]:.3f}, ay={ay_data[-1]:.3f}, az={az_data[-1]:.3f}'
            self.ax_timeseries.text(0.02, 0.98, current_text, transform=self.ax_timeseries.transAxes,
                                  fontsize=10, verticalalignment='top', 
                                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    def update_3d_trajectory(self, ax_data, ay_data, az_data, points=100):
        """Update 3D acceleration trajectory"""
        self.ax_3d.clear()
        
        if len(ax_data) < 10:
            return
            
        # Use recent data for trajectory
        recent_ax = ax_data[-points:]
        recent_ay = ay_data[-points:]
        recent_az = az_data[-points:]
        
        # Create color gradient for trajectory
        colors = plt.cm.viridis(np.linspace(0, 1, len(recent_ax)))
        
        # Plot trajectory with gradient
        for i in range(len(recent_ax)-1):
            self.ax_3d.plot([recent_ax[i], recent_ax[i+1]], 
                           [recent_ay[i], recent_ay[i+1]], 
                           [recent_az[i], recent_az[i+1]], 
                           color=colors[i], linewidth=2, alpha=0.7)
        
        # Highlight current position
        if len(recent_ax) > 0:
            self.ax_3d.scatter([recent_ax[-1]], [recent_ay[-1]], [recent_az[-1]], 
                             color='#FF4444', s=100, marker='o', edgecolors='white', linewidth=2)
        
        # Formatting
        self.ax_3d.set_title('🌌 3D Acceleration Vector', fontsize=14, fontweight='bold', color='#00D4FF')
        self.ax_3d.set_xlabel('ax (m/s²)', color=self.config.COLOR_SCHEME['ax'])
        self.ax_3d.set_ylabel('ay (m/s²)', color=self.config.COLOR_SCHEME['ay'])
        self.ax_3d.set_zlabel('az (m/s²)', color=self.config.COLOR_SCHEME['az'])
        
        # Equal aspect ratio for accurate representation
        max_range = max([np.ptp(recent_ax), np.ptp(recent_ay), np.ptp(recent_az)]) / 2
        mid_ax = np.mean(recent_ax)
        mid_ay = np.mean(recent_ay)
        mid_az = np.mean(recent_az)
        
        self.ax_3d.set_xlim3d(mid_ax - max_range, mid_ax + max_range)
        self.ax_3d.set_ylim3d(mid_ay - max_range, mid_ay + max_range)
        self.ax_3d.set_zlim3d(mid_az - max_range, mid_az + max_range)
    
    def update_frequency_plots(self, ax_data, ay_data, az_data):
        """Update FFT and PSD analysis plots"""
        # FFT Analysis
        self.ax_fft.clear()
        self.ax_fft.set_facecolor('#0F0F0F')
        
        if len(ax_data) >= 128:  # Need sufficient data for FFT
            # Compute FFT for magnitude
            magnitude = np.sqrt(ax_data**2 + ay_data**2 + az_data**2)
            fft_size = min(512, len(magnitude))
            fft_size = 2 ** int(np.log2(fft_size))  # Power of 2
            
            if fft_size >= 64:
                mag_slice = magnitude[-fft_size:]
                fft_vals = np.abs(np.fft.fft(mag_slice))[:fft_size//2]
                freqs = np.fft.fftfreq(fft_size, 1/100)[:fft_size//2]  # Assume 100Hz
                
                self.ax_fft.semilogy(freqs[1:], fft_vals[1:], color='#39FF14', linewidth=2, alpha=0.8)
                self.ax_fft.fill_between(freqs[1:], fft_vals[1:], alpha=0.3, color='#39FF14')
                
                # Mark dominant frequency
                if len(fft_vals) > 1:
                    peak_idx = np.argmax(fft_vals[1:]) + 1
                    peak_freq = freqs[peak_idx]
                    self.ax_fft.axvline(peak_freq, color='#FF4444', linestyle='--', 
                                       label=f'Peak: {peak_freq:.1f}Hz')
                    self.ax_fft.legend()
        
        self.ax_fft.set_title('⚡ FFT Spectrum', fontsize=12, fontweight='bold', color='#39FF14')
        self.ax_fft.set_xlabel('Frequency (Hz)')
        self.ax_fft.set_ylabel('Amplitude')
        
        # PSD Analysis
        self.ax_psd.clear()
        self.ax_psd.set_facecolor('#0F0F0F')
        
        if len(ax_data) >= 128:
            try:
                from scipy import signal as scipy_signal
                magnitude = np.sqrt(ax_data**2 + ay_data**2 + az_data**2)
                freqs_psd, psd = scipy_signal.welch(magnitude, fs=100, nperseg=min(256, len(magnitude)//4))
                
                self.ax_psd.semilogy(freqs_psd, psd, color='#00D4FF', linewidth=2)
                self.ax_psd.fill_between(freqs_psd, psd, alpha=0.3, color='#00D4FF')
            except:
                pass  # Skip if scipy not available
        
        self.ax_psd.set_title('📊 Power Spectral Density', fontsize=12, fontweight='bold', color='#00D4FF')
        self.ax_psd.set_xlabel('Frequency (Hz)')
        self.ax_psd.set_ylabel('PSD')
    
    def update_statistics_panel(self, statistics, signal_quality, motion_state):
        """Update statistics and quality metrics"""
        self.ax_statistics.clear()
        self.ax_statistics.set_facecolor('#0F0F0F')
        self.ax_statistics.axis('off')
        
        # Display key metrics
        y_pos = 0.9
        line_height = 0.15
        
        # Motion state
        motion_colors = {
            'static': '#39FF14',
            'gentle_motion': '#FFFF00',
            'active_motion': '#FF6600',
            'vigorous_motion': '#FF4444',
            'unknown': '#CCCCCC'
        }
        motion_color = motion_colors.get(motion_state, '#CCCCCC')
        
        self.ax_statistics.text(0.05, y_pos, f'Motion State:', fontweight='bold', 
                              transform=self.ax_statistics.transAxes, color=self.config.COLOR_SCHEME['text'])
        self.ax_statistics.text(0.05, y_pos-0.05, motion_state.replace('_', ' ').title(), 
                              transform=self.ax_statistics.transAxes, color=motion_color, fontsize=12)
        y_pos -= line_height
        
        # Signal quality
        if signal_quality:
            quality_color = '#39FF14' if signal_quality.get('quality_score', 0) > 70 else '#FF4444'
            self.ax_statistics.text(0.05, y_pos, f'Signal Quality:', fontweight='bold',
                                  transform=self.ax_statistics.transAxes, color=self.config.COLOR_SCHEME['text'])
            self.ax_statistics.text(0.05, y_pos-0.05, f"{signal_quality.get('quality_score', 0):.1f}%",
                                  transform=self.ax_statistics.transAxes, color=quality_color, fontsize=12)
            y_pos -= line_height
            
            # SNR
            snr_db = signal_quality.get('snr_db', 0)
            snr_color = '#39FF14' if snr_db > 10 else '#FF4444'
            self.ax_statistics.text(0.05, y_pos, f'SNR: {snr_db:.1f} dB',
                                  transform=self.ax_statistics.transAxes, color=snr_color)
            y_pos -= line_height
        
        # Statistics summary
        if statistics:
            for axis in ['ax', 'ay', 'az']:
                if axis in statistics:
                    stats_data = statistics[axis]
                    rms = stats_data.get('rms', 0)
                    std = stats_data.get('std', 0)
                    
                    axis_color = self.config.COLOR_SCHEME[axis]
                    self.ax_statistics.text(0.05, y_pos, f'{axis.upper()}: RMS={rms:.2f}, σ={std:.2f}',
                                          transform=self.ax_statistics.transAxes, color=axis_color, fontsize=10)
                    y_pos -= 0.1
        
        self.ax_statistics.set_title('📊 Analysis Metrics', fontsize=12, fontweight='bold', color='#C77CFF')
    
    def update_individual_axes(self, times, ax_data, ay_data, az_data, mag_data):
        """Update individual axis plots"""
        axes_data = [
            (self.ax_x, ax_data, 'ax (X-axis)', self.config.COLOR_SCHEME['ax']),
            (self.ax_y, ay_data, 'ay (Y-axis)', self.config.COLOR_SCHEME['ay']),
            (self.ax_z, az_data, 'az (Z-axis)', self.config.COLOR_SCHEME['az']),
            (self.ax_magnitude, mag_data, 'Magnitude', self.config.COLOR_SCHEME['magnitude'])
        ]
        
        for ax, data, title, color in axes_data:
            ax.clear()
            ax.set_facecolor('#0F0F0F')
            
            if len(data) > 0:
                # Plot recent data
                recent_times = times[-200:] if len(times) > 200 else times
                recent_data = data[-200:] if len(data) > 200 else data
                
                ax.plot(recent_times, recent_data, color=color, linewidth=2, alpha=0.9)
                ax.fill_between(recent_times, recent_data, alpha=0.2, color=color)
                
                # Current value annotation
                current_val = data[-1]
                ax.text(0.02, 0.98, f'{current_val:.3f}', transform=ax.transAxes,
                       fontsize=12, fontweight='bold', color=color, verticalalignment='top')
            
            ax.set_title(title, fontsize=11, fontweight='bold', color=color)
            ax.tick_params(colors=self.config.COLOR_SCHEME['text'], labelsize=8)
            ax.grid(True, alpha=0.3)
    
    def update_main_title(self, sample_count, signal_quality):
        """Update main dashboard title with current status"""
        quality_score = signal_quality.get('quality_score', 0) if signal_quality else 0
        avg_frame_time = np.mean(self.frame_times) if self.frame_times else 0
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        status_text = (f'📊 Samples: {sample_count} | '
                      f'🎯 Quality: {quality_score:.1f}% | '
                      f'⚡ FPS: {fps:.1f} | '
                      f'🕐 {datetime.now().strftime("%H:%M:%S")}')
        
        self.fig.suptitle(f'🚀 ACCELEROMETER DATA SCIENCE ANALYZER - {status_text}',
                         fontsize=16, fontweight='bold', color='#00FF88', y=0.98)

class AccelerometerAnalyzer:
    """Main application class orchestrating all components"""
    
    def __init__(self):
        self.config = AccelerometerConfig()
        self.running = False
        
        # Initialize components
        self.receiver = DataReceiver(self.config.UDP_PORT)
        self.signal_processor = SignalProcessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.dashboard = VisualizationDashboard(self.config)
        
        # Data storage
        self.raw_data_log = []
        self.processed_data_log = []
        
        # Performance monitoring
        self.start_time = None
        self.packet_count = 0
        self.error_count = 0
        
    def start_analysis(self):
        """Start real-time analysis system"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting Accelerometer Data Science Analyzer")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 Listening on UDP port {self.config.UDP_PORT}")
        
        # Setup data receiver
        success = self.receiver.start_receiving(
            data_callback=self.process_data_packet,
            error_callback=self.handle_error
        )
        
        if not success:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Failed to start UDP receiver")
            return False
        
        self.running = True
        self.start_time = time.time()
        
        # Start visualization
        plt.ion()
        plt.show(block=False)
        
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Analysis system started successfully")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 💡 Close the plot window or press Ctrl+C to stop")
            
            # Main update loop
            while self.running and plt.get_fignums():
                self.update_dashboard()
                time.sleep(self.config.UPDATE_INTERVAL / 1000.0)
                
        except KeyboardInterrupt:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🛑 Received shutdown signal")
        finally:
            self.stop_analysis()
        
        return True
    
    def process_data_packet(self, data):
        """Process incoming accelerometer data packet"""
        try:
            # Extract accelerometer values
            ax, ay, az = data['ax'], data['ay'], data['az']
            timestamp = data['timestamp']
            
            # Update statistical analyzer
            self.statistical_analyzer.update_statistics(ax, ay, az)
            
            # Update visualization data
            self.dashboard.update_data(timestamp, ax, ay, az)
            
            # Log raw data (keep last 10000 samples)
            self.raw_data_log.append({
                'timestamp': timestamp,
                'ax': ax, 'ay': ay, 'az': az
            })
            if len(self.raw_data_log) > 10000:
                self.raw_data_log.pop(0)
            
            self.packet_count += 1
            
            # Periodic status update
            if self.packet_count % 1000 == 0:
                elapsed = time.time() - self.start_time
                rate = self.packet_count / elapsed if elapsed > 0 else 0
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Processed {self.packet_count} packets at {rate:.1f} Hz")
                
        except Exception as e:
            self.handle_error(f"Data processing error: {e}")
    
    def update_dashboard(self):
        """Update visualization dashboard with current data"""
        if len(self.dashboard.time_buffer) < 2:
            return
        
        try:
            # Get current statistics
            current_stats = self.statistical_analyzer.get_current_statistics()
            
            # Compute signal quality
            if len(self.dashboard.ax_buffer) > 100:
                magnitude_data = list(self.dashboard.magnitude_buffer)
                signal_quality = self.signal_processor.compute_signal_quality(magnitude_data)
            else:
                signal_quality = None
            
            # Detect motion state
            if len(self.dashboard.ax_buffer) > 50:
                ax_data = list(self.dashboard.ax_buffer)
                ay_data = list(self.dashboard.ay_buffer)
                az_data = list(self.dashboard.az_buffer)
                motion_state = self.signal_processor.detect_motion_state(ax_data, ay_data, az_data)
            else:
                motion_state = "initializing"
            
            # Update dashboard
            self.dashboard.update_plots(current_stats, signal_quality, motion_state)
            
        except Exception as e:
            self.handle_error(f"Dashboard update error: {e}")
    
    def handle_error(self, error_msg):
        """Handle system errors"""
        self.error_count += 1
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] ⚠️ Error #{self.error_count}: {error_msg}")
        
        # Stop system if too many errors
        if self.error_count > 100:
            print(f"[{timestamp}] 🚨 Too many errors, stopping system")
            self.running = False
    
    def stop_analysis(self):
        """Stop analysis system and cleanup"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🛑 Stopping analysis system...")
        
        self.running = False
        
        # Stop data receiver
        self.receiver.stop_receiving()
        
        # Generate final report
        self.generate_session_report()
        
        # Cleanup
        plt.close('all')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Analysis system stopped")
    
    def generate_session_report(self):
        """Generate analysis session report"""
        if self.start_time is None:
            return
        
        elapsed_time = time.time() - self.start_time
        avg_rate = self.packet_count / elapsed_time if elapsed_time > 0 else 0
        
        print("\n" + "="*60)
        print("📊 ACCELEROMETER ANALYSIS SESSION REPORT")
        print("="*60)
        print(f"⏱️  Session Duration: {elapsed_time:.1f} seconds")
        print(f"📦 Total Packets: {self.packet_count}")
        print(f"⚡ Average Sample Rate: {avg_rate:.1f} Hz")
        print(f"❌ Error Count: {self.error_count}")
        print(f"💾 Data Samples Logged: {len(self.raw_data_log)}")
        
        # Final statistics
        if len(self.raw_data_log) > 10:
            final_stats = self.statistical_analyzer.get_current_statistics()
            if 'magnitude' in final_stats:
                mag_stats = final_stats['magnitude']
                print(f"📏 Final Magnitude - Mean: {mag_stats['mean']:.3f}, RMS: {mag_stats['rms']:.3f}")
        
        print("="*60)
        print(f"🎯 Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
    
    def export_data(self, filename=None):
        """Export collected data to file"""
        if not self.raw_data_log:
            print("No data to export")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'accelerometer_data_{timestamp}.json'
        
        try:
            export_data = {
                'session_info': {
                    'start_time': self.start_time,
                    'duration': time.time() - self.start_time if self.start_time else 0,
                    'packet_count': self.packet_count,
                    'error_count': self.error_count
                },
                'configuration': {
                    'sampling_rate': self.config.SAMPLING_RATE,
                    'buffer_size': self.config.MAX_SAMPLES,
                    'filter_settings': {
                        'low_cutoff': self.config.LOW_CUTOFF,
                        'high_cutoff': self.config.HIGH_CUTOFF
                    }
                },
                'raw_data': self.raw_data_log[-1000:]  # Export last 1000 samples
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Data exported to {filename}")
            return True
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Export failed: {e}")
            return False

def main():
    """Main entry point"""
    print("🚀 ACCELEROMETER DATA SCIENCE ANALYZER v2.0.0")
    print("=" * 60)
    print("Real-time accelerometer data analysis and visualization")
    print("Optimized for ax, ay, az data streams")
    print("=" * 60)
    
    try:
        analyzer = AccelerometerAnalyzer()
        analyzer.start_analysis()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())