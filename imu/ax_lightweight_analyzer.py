#!/usr/bin/env python3
"""
Lightweight Accelerometer Real-time Analyzer
High-performance version optimized for smooth real-time visualization

Performance Optimizations:
- Reduced plot complexity
- Optimized update cycles  
- Minimal dependencies
- Efficient data structures
"""

import socket
import csv
import io
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading
import time
from datetime import datetime

class LightweightConfig:
    """Optimized configuration for performance"""
    UDP_PORT = 2055
    MAX_SAMPLES = 500        # Reduced for performance
    UPDATE_INTERVAL = 100    # Slower updates (10 FPS)
    PLOT_POINTS = 300        # Fewer points to plot
    
    COLORS = {
        'ax': '#FF4444',
        'ay': '#44FF44', 
        'az': '#4444FF',
        'bg': '#0A0A0A'
    }

class FastDataReceiver:
    """Optimized UDP receiver"""
    
    def __init__(self, port=2055):
        self.port = port
        self.running = False
        self.callback = None
        self.stats = {'packets': 0, 'errors': 0}
        
    def start(self, callback):
        """Start receiving with minimal overhead"""
        self.callback = callback
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(("", self.port))
            self.socket.settimeout(0.1)  # Fast timeout
            self.running = True
            
            # Start in separate thread
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"❌ Socket error: {e}")
            return False
    
    def _loop(self):
        """Minimal processing loop"""
        while self.running:
            try:
                data, _ = self.socket.recvfrom(1024)
                s = data.decode('utf-8', errors='ignore').strip().rstrip('#')
                
                # Quick CSV parse
                values = [float(x) for x in s.split(',')[:3]]
                if len(values) >= 3:
                    self.callback(values[0], values[1], values[2], time.time())
                    self.stats['packets'] += 1
                    
            except socket.timeout:
                continue
            except Exception:
                self.stats['errors'] += 1
    
    def stop(self):
        """Stop receiver"""
        self.running = False
        try:
            self.socket.close()
        except:
            pass

class FastVisualizer:
    """High-performance visualization"""
    
    def __init__(self):
        self.setup_plot()
        self.init_buffers()
        
    def setup_plot(self):
        """Minimal plot setup"""
        plt.style.use('dark_background')
        
        # Single figure, simple layout
        self.fig, ((self.ax_main, self.ax_3d), (self.ax_x, self.ax_y)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.patch.set_facecolor(LightweightConfig.COLORS['bg'])
        
        # 3D subplot
        self.ax_3d.remove()
        self.ax_3d = self.fig.add_subplot(2, 2, 2, projection='3d')
        
        # Style all axes quickly
        for ax in [self.ax_main, self.ax_x, self.ax_y]:
            ax.set_facecolor('#0F0F0F')
            ax.grid(True, alpha=0.3)
        
        self.ax_3d.set_facecolor('#0F0F0F')
        
        # Set titles once
        self.ax_main.set_title('🚀 Real-time ax, ay, az', color='#00FF88', fontweight='bold')
        self.ax_3d.set_title('3D Trajectory', color='#00D4FF')
        self.ax_x.set_title('X-axis Detail', color=LightweightConfig.COLORS['ax'])
        self.ax_y.set_title('Y-axis Detail', color=LightweightConfig.COLORS['ay'])
        
    def init_buffers(self):
        """Initialize data buffers"""
        maxlen = LightweightConfig.MAX_SAMPLES
        self.times = deque(maxlen=maxlen)
        self.ax_data = deque(maxlen=maxlen)
        self.ay_data = deque(maxlen=maxlen)
        self.az_data = deque(maxlen=maxlen)
        
        # Performance tracking
        self.last_update = 0
        self.frame_count = 0
        
    def add_data(self, ax, ay, az, timestamp):
        """Add data point efficiently"""
        self.times.append(timestamp)
        self.ax_data.append(ax)
        self.ay_data.append(ay)
        self.az_data.append(az)
        
    def update_plots(self):
        """Fast plot update"""
        if len(self.times) < 10:
            return
            
        # Limit update rate
        now = time.time()
        if now - self.last_update < LightweightConfig.UPDATE_INTERVAL / 1000:
            return
        self.last_update = now
        
        # Convert to arrays efficiently
        n = min(LightweightConfig.PLOT_POINTS, len(self.times))
        times = np.array(list(self.times)[-n:])
        ax_vals = np.array(list(self.ax_data)[-n:])
        ay_vals = np.array(list(self.ay_data)[-n:])
        az_vals = np.array(list(self.az_data)[-n:])
        
        # Normalize time
        times = times - times[0] if len(times) > 0 else times
        
        # Update main plot (fastest)
        self.ax_main.clear()
        self.ax_main.set_facecolor('#0F0F0F')
        self.ax_main.plot(times, ax_vals, color=LightweightConfig.COLORS['ax'], linewidth=1.5, label='ax')
        self.ax_main.plot(times, ay_vals, color=LightweightConfig.COLORS['ay'], linewidth=1.5, label='ay') 
        self.ax_main.plot(times, az_vals, color=LightweightConfig.COLORS['az'], linewidth=1.5, label='az')
        self.ax_main.legend(loc='upper right')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_title('🚀 Real-time ax, ay, az', color='#00FF88', fontweight='bold')
        
        # Update 3D (every few frames for performance)
        if self.frame_count % 3 == 0 and len(ax_vals) > 5:
            self.ax_3d.clear()
            recent = min(50, len(ax_vals))
            self.ax_3d.plot(ax_vals[-recent:], ay_vals[-recent:], az_vals[-recent:], 
                           color='#00D4FF', linewidth=2, alpha=0.7)
            if recent > 0:
                self.ax_3d.scatter([ax_vals[-1]], [ay_vals[-1]], [az_vals[-1]], 
                                 color='#FF4444', s=50)
            self.ax_3d.set_title('3D Trajectory', color='#00D4FF')
            
        # Update detail plots (every other frame)
        if self.frame_count % 2 == 0:
            # X-axis detail
            self.ax_x.clear()
            self.ax_x.set_facecolor('#0F0F0F')
            self.ax_x.plot(times, ax_vals, color=LightweightConfig.COLORS['ax'], linewidth=2)
            self.ax_x.fill_between(times, ax_vals, alpha=0.3, color=LightweightConfig.COLORS['ax'])
            self.ax_x.set_title('X-axis Detail', color=LightweightConfig.COLORS['ax'])
            self.ax_x.grid(True, alpha=0.3)
            
            # Y-axis detail  
            self.ax_y.clear()
            self.ax_y.set_facecolor('#0F0F0F')
            self.ax_y.plot(times, ay_vals, color=LightweightConfig.COLORS['ay'], linewidth=2)
            self.ax_y.fill_between(times, ay_vals, alpha=0.3, color=LightweightConfig.COLORS['ay'])
            self.ax_y.set_title('Y-axis Detail', color=LightweightConfig.COLORS['ay'])
            self.ax_y.grid(True, alpha=0.3)
        
        # Update title with minimal info
        if len(self.ax_data) > 0:
            current = f"ax={ax_vals[-1]:.2f}, ay={ay_vals[-1]:.2f}, az={az_vals[-1]:.2f}"
            sample_rate = len(self.times) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0
            self.fig.suptitle(f'⚡ FAST ACCELEROMETER ANALYZER | {current} | {sample_rate:.0f}Hz | {datetime.now().strftime("%H:%M:%S")}',
                            color='#00FF88', fontsize=14)
        
        self.frame_count += 1
        
        # Fast draw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

class LightweightAnalyzer:
    """Main lightweight analyzer"""
    
    def __init__(self):
        print("⚡ FAST ACCELEROMETER ANALYZER v1.0")
        print("=" * 50)
        print("🎯 Optimized for smooth real-time performance")
        print("📡 Listening on UDP port 2055")
        print("=" * 50)
        
        self.receiver = FastDataReceiver()
        self.visualizer = FastVisualizer()
        self.running = False
        self.start_time = None
        
    def start(self):
        """Start the analyzer"""
        # Start receiver
        if not self.receiver.start(self.on_data):
            return False
            
        self.running = True
        self.start_time = time.time()
        
        # Setup matplotlib
        plt.ion()
        plt.show(block=False)
        
        print("✅ Analyzer started - close plot window to stop")
        
        try:
            # Main loop
            while self.running and plt.get_fignums():
                self.visualizer.update_plots()
                plt.pause(0.01)  # Minimal pause
                
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            self.stop()
            
        return True
    
    def on_data(self, ax, ay, az, timestamp):
        """Handle new data point"""
        self.visualizer.add_data(ax, ay, az, timestamp)
        
        # Periodic stats (minimal overhead)
        if self.receiver.stats['packets'] % 1000 == 0:
            elapsed = time.time() - self.start_time if self.start_time else 1
            rate = self.receiver.stats['packets'] / elapsed
            print(f"📊 {self.receiver.stats['packets']} packets | {rate:.0f} Hz | {self.receiver.stats['errors']} errors")
    
    def stop(self):
        """Stop analyzer"""
        print("🛑 Stopping analyzer...")
        self.running = False
        self.receiver.stop()
        
        # Final stats
        if self.start_time:
            elapsed = time.time() - self.start_time
            total_packets = self.receiver.stats['packets']
            avg_rate = total_packets / elapsed if elapsed > 0 else 0
            print(f"📈 Final Stats: {total_packets} packets in {elapsed:.1f}s ({avg_rate:.0f} Hz avg)")
        
        plt.close('all')
        print("✅ Stopped")

def main():
    """Run the lightweight analyzer"""
    analyzer = LightweightAnalyzer()
    return 0 if analyzer.start() else 1

if __name__ == "__main__":
    exit(main())