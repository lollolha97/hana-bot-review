import socket, csv, io, re
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import threading
import time
from datetime import datetime
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

PORT = 2055
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", PORT))
print(f"listening udp://0.0.0.0:{PORT}")

header = None
data_buffer = {}
time_buffer = deque(maxlen=1000)
max_points = 1000
sample_rate = 0
last_update = time.time()
packet_count = 0
connection_status = "DISCONNECTED"
data_quality = "UNKNOWN"

# 실시간 그래프 설정 - 더 멋있게!
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor('#0a0a0a')
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.25)

# 메인 센서 데이터 플롯 (더 크게)
ax_main = fig.add_subplot(gs[0:2, 0:3])  # 메인 센서 데이터
ax_3d = fig.add_subplot(gs[0:2, 3], projection='3d')  # 3D 궤적

# 하단 분석 패널
ax_fft = fig.add_subplot(gs[2, 0])      # FFT 분석
ax_stats = fig.add_subplot(gs[2, 1])    # 통계
ax_phase = fig.add_subplot(gs[2, 2])    # 위상 플롯
ax_status = fig.add_subplot(gs[2, 3])   # 상태 표시

# 최하단 - 개별 센서
ax_x = fig.add_subplot(gs[3, 0])        # X축
ax_y = fig.add_subplot(gs[3, 1])        # Y축  
ax_z = fig.add_subplot(gs[3, 2])        # Z축
ax_magnitude = fig.add_subplot(gs[3, 3]) # 벡터 크기

# 전체 스타일 설정
for ax in [ax_main, ax_fft, ax_stats, ax_phase, ax_status, ax_x, ax_y, ax_z, ax_magnitude]:
    ax.set_facecolor('#0f0f0f')
    ax.grid(True, alpha=0.2, color='#333333')
    ax.tick_params(colors='#cccccc', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#444444')
        spine.set_linewidth(0.8)

ax_3d.set_facecolor('#0f0f0f')

fig.suptitle('🚀 PROFESSIONAL IMU DATA ANALYZER 🚀', 
             fontsize=20, fontweight='bold', color='#00ff88', y=0.98)

def calculate_stats():
    if not data_buffer:
        return {}
    
    stats = {}
    for key, values in data_buffer.items():
        if len(values) > 10:
            arr = np.array(list(values))
            stats[key] = {
                'mean': np.mean(arr),
                'std': np.std(arr),
                'rms': np.sqrt(np.mean(arr**2)),
                'peak': np.max(np.abs(arr))
            }
    return stats

def update_plot():
    global last_update, sample_rate, packet_count, connection_status, data_quality
    
    if not data_buffer or len(time_buffer) < 2:
        return
    
    current_time = time.time()
    
    # 샘플링 레이트 계산
    if current_time - last_update > 1.0:
        sample_rate = packet_count / (current_time - last_update)
        packet_count = 0
        last_update = current_time
        connection_status = "CONNECTED" if sample_rate > 0 else "DISCONNECTED"
        
        # 데이터 품질 평가
        if sample_rate > 50:
            data_quality = "EXCELLENT"
        elif sample_rate > 20:
            data_quality = "GOOD"  
        elif sample_rate > 5:
            data_quality = "FAIR"
        else:
            data_quality = "POOR"
    
    # 시간 축 생성
    time_axis = np.array(list(time_buffer))
    if len(time_axis) > 0:
        time_axis = time_axis - time_axis[0]  # 상대 시간
    
    # 색상 팔레트 - 네온 스타일
    neon_colors = ['#ff073a', '#39ff14', '#00d4ff', '#ff6600', '#c77cff', '#ffff00']
    
    # 🔥 메인 센서 데이터 플롯 - 모든 축을 오버레이
    ax_main.clear()
    ax_main.set_facecolor('#0f0f0f')
    
    if len(data_buffer) >= 3:
        keys = list(data_buffer.keys())[:3]
        for i, key in enumerate(keys):
            if len(data_buffer[key]) > 0:
                data = list(data_buffer[key])
                min_len = min(len(time_axis), len(data))
                if min_len > 1:
                    ax_main.plot(time_axis[-min_len:], data[-min_len:], 
                               color=neon_colors[i], linewidth=2.5, label=key, alpha=0.9,
                               linestyle='-', marker='.', markersize=1, markevery=5)
        
        ax_main.set_title('🎯 LIVE SENSOR DATA STREAM', fontsize=16, fontweight='bold', color='#00ff88')
        ax_main.set_xlabel('Time (seconds)', color='#cccccc', fontsize=12)
        ax_main.set_ylabel('Sensor Value', color='#cccccc', fontsize=12)
        ax_main.legend(loc='upper left', fancybox=True, shadow=True, framealpha=0.8)
        ax_main.grid(True, alpha=0.3, linestyle='--', color='#333333')
    
    # 🌀 3D 궤적 플롯
    ax_3d.clear()
    if len(data_buffer) >= 3:
        keys = list(data_buffer.keys())[:3]
        if all(len(data_buffer[key]) > 20 for key in keys):
            x_data = list(data_buffer[keys[0]])[-50:]
            y_data = list(data_buffer[keys[1]])[-50:]
            z_data = list(data_buffer[keys[2]])[-50:]
            
            # 궤적을 그라디언트로 표시
            for i in range(len(x_data)-1):
                alpha = i / len(x_data)
                ax_3d.plot([x_data[i], x_data[i+1]], 
                          [y_data[i], y_data[i+1]], 
                          [z_data[i], z_data[i+1]], 
                          color=neon_colors[i%3], alpha=alpha, linewidth=2)
            
            # 현재 위치 강조
            ax_3d.scatter([x_data[-1]], [y_data[-1]], [z_data[-1]], 
                         color='#ff073a', s=100, marker='o', edgecolors='white')
            
        ax_3d.set_title('🌌 3D TRAJECTORY', fontsize=12, fontweight='bold', color='#00d4ff')
        ax_3d.set_xlabel('X', color='#ff073a')
        ax_3d.set_ylabel('Y', color='#39ff14') 
        ax_3d.set_zlabel('Z', color='#00d4ff')
    
    # 📊 FFT 스펙트럼 분석
    ax_fft.clear()
    ax_fft.set_facecolor('#0f0f0f')
    
    if data_buffer and sample_rate > 1:
        first_key = list(data_buffer.keys())[0]
        data = list(data_buffer[first_key])
        if len(data) > 64:
            fft_size = min(512, len(data))
            fft_size = 2 ** int(np.log2(fft_size))
            
            if fft_size >= 64:
                data_slice = data[-fft_size:]
                fft = np.abs(np.fft.fft(data_slice))[:fft_size//2]
                freqs = np.fft.fftfreq(fft_size, 1/sample_rate)[:fft_size//2]
                
                if len(freqs) > 1 and len(fft) > 1 and len(freqs) == len(fft):
                    ax_fft.semilogy(freqs[1:], fft[1:], color='#39ff14', linewidth=2, alpha=0.8)
                    ax_fft.fill_between(freqs[1:], fft[1:], alpha=0.3, color='#39ff14')
    
    ax_fft.set_title('⚡ FREQUENCY SPECTRUM', fontsize=12, fontweight='bold', color='#39ff14')
    ax_fft.set_xlabel('Frequency (Hz)', color='#cccccc')
    ax_fft.set_ylabel('Amplitude', color='#cccccc')
    
    # 📈 통계 레이더 차트
    ax_stats.clear()
    ax_stats.set_facecolor('#0f0f0f')
    
    stats = calculate_stats()
    if stats and len(stats) >= 3:
        keys = list(stats.keys())[:3]
        rms_values = [stats[key]['rms'] for key in keys]
        std_values = [stats[key]['std'] for key in keys]
        
        x = np.arange(len(keys))
        width = 0.35
        
        bars1 = ax_stats.bar(x - width/2, rms_values, width, label='RMS', 
                            color='#ff073a', alpha=0.8, edgecolor='white', linewidth=1)
        bars2 = ax_stats.bar(x + width/2, std_values, width, label='STD',
                            color='#00d4ff', alpha=0.8, edgecolor='white', linewidth=1)
        
        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels([k.split('_')[-1] for k in keys], color='#cccccc')
        ax_stats.legend(fancybox=True, shadow=True)
        
        # 값 표시
        for bar in bars1:
            height = bar.get_height()
            ax_stats.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                         f'{height:.2f}', ha='center', va='bottom', color='#cccccc', fontsize=8)
    
    ax_stats.set_title('📊 STATISTICAL ANALYSIS', fontsize=12, fontweight='bold', color='#ff073a')
    
    # 🌊 위상 플롯 (Lissajous curve)
    ax_phase.clear()
    ax_phase.set_facecolor('#0f0f0f')
    
    if len(data_buffer) >= 2:
        keys = list(data_buffer.keys())[:2]
        if all(len(data_buffer[key]) > 50 for key in keys):
            x_data = list(data_buffer[keys[0]])[-100:]
            y_data = list(data_buffer[keys[1]])[-100:]
            
            # 그라디언트 효과
            for i in range(len(x_data)-1):
                alpha = 0.3 + 0.7 * i / len(x_data)
                ax_phase.plot([x_data[i], x_data[i+1]], [y_data[i], y_data[i+1]], 
                             color='#c77cff', alpha=alpha, linewidth=2)
            
            ax_phase.scatter(x_data[-1], y_data[-1], color='#ffff00', s=80, marker='*', 
                           edgecolors='white', linewidth=2, zorder=10)
    
    ax_phase.set_title('🌊 PHASE PORTRAIT', fontsize=12, fontweight='bold', color='#c77cff')
    ax_phase.set_xlabel('X Channel', color='#cccccc')
    ax_phase.set_ylabel('Y Channel', color='#cccccc')
    
    # 🚦 상태 표시 패널
    ax_status.clear()
    ax_status.set_facecolor('#0f0f0f')
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.axis('off')
    
    # 연결 상태
    status_color = '#39ff14' if connection_status == "CONNECTED" else '#ff073a'
    ax_status.text(0.5, 0.8, f'CONNECTION', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='#cccccc')
    ax_status.text(0.5, 0.65, connection_status, ha='center', va='center',
                   fontsize=16, fontweight='bold', color=status_color)
    
    # 데이터 품질
    quality_colors = {'EXCELLENT': '#39ff14', 'GOOD': '#ffff00', 'FAIR': '#ff6600', 'POOR': '#ff073a'}
    quality_color = quality_colors.get(data_quality, '#cccccc')
    ax_status.text(0.5, 0.4, f'DATA QUALITY', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='#cccccc')
    ax_status.text(0.5, 0.25, data_quality, ha='center', va='center',
                   fontsize=16, fontweight='bold', color=quality_color)
    
    # 샘플링 레이트
    ax_status.text(0.5, 0.05, f'{sample_rate:.1f} Hz', ha='center', va='center',
                   fontsize=20, fontweight='bold', color='#00d4ff')
    
    # 개별 축 플롯
    axes_info = [(ax_x, 'X-AXIS', '#ff073a'), (ax_y, 'Y-AXIS', '#39ff14'), 
                 (ax_z, 'Z-AXIS', '#00d4ff'), (ax_magnitude, 'MAGNITUDE', '#ffff00')]
    
    for idx, (ax, title, color) in enumerate(axes_info):
        ax.clear()
        ax.set_facecolor('#0f0f0f')
        
        if idx < len(data_buffer):
            key = list(data_buffer.keys())[idx]
            data = list(data_buffer[key])
            
            if len(data) > 10:
                # 최근 데이터만 표시
                recent_data = data[-200:]
                recent_time = time_axis[-len(recent_data):]
                
                ax.plot(recent_time, recent_data, color=color, linewidth=2, alpha=0.9)
                ax.fill_between(recent_time, recent_data, alpha=0.2, color=color)
                
                # 현재 값 표시
                current_val = recent_data[-1]
                ax.text(0.02, 0.98, f'{current_val:.3f}', transform=ax.transAxes,
                       fontsize=12, fontweight='bold', color=color, va='top')
        
        elif idx == 3 and len(data_buffer) >= 3:  # Magnitude 계산
            keys = list(data_buffer.keys())[:3]
            if all(len(data_buffer[k]) > 10 for k in keys):
                x_vals = np.array(list(data_buffer[keys[0]])[-200:])
                y_vals = np.array(list(data_buffer[keys[1]])[-200:])
                z_vals = np.array(list(data_buffer[keys[2]])[-200:])
                magnitude = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)
                
                recent_time = time_axis[-len(magnitude):]
                ax.plot(recent_time, magnitude, color=color, linewidth=2, alpha=0.9)
                ax.fill_between(recent_time, magnitude, alpha=0.2, color=color)
                
                current_mag = magnitude[-1]
                ax.text(0.02, 0.98, f'{current_mag:.3f}', transform=ax.transAxes,
                       fontsize=12, fontweight='bold', color=color, va='top')
        
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.tick_params(colors='#cccccc', labelsize=8)
    
    # 메인 타이틀 업데이트
    status_text = f'🔥 LIVE at {sample_rate:.1f} Hz | 📦 {len(time_buffer)} packets | ⏰ {datetime.now().strftime("%H:%M:%S")} | 🎯 {connection_status}'
    fig.suptitle(f'🚀 PROFESSIONAL IMU DATA ANALYZER - {status_text}', 
                fontsize=16, fontweight='bold', color='#00ff88', y=0.98)
    
    plt.draw()
    plt.pause(0.01)

def data_receiver():
    global header, data_buffer, packet_count
    
    while True:
        try:
            data, addr = sock.recvfrom(65535)
            s = data.decode("utf-8", errors="ignore").strip()
            
            if s.endswith("#"):
                s = s[:-1]

            # 첫 번째 데이터 패킷을 받으면 자동으로 헤더 생성
            if header is None:
                row = next(csv.reader(io.StringIO(s)))
                vals = [float(x) if re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", x.strip(), re.I) else x for x in row]
                
                # 숫자 데이터인 경우 자동으로 채널명 생성
                if all(isinstance(v, float) for v in vals):
                    if len(vals) == 3:
                        header = ['Accel_X', 'Accel_Y', 'Accel_Z']  # 3축 가속도계로 가정
                    elif len(vals) == 6:
                        header = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
                    elif len(vals) == 9:
                        header = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']
                    else:
                        header = [f'Channel_{i+1}' for i in range(len(vals))]
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Auto-generated header: {header}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Data source: {addr[0]}:{addr[1]}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Detected {len(vals)} data channels")
                    
                    # 데이터 버퍼 초기화
                    data_buffer = {col: deque(maxlen=max_points) for col in header}

            if header:
                row = next(csv.reader(io.StringIO(s)))
                vals = [float(x) if re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", x.strip(), re.I) else x for x in row]

                if len(header) == len(vals):
                    current_time = time.time()
                    time_buffer.append(current_time)
                    packet_count += 1
                    
                    # 숫자 데이터만 버퍼에 추가
                    for key, value in zip(header, vals):
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            data_buffer[key].append(value)
                    
                    # 주기적으로 상태 출력 (10초마다)
                    if packet_count % (int(sample_rate * 10) if sample_rate > 0 else 100) == 0:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Packets received: {packet_count}, Sample rate: {sample_rate:.1f} Hz")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            if "Bad file descriptor" not in str(e):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Data processing error: {e}")
            continue

# 시작 메시지
print(f"[{datetime.now().strftime('%H:%M:%S')}] IMU Data Analyzer started")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for data on UDP port {PORT}...")

# 데이터 수신을 별도 스레드에서 실행
receiver_thread = threading.Thread(target=data_receiver, daemon=True)
receiver_thread.start()

# 인터랙티브 모드 활성화
plt.ion()

# 메인 스레드에서 그래프 업데이트
try:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting real-time visualization...")
    
    # 그래프 창이 열릴 때까지 대기
    plt.show(block=False)
    
    while True:
        if plt.get_fignums():  # 그래프 창이 열려있는지 확인
            update_plot()
            plt.pause(0.1)  # 10 FPS 업데이트
        else:
            break
        
except KeyboardInterrupt:
    pass
finally:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Shutdown initiated...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Total packets processed: {len(time_buffer)}")
    try:
        sock.close()
    except:
        pass
    plt.close('all')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] IMU Data Analyzer stopped.")
