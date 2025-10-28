import subprocess
import paramiko  # pip install paramiko
import time
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

# 配置：替换为你的设备IP和类型
DEVICES = {
    "192.168.1.101": "Pi2B",  # 示例IP，替换为真实
    "192.168.1.102": "Pi2B",
    # ... 添加所有13个IP
    "192.168.1.113": "Pi4B",
}
USERNAME = "pi"  # 默认用户名
SSH_KEY = "/home/pi/.ssh/id_rsa"  # SSH私钥路径
N_DEVICES = len(DEVICES)
IP_LIST = list(DEVICES.keys())

# 辅助函数：SSH执行命令
def ssh_exec(ip: str, cmd: str, timeout: int = 30) -> str:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=USERNAME, key_filename=SSH_KEY)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    output = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    client.close()
    if error:
        raise RuntimeError(f"SSH error on {ip}: {error}")
    return output

# 步骤1: 测量计算C（每个设备运行PyTorch基准）
def measure_compute(ip: str) -> float:
    try:
        # 上传并运行compute_bench.py（假设已复制到所有设备，或用scp）
        cmd = "python3 compute_bench.py"
        output = ssh_exec(ip, cmd)
        c_value = float(output.split()[-1])  # 假设输出最后是C值
        print(f"Compute C for {ip} ({DEVICES[ip]}): {c_value}")
        return c_value
    except Exception as e:
        print(f"Error measuring compute for {ip}: {e}")
        return 3.0  # 默认fallback

# 步骤2: 测量延迟（ping矩阵）
def measure_latency(ip_from: str, ip_to: str) -> float:
    cmd = f"ping -c 10 -W 1 {ip_to} | tail -1 | awk '{{print $4}}' | cut -d'/' -f2"
    output = ssh_exec(ip_from, cmd)
    return float(output) / 1000.0  # ms to s

# 步骤3: 测量带宽（iperf3矩阵）
def measure_bandwidth(ip_server: str, ip_client: str) -> float:
    # 先确保server运行：ssh_exec(ip_server, "iperf3 -s -D &")  # 后台启动，假设预启动
    cmd = f"iperf3 -c {ip_server} -t 5 -P 1 | tail -1 | awk '{{print $7}}'"  # bytes/s
    output = ssh_exec(ip_client, cmd)
    bw = float(output)
    # 停止server: ssh_exec(ip_server, "pkill iperf3")
    return bw
"""
if __name__ == "__main__":
    print("开始Profile...")
    
    # 测量C
    C_values = {}
    with ThreadPoolExecutor(max_workers=5) as executor:  # 并行测量C
        futures = {executor.submit(measure_compute, ip): ip for ip in IP_LIST}
        for future in as_completed(futures):
            ip = futures[future]
            C_values[ip] = future.result()
    
    # 预启动所有iperf3 server
    for ip in IP_LIST:
        ssh_exec(ip, "iperf3 -s -D &")
    time.sleep(5)  # 等待启动
    
    # 测量网络矩阵
    latency_matrix = [[0.0] * N_DEVICES for _ in range(N_DEVICES)]
    bandwidth_matrix = [[0.0] * N_DEVICES for _ in range(N_DEVICES)]
    
    for i, ip_from in enumerate(IP_LIST):
        for j, ip_to in enumerate(IP_LIST):
            if i == j:
                latency_matrix[i][j] = 0.0
                bandwidth_matrix[i][j] = float('inf')  # 自环无限带宽
                continue
            # 并行ping延迟
            lat_future = executor.submit(measure_latency, ip_from, ip_to)
            # 串行带宽（避免冲突）
            bw = measure_bandwidth(ip_to, ip_from)  # 双向取平均，可加反向
            latency_matrix[i][j] = lat_future.result()
            bandwidth_matrix[i][j] = bw
    
    # 停止所有server
    for ip in IP_LIST:
        ssh_exec(ip, "pkill iperf3")
    
    # 保存结果
    with open("C_values.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["IP", "Type", "C"])
        for ip, typ in DEVICES.items():
            writer.writerow([ip, typ, C_values[ip]])
    
    with open("latency_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["From\\To"] + IP_LIST)
        for i in range(N_DEVICES):
            row = [IP_LIST[i]] + [latency_matrix[i][j] for j in range(N_DEVICES)]
            writer.writerow(row)
    
    with open("bandwidth_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["From\\To"] + IP_LIST)
        for i in range(N_DEVICES):
            row = [IP_LIST[i]] + [bandwidth_matrix[i][j] for j in range(N_DEVICES)]
            writer.writerow(row)
    
    # JSON摘要
    summary = {"C_values": C_values, "latency_avg": sum(sum(row) for row in latency_matrix)/ (N_DEVICES*(N_DEVICES-1)), 
               "bandwidth_avg": sum(sum(row) for row in bandwidth_matrix)/ (N_DEVICES*(N_DEVICES-1))}
    with open("profile_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Profile完成！检查CSV文件和summary.json")
"""