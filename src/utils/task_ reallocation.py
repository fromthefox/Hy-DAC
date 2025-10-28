"""
Task粒度的组合重新分配算法.
在每个Task完成后,根据上一个Task的实际执行时间,對設備在組之間進行邏輯的調整.
接收參數:
- prev_allocation: 上一個Task的分配方案,格式為Dict,示例: {"Group1": {"Device":[1,2,3], "Leader":[2], "Layers":(3,7)}, "Group2":...}
- actual_times: 上一個Task的實際執行時間,格式為Dict,示例: {"Group1": 12.5, "Group2": 15.0, ...}
- C_values: 設備的計算能力,格式為Dict,示例: {"Device1": 5.0, "Device2": 4.0, ...}
"""

from typing import Dict, List, Tuple
import math

def compute_group_capacity(devices: List[int], C_values: Dict[int, float]) -> float:
    """計算組的總計算能力"""
    return sum(C_values[d] for d in devices)

def estimate_layer_count(exec_time: float, group_capacity: float) -> float:
    """根據執行時間反推目前層數分配"""
    return exec_time * group_capacity

def task_reallocation(
    prev_allocation: Dict[str, Dict],
    actual_times: Dict[str, Dict],
    C_values: Dict[int, float],
    total_layers: int,
    threshold: float = 0.15,  # 15% 偏差閾值
    strategy: str = "greedy"
) -> Dict[str, Dict]:
    """
    Task粒度設備重新分配
    
    Args:
        prev_allocation: 上一個Task的分配方案
        actual_times: 各Group的實際執行時間 {"Group_i": {"exec_time": x, "idle_time": y}}
        C_values: 全局設備計算能力
        total_layers: 當前Task的總層數（通常與模型層數相同）
        threshold: 觸發重新分配的時間偏差閾值
        strategy: "greedy" | "optimal_dp" | "flow_based"
    
    Returns:
        new_allocation: 新的分配方案
    """
    
    # Step 1: 計算各Group的性能指標
    group_metrics = {}
    max_time = 0
    min_time = float('inf')
    total_time = 0
    
    for group_name, timing in actual_times.items():
        exec_time = timing["exec_time"]
        devices = prev_allocation[group_name]["devices"]
        capacity = compute_group_capacity(devices, C_values)
        
        # 計算目前分配的層數
        layer_count = estimate_layer_count(exec_time, capacity)
        
        # 計算閒置率
        total_slot_time = exec_time + timing["idle_time"]
        idle_ratio = timing["idle_time"] / total_slot_time if total_slot_time > 0 else 0
        
        group_metrics[group_name] = {
            "exec_time": exec_time,
            "idle_time": timing["idle_time"],
            "capacity": capacity,
            "layer_count": layer_count,
            "idle_ratio": idle_ratio,
            "devices": devices.copy(),
            "leader": prev_allocation[group_name]["leader"]
        }
        
        max_time = max(max_time, exec_time)
        min_time = min(min_time, exec_time)
        total_time += exec_time
    
    # Step 2: 檢查是否需要重新分配
    imbalance_ratio = (max_time - min_time) / max_time if max_time > 0 else 0
    
    if imbalance_ratio < threshold:
        # 負載均衡已足夠好，無需調整
        return prev_allocation
    
    # Step 3: 根據策略執行重新分配
    if strategy == "greedy":
        new_allocation = _greedy_reallocation(group_metrics, prev_allocation, C_values)
    elif strategy == "optimal_dp":
        new_allocation = _optimal_dp_reallocation(group_metrics, prev_allocation, C_values, total_layers)
    else:
        new_allocation = prev_allocation
    
    return new_allocation


def _greedy_reallocation(
    group_metrics: Dict,
    prev_allocation: Dict,
    C_values: Dict
) -> Dict:
    """
    貪心策略：從快組中移出高性能設備到慢組
    
    削峰：找出最慢的Group（max_time），將其設備移入
    填谷：找出最快的Group（min_time + idle），從中移出設備
    """
    
    # 找出最慢和最快的組
    sorted_groups = sorted(
        group_metrics.items(),
        key=lambda x: x[1]["exec_time"],
        reverse=True
    )
    slowest_group = sorted_groups[0][0]
    fastest_group = sorted_groups[-1][0]
    
    new_allocation = {}
    for group_name in prev_allocation.keys():
        new_allocation[group_name] = prev_allocation[group_name].copy()
    
    # 策略1：從快組移出低性能設備到慢組
    slowest_devices = group_metrics[slowest_group]["devices"]
    fastest_devices = group_metrics[fastest_group]["devices"]
    slowest_cap = group_metrics[slowest_group]["capacity"]
    fastest_cap = group_metrics[fastest_group]["capacity"]
    
    # 找到最慢組中性能最低的設備和最快組中性能最高的設備
    if len(fastest_devices) > 1:
        # 從快組移出一個低性能設備
        min_c_device = min(fastest_devices, key=lambda d: C_values[d])
        device_to_move = min_c_device
        
        # 執行移動
        new_allocation[fastest_group]["devices"].remove(device_to_move)
        new_allocation[slowest_group]["devices"].append(device_to_move)
        
        # 如果移走的是Leader，重新選擇Leader
        if new_allocation[fastest_group]["leader"] == device_to_move:
            new_allocation[fastest_group]["leader"] = max(
                new_allocation[fastest_group]["devices"],
                key=lambda d: C_values[d]
            )
    
    return new_allocation


def _optimal_dp_reallocation(
    group_metrics: Dict,
    prev_allocation: Dict,
    C_values: Dict,
    total_layers: int
) -> Dict:
    """
    DP策略：使用動態規劃重新優化所有設備的分配
    類似fast_cold_start.py中的dp_optimal_partition_leader_star
    """
    
    # 收集所有設備和它們的實際容量
    all_devices = []
    for group_data in prev_allocation.values():
        all_devices.extend(group_data["devices"])
    
    all_devices = list(set(all_devices))  # 去重
    C_list = [C_values[d] for d in all_devices]
    num_groups = len(prev_allocation)
    
    # 調用DP分配（簡化版）
    min_bottleneck = float('inf')
    best_partition = None
    
    for partition in _generate_partitions(all_devices, num_groups):
        max_stage_time = 0
        for group_devices in partition:
            group_capacity = sum(C_values[d] for d in group_devices)
            layers_for_group = total_layers // num_groups  # 簡化假設均分
            stage_time = layers_for_group / group_capacity if group_capacity > 0 else float('inf')
            max_stage_time = max(max_stage_time, stage_time)
        
        if max_stage_time < min_bottleneck:
            min_bottleneck = max_stage_time
            best_partition = partition
    
    # 轉換回allocation格式
    new_allocation = {}
    group_names = list(prev_allocation.keys())
    for i, group_name in enumerate(group_names):
        if best_partition and i < len(best_partition):
            new_allocation[group_name] = {
                "devices": best_partition[i],
                "leader": max(best_partition[i], key=lambda d: C_values[d]),
                "layers": prev_allocation[group_name].get("layers", (0, 0))
            }
        else:
            new_allocation[group_name] = prev_allocation[group_name].copy()
    
    return new_allocation


def _generate_partitions(devices: List[int], num_groups: int, depth: int = 0):
    """生成設備到num_groups個組的所有可能分割（簡化版，僅生成幾個候選）"""
    if num_groups == 1:
        yield [devices]
        return
    
    if len(devices) < num_groups:
        return
    
    # 僅生成固定的幾個分割方案（避免組合爆炸）
    if num_groups == 2:
        for i in range(1, len(devices)):
            yield [devices[:i], devices[i:]]
    else:
        for i in range(1, len(devices) - num_groups + 2):
            first_group = devices[:i]
            for rest in _generate_partitions(devices[i:], num_groups - 1):
                yield [first_group] + rest


if __name__ == "__main__":
    # 示例測試
    C_values = {0: 5.0, 1: 5.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: 3.0, 6: 3.0, 7: 3.0, 8: 3.0, 9: 3.0, 10: 3.0, 11: 3.0, 12: 3.0}
    
    prev_allocation = {
        "Group1": {"devices": [0, 1], "leader": 0, "layers": (0, 10)},
        "Group2": {"devices": [2, 3, 4], "leader": 2, "layers": (11, 21)},
        "Group3": {"devices": [5, 6, 7, 8, 9, 10, 11, 12], "leader": 5, "layers": (22, 31)}
    }
    
    # 模擬實際執行時間（Group2成為瓶頸）
    actual_times = {
        "Group1": {"exec_time": 2.0, "idle_time": 8.5},
        "Group2": {"exec_time": 10.0, "idle_time": 0.5},
        "Group3": {"exec_time": 10.5, "idle_time": 0.0}
    }
    
    new_allocation = task_reallocation(
        prev_allocation,
        actual_times,
        C_values,
        total_layers=32,
        threshold=0.15,
        strategy="greedy"
    )
    
    print("新的分配方案：")
    for group_name, config in new_allocation.items():
        print(f"{group_name}: devices={config['devices']}, leader={config['leader']}")