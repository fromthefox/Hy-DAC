import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

"""
TODO:这里需要去测试一下真实的计算数据,然后算一下真实的 Llama-3-8B 的通信开销,带入算一下结果如何.
"""

@dataclass
class GroupAssignment:
    device_indices: List[int]
    leader: int
    layer_range: Tuple[int, int]
    per_layer_time: float
    stage_time: float

def compute_single_layer_time_leader_star(
    devices: List[int],
    C: List[float],
    bandwidth: List[List[float]],
    latency: List[List[float]],
    S_in: float,
    S_out: float,
    alpha_dispatch: float = 1.0,
    beta_gather: float = 1.0,
    gamma_coord: float = 0.05,
    rho_bcast: float = 0.5,
    rho_gather: float = 0.5,
    overlap_mode: str = "partial",
    theta_imbalance: float = 0.3,
) -> Tuple[float,int]:
    """
    计算星型 Leader-Worker 下该设备集合执行一层的时间。
    返回：(最优单层时间, 选定leader原始索引)
    """
    n = len(devices)
    if n == 0:
        raise ValueError("Empty device group.")
    if n == 1:
        d = devices[0]
        T_compute = 1.0 / C[d]
        return T_compute, d

    sum_C = sum(C[d] for d in devices)
    T_compute = 1.0 / sum_C

    maxC = max(C[d] for d in devices)
    minC = min(C[d] for d in devices)
    imbalance = (maxC - minC) / sum_C
    T_imbalance = theta_imbalance * imbalance * T_compute

    best_time = math.inf
    best_leader = devices[0]

    for leader in devices:
        bcast_terms_seq = []
        gather_terms_seq = []
        bcast_terms_par = []
        gather_terms_par = []

        for w in devices:
            if w == leader:
                continue
            t_bw = latency[leader][w] + S_in / bandwidth[leader][w]
            bcast_terms_seq.append(t_bw)
            bcast_terms_par.append(t_bw)
            t_gw = latency[w][leader] + S_out / bandwidth[w][leader]
            gather_terms_seq.append(t_gw)
            gather_terms_par.append(t_gw)

        if bcast_terms_seq:
            T_bcast_seq = sum(bcast_terms_seq)
            T_bcast_par = max(bcast_terms_par)
            T_bcast = (1 - rho_bcast) * T_bcast_seq + rho_bcast * T_bcast_par
        else:
            T_bcast = 0.0

        if gather_terms_seq:
            T_gather_seq = sum(gather_terms_seq)
            T_gather_par = max(gather_terms_par)
            T_gather = (1 - rho_gather) * T_gather_seq + rho_gather * T_gather_par
        else:
            T_gather = 0.0

        T_bcast_eff = alpha_dispatch * T_bcast
        T_gather_eff = beta_gather * T_gather
        T_coord = gamma_coord * (n - 1) * T_compute

        if overlap_mode == "none":
            T_layer = T_bcast_eff + T_compute + T_gather_eff + T_coord
        elif overlap_mode == "partial":
            T_layer = max(T_bcast_eff + T_coord, T_compute) + T_gather_eff
        elif overlap_mode == "full":
            T_layer = max(T_bcast_eff + T_gather_eff + T_coord, T_compute)
        else:
            raise ValueError("Invalid overlap_mode")

        T_layer += T_imbalance

        if T_layer < best_time:
            best_time = T_layer
            best_leader = leader

    return best_time, best_leader

def dp_optimal_partition_leader_star(
    C: List[float],
    M: int,
    bandwidth: List[List[float]],
    latency: List[List[float]],
    S_in: float,
    S_out: float,
    alpha_dispatch: float = 1.0,
    beta_gather: float = 1.0,
    gamma_coord: float = 0.05,
    rho_bcast: float = 0.5,
    rho_gather: float = 0.5,
    overlap_mode: str = "partial",
    theta_imbalance: float = 0.3,
    sort_devices: bool = True
) -> Dict[str, Any]:
    """
    使用 DP 最小化最大阶段时间（Leader-Star 模型）。
    """
    N = len(C)
    original_indices = list(range(N))
    if sort_devices:
        sorted_pairs = sorted(zip(original_indices, C), key=lambda x: x[1], reverse=True)
        sorted_indices = [p[0] for p in sorted_pairs]
        C_sorted = [p[1] for p in sorted_pairs]
    else:
        sorted_indices = original_indices
        C_sorted = C

    single_layer_time = [[0.0]*N for _ in range(N)]
    leader_choice = [[-1]*N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            devices_range_sorted_pos = list(range(i, j+1))
            devices_original = [sorted_indices[d] for d in devices_range_sorted_pos]
            t_layer, leader = compute_single_layer_time_leader_star(
                devices_original,
                C,
                bandwidth,
                latency,
                S_in,
                S_out,
                alpha_dispatch,
                beta_gather,
                gamma_coord,
                rho_bcast,
                rho_gather,
                overlap_mode,
                theta_imbalance
            )
            single_layer_time[i][j] = t_layer
            leader_choice[i][j] = leader

    INF = 1e18
    DP = [[INF]*(M+1) for _ in range(N+1)]
    Prev = [[None]*(M+1) for _ in range(N+1)]
    DP[0][0] = 0.0

    for i in range(1, N+1):
        for m in range(1, M+1):
            for d in range(0, i):
                for l in range(0, m):
                    if DP[d][l] == INF:
                        continue
                    per_layer_t = single_layer_time[d][i-1]
                    layer_count = m - l
                    stage_time = per_layer_t * layer_count
                    candidate = max(DP[d][l], stage_time)
                    if candidate < DP[i][m]:
                        DP[i][m] = candidate
                        Prev[i][m] = (d, l, (d, i-1), (l+1, m), per_layer_t, stage_time)

    if DP[N][M] == INF:
        raise ValueError("No feasible partition.")

    assignments: List[GroupAssignment] = []
    cur_i, cur_m = N, M
    while cur_i > 0 and cur_m > 0:
        prev = Prev[cur_i][cur_m]
        if prev is None:
            raise ValueError("Backtracking failed.")
        d_prev, l_prev, dev_range, layer_range, per_layer_t, stage_t = prev
        di_start, di_end = dev_range
        devices_sorted_range = list(range(di_start, di_end+1))
        devices_original = [sorted_indices[x] for x in devices_sorted_range]
        leader = leader_choice[di_start][di_end]
        assignments.append(GroupAssignment(
            device_indices=devices_original,
            leader=leader,
            layer_range=layer_range,
            per_layer_time=per_layer_t,
            stage_time=stage_t
        ))
        cur_i, cur_m = d_prev, l_prev

    assignments.reverse()

    return {
        "optimal_max_stage_time": DP[N][M],
        "Q": len(assignments),
        "groups": assignments
    }

if __name__ == "__main__":
    # 这里的例子：
    # 模型：Llama-3-8B fp16
    # 设备：8个Pi2B, 3个Pi3B, 2个Pi4B
    C = [3.0]*8 + [4.0]*3 + [5.0]*2
    N = len(C)
    M = 32  # Llama-3-8B 层数

    B_base = 10_000_000.0  # 10MB/s
    L_base = 0.040  # 40ms
    bandwidth = [[B_base for _ in range(N)] for _ in range(N)]
    latency   = [[0.0 if i==j else L_base for j in range(N)] for i in range(N)]

    # 激活大小：Llama-3-8B fp16, hidden=4096, 8192 bytes
    S_in  = 8192.0
    S_out = 8192.0

    result = dp_optimal_partition_leader_star(
        C=C,
        M=M,
        bandwidth=bandwidth,
        latency=latency,
        S_in=S_in,
        S_out=S_out,
        alpha_dispatch=1.0,
        beta_gather=1.0,
        gamma_coord=0.05,
        rho_bcast=0.5,
        rho_gather=0.5,
        overlap_mode="full",
        theta_imbalance=0.3,
        sort_devices=True
    )
    print("Optimal max stage time (Leader-Star):", result["optimal_max_stage_time"])
    print("Q (number of groups):", result["Q"])
    for idx, g in enumerate(result["groups"], 1):
        print(f"[Group {idx}] leader={g.leader} devices={g.device_indices} "
              f"layers={g.layer_range} per_layer_time={g.per_layer_time:.6f} "
              f"stage_time={g.stage_time:.6f}")