import numpy as np
import pandas as pd


class NetworkPath:
    def __init__(self, path_id: int, base_latency_ms: float, capacity_mbps: float, buffer_limit_pkts: int):
        self.path_id = path_id
        self.base_latency_ms = base_latency_ms
        self.capacity_mbps = capacity_mbps
        self.buffer_limit_pkts = buffer_limit_pkts
        self.queue_pkts = 0.0
        self.current_loss_rate = 0.0
        self.current_latency_ms = base_latency_ms

    def reset(self):
        self.queue_pkts = 0.0
        self.current_loss_rate = 0.0
        self.current_latency_ms = self.base_latency_ms

    def step(self, routed_pkts: float, bg_noise_factor: float, interval_sec: float = 0.1):
        # Service rate in packets per interval (assume 1500-byte packets ~ 12,000 bits)
        bits_per_packet = 12000.0
        max_service_pkts = (self.capacity_mbps * 1e6 * interval_sec) / bits_per_packet

        # Background transient load variation
        effective_capacity = max_service_pkts * max(0.2, (1.0 - bg_noise_factor))

        total_arrival = self.queue_pkts + routed_pkts

        # Buffer overflow / packet loss calculation
        if total_arrival > self.buffer_limit_pkts:
            dropped = total_arrival - self.buffer_limit_pkts
            self.current_loss_rate = dropped / max(total_arrival, 1.0)
            retained = self.buffer_limit_pkts
        else:
            dropped = 0.0
            self.current_loss_rate = 0.0
            retained = total_arrival

        # Departures
        departed = min(retained, effective_capacity)
        self.queue_pkts = max(0.0, retained - departed)

        # Queuing delay: M/D/1-inspired buffer queuing + base latency
        queuing_delay = (self.queue_pkts / max(effective_capacity, 1.0)) * (interval_sec * 1000.0)
        jitter = np.random.normal(0, 1.5)
        self.current_latency_ms = max(self.base_latency_ms, self.base_latency_ms + queuing_delay + jitter)

        return departed, dropped, self.current_latency_ms, self.current_loss_rate


class AdaptiveMLRouter:
    """
    Offline-trained Q-table for state-action mapping across 3 alternate paths.
    State representation: Discretized relative congestion across (Path 1, Path 2, Path 3).
    Action: Choose Path {0, 1, 2}.
    """
    def __init__(self, n_paths: int = 3):
        self.n_paths = n_paths
        # Discretize each path into 3 states (0: Low, 1: Med, 2: High congestion) -> 3^3 = 27 states
        self.num_states = 3 ** n_paths
        self.q_table = np.zeros((self.num_states, self.n_paths))
        self._train_offline()

    def _discretize_metrics(self, path_latencies, path_losses):
        indices = []
        for lat, loss in zip(path_latencies, path_losses):
            score = lat + (loss * 300.0)
            if score < 25:
                lvl = 0
            elif score < 60:
                lvl = 1
            else:
                lvl = 2
            indices.append(lvl)
        # Convert base-3 tuple to integer state ID
        state_id = indices[0] * 9 + indices[1] * 3 + indices[2]
        return min(max(state_id, 0), self.num_states - 1)

    def _train_offline(self, episodes: int = 2500):
        # Lightweight offline bootstrap to seed policy weights
        for s in range(self.num_states):
            p0 = s // 9
            p1 = (s % 9) // 3
            p2 = s % 3
            # Reward is inversely related to congestion level
            self.q_table[s, 0] = 10.0 - (p0 * 4.0) + np.random.uniform(-0.1, 0.1)
            self.q_table[s, 1] = 10.0 - (p1 * 4.0) + np.random.uniform(-0.1, 0.1)
            self.q_table[s, 2] = 10.0 - (p2 * 4.0) + np.random.uniform(-0.1, 0.1)

    def select_path(self, path_latencies: list, path_losses: list, epsilon: float = 0.05) -> int:
        state = self._discretize_metrics(path_latencies, path_losses)
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.n_paths))
        return int(np.argmax(self.q_table[state]))

    def update_feedback(self, path_latencies, path_losses, action, reward, next_latencies, next_losses, alpha=0.1, gamma=0.9):
        s = self._discretize_metrics(path_latencies, path_losses)
        next_s = self._discretize_metrics(next_latencies, next_losses)
        best_next_action = np.argmax(self.q_table[next_s])
        td_target = reward + gamma * self.q_table[next_s, best_next_action]
        self.q_table[s, action] += alpha * (td_target - self.q_table[s, action])


class MultiPathSimulation:
    def __init__(self, interval_sec: float = 0.1):
        self.interval_sec = interval_sec
        # Initialize 3 asymmetric alternate paths
        self.paths = [
            NetworkPath(path_id=0, base_latency_ms=12.0, capacity_mbps=50.0, buffer_limit_pkts=120),  # Fast, low buffer
            NetworkPath(path_id=1, base_latency_ms=28.0, capacity_mbps=100.0, buffer_limit_pkts=300), # Moderate, high bandwidth
            NetworkPath(path_id=2, base_latency_ms=45.0, capacity_mbps=150.0, buffer_limit_pkts=500), # High baseline latency, bulk pipe
        ]
        self.router = AdaptiveMLRouter(n_paths=3)

    def run(self, total_intervals: int, traffic_load_pkts: float, burstiness: float, enable_ml: bool = True):
        for p in self.paths:
            p.reset()

        records = []
        round_robin_counter = 0

        for t in range(total_intervals):
            # Dynamic traffic flow generator (Poisson arrivals with periodic bursts)
            burst = np.sin(t / 8.0) * burstiness * traffic_load_pkts
            incoming_pkts = max(5.0, np.random.poisson(traffic_load_pkts) + burst)

            current_lats = [p.current_latency_ms for p in self.paths]
            current_losses = [p.current_loss_rate for p in self.paths]

            # Re-evaluate path decision at each interval
            if enable_ml:
                selected_path_idx = self.router.select_path(current_lats, current_losses, epsilon=0.03)
            else:
                selected_path_idx = round_robin_counter % 3
                round_robin_counter += 1

            # Synthetic background interference (e.g. cross-traffic pulses on Path 0 & 1)
            bg_noise = [
                0.6 if (20 <= t <= 45 or 80 <= t <= 100) else 0.05,
                0.5 if (50 <= t <= 75) else 0.1,
                0.05
            ]

            path_stats = {}
            for idx, p in enumerate(self.paths):
                routed = incoming_pkts if idx == selected_path_idx else 0.0
                dep, drop, lat, loss = p.step(routed, bg_noise_factor=bg_noise[idx], interval_sec=self.interval_sec)
                path_stats[f"p{idx}_latency"] = lat
                path_stats[f"p{idx}_queue"] = p.queue_pkts
                path_stats[f"p{idx}_loss"] = loss

            # Calculate network penalty/reward: Minimize latency & drop rate
            chosen_lat = path_stats[f"p{selected_path_idx}_latency"]
            chosen_loss = path_stats[f"p{selected_path_idx}_loss"]
            reward = - (chosen_lat / 10.0 + chosen_loss * 50.0)

            next_lats = [self.paths[i].current_latency_ms for i in range(3)]
            next_losses = [self.paths[i].current_loss_rate for i in range(3)]

            if enable_ml:
                self.router.update_feedback(current_lats, current_losses, selected_path_idx, reward, next_lats, next_losses)

            record = {
                "step": t,
                "selected_path": f"Path {selected_path_idx}",
                "selected_path_id": selected_path_idx,
                "arrival_pkts": incoming_pkts,
                "effective_latency": chosen_lat,
                "effective_loss_rate": chosen_loss,
                **path_stats
            }
            records.append(record)

        return pd.DataFrame(records)