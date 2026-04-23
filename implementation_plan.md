# Paper vs. Code Comparison: Contrastive Initial State Buffer (ICRA24)

Reference: *Messikommer et al., ICRA 2024 - Contrastive Initial State Buffer for RL*

Five bugs identified, listed by severity.

---

## 🔴 Bug 1: Wrong k-Nearest Cluster Selection (CRITICAL)

**Paper:** Select the $N/K$ states **closest** to each cluster center → diversity through representative states.

**Code ([projection_buffer.py](file:///home/trash/Pictures/buffer_state/cl_initial_buffer/initial_buffer/algorithms/projection_buffer.py), line 220):**
```python
# Uses -cluster_distances and takes the LAST k → selects FARTHEST from centroid!
k_smallest_distances_idx = np.argpartition(-cluster_distances, -k, axis=0)[-k:]
```
**The `-` on `cluster_distances` inverts the sort order** — the code selects the $k$ **farthest** points per cluster, the opposite of what the paper says. This corrupts the initial state buffer completely.

**Fix:**
```python
# Select k CLOSEST points per cluster (smallest raw distance)
k_smallest_distances_idx = np.argpartition(cluster_distances, k, axis=0)[:k]
```

---

## 🔴 Bug 2: Data Misalignment — `full_states` vs [observations](file:///home/trash/Pictures/buffer_state/legged_gym/envs/base/legged_robot.py#254-272) (CRITICAL)

**Paper:** Observation $o_t$ and state $s_t$ must come from the **same timestep**. The embedding $f(o_t)$ is used to identify and then reset to $s_t$.

**Code ([on_policy_runner.py](file:///home/trash/Pictures/buffer_state/rsl_rl/rsl_rl/runners/on_policy_runner.py), lines 132–138):**
```python
# Step 1: obs (o_t) is fed to alg.act → stored in storage.observations[step]
actions = self.alg.act(obs, critic_obs)           # stores o_t
# Step 2: env.step(actions) → obs is now o_{t+1}
obs, ..., dones, infos = self.env.step(actions)
# Step 3: full_states stored AFTER step → s_{t+1} !
full_states_buffer[step] = infos["full_states"]   # ← this is s_{t+1}
```
`storage.observations[step]` = $o_t$ but `full_states_buffer[step]` = $s_{t+1}$. 1-step misalignment.

**Fix:** Record `full_states` from `infos` BEFORE calling `env.step()` by storing the state in a separate variable at the beginning of each iteration, or record `infos["full_states"]` at step `t-1` (shift index).

---

## 🔴 Bug 3: `full_states` Recorded AFTER [reset_idx](file:///home/trash/Pictures/buffer_state/legged_gym/envs/base/legged_robot.py#155-234) (CRITICAL)

**Code ([legged_robot.py](file:///home/trash/Pictures/buffer_state/legged_gym/envs/base/legged_robot.py), [post_physics_step](file:///home/trash/Pictures/buffer_state/legged_gym/envs/base/legged_robot.py#105-143)):**
```python
self.check_termination()
self.compute_reward()
env_ids = self.reset_buf.nonzero(...).flatten()
self.reset_idx(env_ids)          # ← robots teleported to new state
self.compute_observations()

self.extras["full_states"] = torch.cat([
    self.root_states[:, :13],    # ← recorded AFTER reset!
    ...
])
```
For any environment that terminates, `full_states` contains the **default reset state**, not the terminal/pre-reset state. These clean default states contaminate the buffer.

**Fix:** Move the `full_states` computation to **before** [reset_idx](file:///home/trash/Pictures/buffer_state/legged_gym/envs/base/legged_robot.py#155-234) is called.

---

## 🟠 Bug 4: `gradient_improvement` Used for Mining, Not [improvement](file:///home/trash/Pictures/buffer_state/cl_initial_buffer/initial_buffer/algorithms/projection_buffer.py#178-198) (MEDIUM)

**Paper:** Mine positives/negatives based on $\Delta V(s) = V^{\pi_1}(s) - V^{\pi_0}(s)$ (absolute improvement).

**Code ([projection_buffer.py](file:///home/trash/Pictures/buffer_state/cl_initial_buffer/initial_buffer/algorithms/projection_buffer.py), [train_step](file:///home/trash/Pictures/buffer_state/cl_initial_buffer/initial_buffer/algorithms/projection_buffer.py#139-163)):**
```python
# Computes delta between consecutive iterations (second-order)
gradient_improvement = improvement_begin_to_step - self.prev_improvement
# Then mines on this DELTA-of-DELTA
pos_ids = np.argpartition(gradient_improvement, -self.nr_mining_samples)[-self.nr_mining_samples:]
```
The paper mines on the absolute $\Delta V(s)$ (improvement vs old policy). The code further subtracts the previous iteration's improvement, making it a 2nd-order difference (acceleration of improvement). This is **not described in the paper** and will penalize states that improved a lot in a previous iteration but are still good.

**Fix:** Mine directly on `improvement_begin_to_step`, not `gradient_improvement`:
```python
pos_ids = np.argpartition(improvement_begin_to_step, -self.nr_mining_samples)[-self.nr_mining_samples:]
neg_ids = np.argpartition(improvement_begin_to_step, self.nr_mining_samples)[:self.nr_mining_samples]
```

---

## 🟡 Bug 5: Missing Performance Filtering and Probabilistic Reset (LOW-MEDIUM)

**Paper specifies:**
- States within the first **15 timesteps** should be excluded from the buffer.
- States where **accumulated episode reward < 0** should be excluded.
- At reset, use ISB with probability **p=0.8**, otherwise use default $s_0$ (p=0.2).

**Code:**
- No timestep-based filtering.
- No reward-based filtering.
- [reset_idx](file:///home/trash/Pictures/buffer_state/legged_gym/envs/base/legged_robot.py#155-234) ([legged_robot.py](file:///home/trash/Pictures/buffer_state/legged_gym/envs/base/legged_robot.py)) always uses all [external_initial_states](file:///home/trash/Pictures/buffer_state/legged_gym/envs/base/legged_robot.py#151-154) with no probability gate.

**Fix:**
- Add reward accumulation tracking per env in the runner loop and filter below-zero trajectories before passing to [create_train_data](file:///home/trash/Pictures/buffer_state/cl_initial_buffer/initial_buffer/algorithms/projection_buffer.py#63-134).
- Add a `p=0.8` probability mask in [reset_idx](file:///home/trash/Pictures/buffer_state/legged_gym/envs/base/legged_robot.py#155-234) so 20% of resets still use random initial states.

---

## Summary Table

| # | Component | Paper Says | Code Does | Severity |
|---|-----------|-----------|-----------|----------|
| 1 | [create_initial_state_buffer](file:///home/trash/Pictures/buffer_state/cl_initial_buffer/initial_buffer/algorithms/projection_buffer.py#199-226) | k-**closest** to centroid | k-**farthest** (sign flipped) | 🔴 Critical |
| 2 | `on_policy_runner` rollout | $o_t$ aligns with $s_t$ | $o_t$ aligns with $s_{t+1}$ | 🔴 Critical |
| 3 | [post_physics_step](file:///home/trash/Pictures/buffer_state/legged_gym/envs/base/legged_robot.py#105-143) | record $s$ before reset | records $s$ after reset | 🔴 Critical |
| 4 | [train_step](file:///home/trash/Pictures/buffer_state/cl_initial_buffer/initial_buffer/algorithms/projection_buffer.py#139-163) mining | mine on $\Delta V$ | mines on $\Delta(\Delta V)$ | 🟠 Medium |
| 5 | [reset_idx](file:///home/trash/Pictures/buffer_state/legged_gym/envs/base/legged_robot.py#155-234) | p=0.8 buffer, filter rewards | always buffer, no filter | 🟡 Medium |
