# ancnw 部署改动记录（2026-10-13）

在 rl_sar（v4.0.0）+ MuJoCo 仿真环境中，部署 robot_lab（Isaac Lab / rsl_rl）导出的 ancnw 轮式四足策略。

绝大部分改动是**新增文件**（`fsm_ancnw.hpp`、`policy/ancnw/**`、描述包）；对仓库已有代码的
改动共 4 处，全部向后兼容（都带默认值，不影响其它机器人），见 §1.2。

## 0. 当前基线 / 变更时间线

| 时间 | 事件 |
|---|---|
| 10-13 上午 | 描述包 + MJCF 生成、`joint/@ref` 修正（§5）、策略参数反推（§4） |
| 10-13 16:51 | **当前基线**：抖动 `2`（§6.1）/ 原地踏步 `3`（§6.2）/ 踏步转圈 `4`（§6.3）三个独立 FSM 状态定型 |
| 10-13 17:33–17:42 | 实验性改动（在策略输出上叠加步态 `step_use_rl`、姿态平衡外环 `step_balance_*`、幅度渐入 `step_ramp`、支撑腿外展 `step_hip_spread`、航向锁定 `step_yaw_hold_*`，以及 `rl_sim_mujoco.cpp` 里从观测中扣除步态偏置 `obs_pos_bias/obs_vel_bias`） |
| 10-13 18:06 | 上述实验性改动**全部撤销**，代码/参数回到 16:51 基线；撤销前版本备份在 `/tmp/fsm_ancnw_before_rollback_1806.hpp`、`/tmp/base_before_rollback_1806.yaml`（`base.yaml` 另有一份 17:01 的备份 `/tmp/base_backup.yaml`，与 16:51 版完全一致） |
| 10-14 | 核对并补齐本文件的记录（§1.2 补全已改动的公共文件、§6.4 记录实测与已知问题） |

> 因此：`base.yaml` 当前**没有** `step_leg_kp/kd`、`step_use_rl`、`step_balance_*`、
> `step_ramp`、`step_hip_spread`、`step_yaw_hold_*` 这些键。它们要么有代码内默认值
> （`step_leg_kp/kd` → 0，即不覆盖 `fixed_kp/kd`），要么属于已撤销的实验功能。
> 所有 `params.Get<T>(key, default)` 调用都写了默认值，yaml 缺键不会崩。

> 说明：`src/rl_sar_zoo/` 被 `.gitignore` 第 44 行忽略（描述包由
> `scripts/download_robot_descriptions.sh` 下载），所以 `ancnw_description/` 不在版本控制内；
> 改完生成器后需要重新生成 MJCF。
>
> 与 M20 同源的问题（MJCF `joint/@ref` 让运动学整体偏移）记录在
> `M20_DEPLOY_NOTES.md` §7，这里只记 ancnw 侧的数值。

---

## 1. 文件清单

### 1.1 新增文件

| 路径 | 说明 |
|---|---|
| `src/rl_sar/fsm_robot/fsm_ancnw.hpp` | ancnw 状态机：Passive / GetUp / GetDown / RLLocomotion，`GetType() == "ancnw"` 时由工厂自动创建 |
| `policy/ancnw/base.yaml` | 机器人级参数（`num_of_dofs=16`、`wheel_indices=[12..15]`、`fixed_kp/kd`、`torque_limits`、`default_dof_pos`、`joint_mapping`、`joint_names`、`joint_controller_names`），**策略顺序**（12 腿 + 4 轮，见 §3） |
| `policy/ancnw/robot_lab/config.yaml` | 策略级参数 + 观测定义（57 维，见 §4）、`rl_kp/rl_kd`、`action_scale`、`clip_actions_*`、`wheel_tau_damping: false` |
| `policy/ancnw/robot_lab/policy.pt` | 训练端导出的 TorchScript 模型（输入 57 维，已无需裁剪） |
| `policy/ancnw/tools_urdf2mjcf.py` | URDF → MJCF 转换器（由 `policy/M20/tools_urdf2mjcf.py` 适配）。**不输出 `joint/@ref`**，`BASE_Z=0.537` |
| `policy/ancnw/ancnw_description/urdf/ancnw.urdf`、`meshes/*.STL`（17 个） | URDF 与网格，生成器的输入源（`tools_urdf2mjcf.py` 里的 `URDF` / `MESH_SRC` 直接指向这里） |
| `src/rl_sar_zoo/ancnw_description/urdf/ancnw.urdf`、`meshes/*.STL`（17 个） | 从 `policy/ancnw/ancnw_description/` 复制（该目录被 `.gitignore` 忽略，需要 `scripts/download_robot_descriptions.sh` 或手动复制） |
| `src/rl_sar_zoo/ancnw_description/CMakeLists.txt`、`package.ros2.xml` | 与 `M20_description` 对齐 |
| `src/rl_sar_zoo/ancnw_description/mjcf/ancnw.xml`、`scene.xml`、`assets/*.STL` | 由 `python3 policy/ancnw/tools_urdf2mjcf.py` 生成（gitignored） |

### 1.2 修改文件

| 路径 | 改动 |
|---|---|
| `src/rl_sar/fsm_robot/fsm_all.hpp` | 新增 `#include "fsm_ancnw.hpp"`（+1 行；另有一行 `#include "fsm_M20.hpp"` 属于 M20，见 `M20_DEPLOY_NOTES.md`） |
| `src/rl_sar/fsm_robot/fsm_ancnw.hpp` | 新增 3 个状态：`RLFSMStateJitter`（键 `2`，§6.1）、`RLFSMStateStepInPlace`（键 `3`，§6.2）、`RLFSMStateStepTurn`（键 `4`，§6.3）；共享步态生成器 `BuildStepTargets()` / `WriteStepCommand()`；`Passive/GetUp/GetDown/RLLocomotion` 与各状态之间补齐切换分支；工厂 `AncnwFSMFactory` 注册 7 个状态 |
| `policy/ancnw/base.yaml` | 机器人级参数 + 抖动段 `jitter_*` + 踏步段 `step_*`（§6） |
| `src/rl_sar/library/core/rl_sdk/rl_sdk.cpp` | `ComputeOutput()` 末尾新增 `wheel_tau_damping` 开关（默认 `true` = 旧行为）：为 `false` 时把轮子前馈力矩 `output_dof_tau[wheel]=0`，避免轮阻尼被 `ComputeOutput` 与 `SetCommand` 算两次导致稳态轮速只有指令一半。与 M20 共用，ancnw 侧设 `false`（§4） |
| `src/rl_sar/library/core/rl_sdk/rl_sdk.hpp` | 新增两个成员 `std::vector<float> obs_pos_bias / obs_vel_bias`（预留：把某个状态叠加在策略目标上的偏置从观测里扣掉，让策略不去对抗它）。**当前 ancnw FSM 不使用**（使用它的那段 `rl_sim_mujoco.cpp` 代码随 18:06 回退一起撤销），见 §8 |
| `src/rl_sar/src/rl_sim_mujoco.cpp` | 新增 `RL_SIM_DIAG` 诊断打印（默认关闭，不影响其它机器人）：每 100 个控制周期输出一次基座 z/roll/pitch/yaw/x/y、指令、四轮 `qvel`、策略轮速指令、接触点（§2） |
| `src/rl_sar_zoo/ancnw_description/mjcf/ancnw.xml` | 重新生成：去掉 16 处 `joint/@ref`，基座 `BASE_Z` 0.659 → 0.537，关键帧高度改用各自真实高度（§5） |
| `policy/ancnw/tools_urdf2mjcf.py` | 同上（生成侧） |

---

## 2. 运行方式

```bash
cmake src/rl_sar/ -B cmake_build -DUSE_CMAKE=ON -DUSE_MUJOCO=ON
cmake --build cmake_build -j$(nproc) --target rl_sim_mujoco
./cmake_build/bin/rl_sim_mujoco ancnw scene
# 姿态诊断（默认关闭，不影响其它机器人）
RL_SIM_DIAG=1 ./cmake_build/bin/rl_sim_mujoco ancnw scene
```

### 键位表

| 键 | Gamepad | 目标状态 | 说明 |
|---|---|---|---|
| `0` | `A` | `RLFSMStateGetUp` | 起身到站姿（0.659） |
| `1` | `RB_DPadUp` | `RLFSMStateRLLocomotion` | 交给 RL 策略 |
| `2` | `RB_DPadRight` | `RLFSMStateJitter` | 原地抖动（§6.1） |
| `3` | `RB_DPadLeft` | `RLFSMStateStepInPlace` | 原地踏步（§6.2） |
| `4` | `RB_DPadDown` | `RLFSMStateStepTurn` | 原地踏步转圈（§6.3） |
| `9` | `B` | `RLFSMStateGetDown` | 趴下 |
| `P` | `LB_X` | `RLFSMStatePassive` | 回被动 |
| `W/A/S/D` | — | — | 仅在 RL 状态下发速度指令 |

典型流程：`0` 起身 → `1` 交给 RL（`W/A/S/D` 发速度指令）→ `2`/`3`/`4` 三个原地动作之间可
直接互切（只有 `GetUp` 完成、`percent_getup >= 1` 后才接受切换）→ `9` 趴下 → `P` 回被动。

一键回归（键盘通过 stdin 注入，诊断输出到 stdout）：

```bash
( sleep 2; printf '0'; sleep 6; printf '1'; sleep 6; printf 'w'; sleep 8 ) \
  | RL_SIM_DIAG=1 ./cmake_build/bin/rl_sim_mujoco ancnw scene
```

无头（无显示器）跑同一个回归，用 `xvfb-run` 包一层，之后直接解析 `[DIAG]` 行：

```bash
( sleep 3; printf '0'; sleep 7; printf '3'; sleep 25 ) \
  | RL_SIM_DIAG=1 timeout 60 xvfb-run -a ./cmake_build/bin/rl_sim_mujoco ancnw scene > /tmp/step_3.log 2>&1
grep '\[DIAG\]' /tmp/step_3.log
```

---

## 3. 关节顺序与 `joint_mapping`

与 M20 同一套约定（`M20_DEPLOY_NOTES.md` §3.4 / §4c-1）：

- `base.yaml` 与 `robot_lab/config.yaml` 里的 16 长数组都是**策略顺序**：0–11 是 12 条腿（`fl, fr, hl, hr` × `hip, thigh, calf`），12–15 是 4 个轮子（`fl, fr, hl, hr`）
- MJCF（由 URDF 深度优先转换）是**物理顺序**：`fl_hip, fl_thigh, fl_calf, fl_wheel, fr_..., hl_..., hr_...`，轮子在 3/7/11/15

因此 `joint_mapping = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15]`（策略 → MJCF）。
MJCF 的 `<actuator>` / `<sensor>` / `<keyframe>` 必须按 **MJCF 顺序**输出（`SetCommand()` 已经用 `joint_mapping` 映射过一次，再按策略顺序输出就是二次映射）。

---

## 4. 策略参数来源（没有训练侧 config，全部从 `policy.pt` 权重反推）

| 判据 | 结论 |
|---|---|
| `actor.0.weight` 形状 `[512, 57]` | 观测 57 维 |
| 列范数分块 9–24 / 25–40 / 41–56 明显偏离随机初始化 | 布局 = 3(ang_vel) + 3(gravity_vec) + 3(commands) + 16(dof_pos) + 16(dof_vel) + 16(actions)，与 go2w / b2w 一致 |
| 扰动 `dof_pos[j]` 看轮子响应：j=1,2→轮12；4,5→轮13；7,8→轮14；10,11→轮15 | 12 个腿维按 `fl, fr, hl, hr` 分组，组内 `hip, thigh, calf` |
| `vyaw=+1` 时轮速 `[+0.91, −0.96, +0.56, −0.17]` | 左轮后退 / 右轮前进 = 逆时针，轮序 `fl, fr, hl, hr` |
| `vx=+1` 时四轮输出同为负 | MJCF 轮轴是 `(0,−1,0)`，正轮速把机身推向 −x，策略给负轮速 → 符号正确 |

按 go2w / M20 惯例填的（**待训练侧配置核对**）：

| 项 | 值 |
|---|---|
| `rl_kp / rl_kd` | 腿 40 / 1.0（rl_sar 位置项算两次 → 等效 Isaac 80 / 2.0）；轮 0 / 0.6 |
| `fixed_kp / fixed_kd` | 腿 80 / 2.0；轮 0 / 5.0（GetUp 时起驻车制动作用，轮子不能有位置刚度） |
| `torque_limits` | 腿 113 N·m、轮 19.3 N·m（取自 URDF effort） |
| `action_scale` | hip 0.125、thigh/calf 0.25（位置量）；轮 5.0 rad/s（速度量） |
| `clip_actions` | 腿 ±100；轮 ±18.85 = URDF 速度上限 94.248 / 5.0（MuJoCo 关节无速度上限，只能 clip 动作） |
| 观测 scale | `lin_vel 2.0`、`ang_vel 0.25`、`dof_pos 1.0`、`dof_vel 0.05`、`commands 1.0` |
| `wheel_tau_damping` | `false`：不把 `-rl_kd*dq` 计入轮子前馈力矩，否则轮子阻尼被 `ComputeOutput` + `SetCommand` 算了两次，稳态轮速只有指令一半 |

### `policy/ancnw/robot_lab/config.yaml` 实际写入的项

| 项 | 值 |
|---|---|
| `model_name` | `policy.pt` |
| `num_observations` | 57 |
| `observations` | `["ang_vel", "gravity_vec", "commands", "dof_pos", "dof_vel", "actions"]`（3+3+3+16+16+16） |
| `observations_history` | `[]`（不使用历史帧） |
| `clip_obs` | 100.0 |
| `default_commands` | `[0, 0, 0]` |
| `clip_actions_upper/lower` | 腿 ±100；轮 ±18.85 |
| `rl_kp / rl_kd` | 腿 40 / 1.0；轮 0 / 0.6 |
| `fixed_kp / fixed_kd` | 腿 80 / 2.0；轮 0 / 5.0 |
| `action_scale` | hip 0.125、thigh/calf 0.25；轮 5.0 rad/s |
| `wheel_tau_damping` | `false` |
| 观测 scale | `lin_vel 2.0`、`ang_vel 0.25`、`dof_pos 1.0`、`dof_vel 0.05`、`commands [1,1,1]` |
| `torque_limits` | 腿 113、轮 19.3 |
| `default_dof_pos` | 站姿 `[0, ±0.6, ∓1.0] × 4`，轮 0 |
| `joint_mapping` | `[0,1,2,4,5,6,8,9,10,12,13,14,3,7,11,15]`（策略 → MJCF） |

`policy/ancnw/base.yaml` 里另有机器人级 `dt: 0.005`、`decimation: 4`（50 Hz 控制）。

---

## 5. MJCF `joint/@ref` 修正

MuJoCo 对 hinge 的旋转量是 **(qpos − ref)**，不是 qpos。所以带 `ref="<站立角>"` 时，
`qpos == ref` 在几何上是 **URDF 零位姿**，控制器命令的站立角完全失效，机器人退化成
"刚性四轮小车"——看着不倒，但腿对基座没有任何作用，站高等于零位姿高度。
go2w / b2w 的 MJCF 都不带 `ref`。（根因与实测证据见 `M20_DEPLOY_NOTES.md` §7。）

ancnw 的修正（生成侧 + 生成物）：

- 16 处 `ref=` 全部移除
- `BASE_Z` = **零位姿**贴地高度，站立位姿交给 GetUp 拉起
- 关键帧改用各自真实高度：`home` 0.659、`crouch` 0.652

| 姿态 | 轮心相对基座 z | 基座高度（轮半径 0.11） |
|---|---|---|
| 零位姿（qpos0，出生） | −0.4274 | **0.537**（`BASE_Z`） |
| 站立（thigh ±0.6 / calf ∓1.0） | −0.549 | 0.659（`STANDING_Z`） |
| 蹲（thigh ±1.0 / calf ∓1.9） | −0.542 | 0.652（`CROUCH_Z`） |

生成器常量：`BASE_Z = 0.537`、`STANDING_Z = 0.659`、`CROUCH_Z = 0.652`；
`DEFAULT_POSE` = thigh ±0.6 / calf ∓1.0，`CROUCH_POSE` = thigh ±1.0 / calf ∓1.9
（轮心 −0.542，轮子离地约 7 mm）。

> 注意：生成器/MJCF 关键帧的蹲姿是 **±1.0 / ∓1.9**，而 `fsm_ancnw.hpp` 里
> `RLFSMStateGetUp::pre_running_pos` 用的是 **±0.90 / ∓1.60**。两者都是"折起来的蹲姿"，
> 数值不同不影响运行——关键帧只用于 MuJoCo 复位/初始姿态，起身过程由 FSM 自己的
> `pre_running_pos` 决定（出生在 0.537 零位姿，GetUp 先插值到 `pre_running_pos`，
> 再 2 s 插值到 `default_dof_pos`）。

---

## 6. 抖动 / 踏步 / 踏步转圈（三个独立 FSM 状态）

三者是**不同的状态**，键盘分开：`2` 抖动、`3` 踏步、`4` 踏步转圈。

| 键 | 状态 | 运动特征 |
|---|---|---|
| `2` | `RLFSMStateJitter`（原地抖动） | 站姿不变，腿关节小幅高频正弦（默认 0.05 rad / 5 Hz ≈ 1 cm），四轮始终着地，机身原地抖 |
| `3` | `RLFSMStateStepInPlace`（原地踏步，**位置不偏移**） | 对角双抬，轮子抱死保持进入时的角度，机身原地不动 |
| `4` | `RLFSMStateStepTurn`（原地踏步转圈） | 同一套步态，但左右轮差速（skid steer），机身绕自身轴原地转 |

其它键不变：`0` GetUp、`1` RL、`9` GetDown、`P` Passive。三个状态之间可用 `2`/`3`/`4`
直接互切，`GetUp` 完成（`percent_getup >= 1`）后才接受。

> ⚠ 当前 `base.yaml` 里的抬腿参数（`step_lift` 0.45 / `step_stance_drop` 0.10 /
> `step_period` 1.0）**偏激进，按 `3` 会被弹翻**，详见 §6.4 的调参方向。

`3` 与 `4` 共用一套步态生成器（`BuildStepTargets()` / `WriteStepCommand()`，见
`fsm_ancnw.hpp`），唯一区别是轮子目标：
- `3`：轮子目标 = 进入时的角度（抱死）→ 无净位移
- `4`：轮子目标 = 进入时的角度 ± 累计转角（左 +、右 −）→ 净力抵消，只剩偏航力矩

### 6.1 抖动（`2`）参数

```yaml
jitter_amp: 0.05           # rad（thigh），≈1.0 cm 轮位移
jitter_freq: 5.0           # Hz
jitter_calf_ratio: 2.0     # dcalf = -ratio * dthigh，沿"抬腿方向"抖
jitter_phase_spread: 0.0   # 0 = 四腿同相（机身上下弹）；0.5 = 对角反相
jitter_bias: 0.0           # 0 = 对称正弦；1 = 只抬不压
jitter_transition: 0.5     # 进入时插值到站姿的时间
# jitter_leg_kp / jitter_leg_kd 同时 > 0 时覆盖 fixed_kp/fixed_kd
```

抖动方向刻意复用抬腿方向（前腿 `dthigh = -s, dcalf = +2s`，后腿反号），所以视觉上
是整机上下小幅颠，而不是腿乱摆。

轮子和踏步一样抱死：进入时记下四个轮角 `wheel_locked_pos`，之后用 `step_wheel_kp/kd`
（15 / 2）位置环保持住，所以抖动也不会把机身推走。

运行行实时打印：

```
Jitter in place amp:0.05 rad (1.0 cm) freq:5 Hz
```

### 6.2 原地踏步（新增 FSM 状态 `RLFSMStateStepInPlace`，键 `3`）

按 `3`（Gamepad `RB_DPadLeft`）进入：从当前姿态 0.8 s 插值到站姿，然后对角腿（fl+hr ↔ fr+hl）
交替抬起/落下。**轮子抱死**在进入时的角度，所以踏步不会走偏（实测 16 s 位移 −0.01 m）。
`4` 踏步转圈、`2` 抖动、`1` 回 RL、`0` 回 GetUp、`9` 趴下、`P` 回被动。

运行行实时打印抬腿信息，用来确认腿真的抬起来了：

```
Step in place [trot] clearance:11.2 cm  up:fl+hr (9.1 cm now)  period:1 s
```

`clearance` 是峰值离地间隙（`0.203 m/rad × (step_lift + step_stance_drop)`，纯解析换算），
`up` 是当前处于摆动相的对角腿和它此刻的高度。

#### 为什么默认是对角双抬（trot）而不是 4 拍单腿

实测过两种（同一 `step_lift`）：

| 步态 | 现象 |
|---|---|
| 4 拍 crawl（单腿依次） | 抬起的那条腿**基本看不出离地**：三轮承重时机身向缺腿侧塌，塌下去正好把抬起的轮子压回地面；还会累积偏航（16 s 约 −32°） |
| **对角 trot（默认）** | 支撑的两个轮子连线过质心，机身不塌，抬起的一对**真正离地**（接触在 2 个对角轮 ↔ 4 轮之间交替，实测 16 s 位移 −0.01 m、yaw ±4° 不累积） |

#### 支撑相伸展（`step_stance_drop`）

除了加大 `step_lift`，着地的那一对腿再反向伸展 `step_stance_drop`（负抬升），把机身顶高，
抬起那一对的离地间隙变成 `0.203 × (step_lift + step_stance_drop)`。默认 0.05 → 合计约 8.1 cm。
设 `step_stance_drop: 0` 可关掉。

#### 抬腿量的来源

在站姿附近解腿的 IK（约束"轮子只在竖直方向移动"，`dx = 0`）：

| 抬升 | fl: dthigh / dcalf | hl: dthigh / dcalf |
|---|---|---|
| 3 cm | −0.186 / +0.373 | +0.186 / −0.373 |
| 5 cm | −0.271 / +0.546 | +0.271 / −0.546 |
| 8 cm | −0.378 / +0.760 | +0.378 / −0.760 |

比例恒为 `dcalf = −2 · dthigh`，所以用一个参数 `step_lift = s` 表示：
前腿 `thigh −= s, calf += 2s`，后腿 `thigh += s, calf −= 2s`。

#### 轮子必须抱死

轮子在 GetUp/RL 阶段是速度关节（`fixed_kp = 0`），只有阻尼制动 → 踏步时机身会被反作用力推着走：

| 制动方式 | 16 s 后 x | yaw |
|---|---|---|
| 阻尼 `kd = 5`（`fixed_kd`） | +0.90 m | 持续漂 +19° |
| 阻尼 `kd = 20` | +1.63 m（轮速被拖到 ±12 rad/s） | 更差 |
| **位置环抱死 `kp = 15 / kd = 2`** | **−0.01 m** | ±10° 摆动，不累积 |

#### 步态对比（同一 `step_lift = 0.20`）

| 步态 | 16 s 后 x | yaw | 姿态 |
|---|---|---|---|
| 4 拍 crawl（单腿依次） | +0.13 m | 单调漂到 −32° | roll ±2° |
| **对角 trot（默认）** | **−0.01 m** | ±4° 摆动 | roll/pitch ≈ 0° |

crawl 的抬腿顺序是不对称的，会积累一个偏航力矩；对角同时抬是对称的，不偏航。
上一版曾默认 crawl，实机验证抬起的那条腿看不出离地（机身向缺腿侧塌，把轮子压回地面），
所以默认改回 **trot**。

#### 参数（`policy/ancnw/base.yaml`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `step_period` | 1.0 | 一个完整步态周期（s） |
| `step_lift` | 0.45 | 抬起腿的摆幅（rad），`0.203 m/rad` → 9.1 cm |
| `step_stance_drop` | 0.10 | 支撑腿反向伸展量（rad），把机身顶高；合计离地间隙 `0.203 × (lift + drop)` ≈ **11.2 cm** |
| `step_calf_ratio` | 2.0 | `dcalf = −ratio · dthigh`，保证轮子只在竖直方向动 |
| `step_duty` | 0.40 | 一个周期里该腿滞空占比（最大 0.49） |
| `step_transition` | 0.8 | 进入时插值到站姿的时间（s） |
| `step_wheel_kp / step_wheel_kd` | 15 / 2 | 踏步时的驻车制动；`kp = 0` 退回阻尼制动 |
| `step_trot` | true | `true` 对角双抬（fl+hr ↔ fr+hl，默认），`false` 4 拍单腿依次 |
| `step_leg_kp / step_leg_kd` | 0 / 0 | 同时 > 0 时覆盖 `fixed_kp/fixed_kd`，用来减落腿冲击带来的抖动 |

抬得还不够就依次调大：`step_stance_drop` 0.10 → 0.15、`step_lift` 0.45 → 0.55（≈13 cm），
`step_duty` 0.40 → 0.45。关节限位还很远（thigh ±2.3、calf ±2.8），真正的限制是支撑腿伸展时
机身被顶得太高会晃，所以优先加 `step_lift` 而不是 `step_stance_drop`。

改 `base.yaml` 不用重新编译。

### 6.3 原地踏步转圈（新增 FSM 状态 `RLFSMStateStepTurn`，键 `4`）

按 `4`（Gamepad `RB_DPadDown`）进入：步态与 `3` 完全相同，但轮子不再抱死，而是左右差速：

```
左轮（fl/hl）目标角 = 进入时角度 + θ(t)
右轮（fr/hr）目标角 = 进入时角度 − θ(t)
θ(t) = ω_wheel · t,  ω_wheel = step_yaw_rate · half_track / r_wheel
```

左右速度大小相等方向相反 → 净力为零（不会平移），只剩绕竖直轴的力偶 → 原地转圈。
URDF：轮半径 `r = 0.11 m`，站姿下轮心 `|y| = 0.22 m`（half track），所以
`ω_wheel = 2 × step_yaw_rate`（默认 0.6 rad/s → 轮速 1.2 rad/s）。

方向：正的 `step_yaw_rate` = 逆时针（yaw +）= 左轮后退、右轮前进。要反向就填负值。

```yaml
step_yaw_rate: 0.6           # rad/s 机身角速度，+ = 逆时针
step_turn_half_track: 0.22   # m，URDF 站姿下轮心 |y|
step_turn_wheel_radius: 0.11 # m，URDF 轮半径
step_turn_wheel_omega: 0.0   # > 0 时直接指定轮速 rad/s，跳过上面的换算
```

打印示例：

```
Step turn [trot] clearance:11.2 cm  up:fl+hr  yaw:0.60 rad/s (wheel 1.20 rad/s)
```

注意：转圈靠轮胎横向滑移（skid steer），实际 yaw 速率会低于 `step_yaw_rate`，且抬起
一对腿时只有两个轮子着地、转向力矩更小。转得慢/转不动就加大 `step_yaw_rate`（或
`step_turn_wheel_omega` 直接给轮速），想减少滑移就减小 `step_lift`。

#### 实测（`0` → `3` 踏 16 s → `1`）

> 这一组数据是在**温和抬腿量**（`step_lift` 0.20–0.25、`step_stance_drop` 0、
> `step_duty` 0.35、`step_period` 2.0–3.0 s、`step_wheel_kp` 15）下测的，不是当前
> `base.yaml` 里的 0.45 / 0.10 / 0.40 / 1.0（那组会翻，见 §6.4）。

| 指标 | 结果 |
|---|---|
| z | 0.643–0.651（站姿 0.654） |
| x / y 漂移 | x −0.01 m，y ±0.00 m |
| yaw | ±10° 摆动，不累积 |
| roll / pitch | ≈ 0° |
| 接触 | 在 2 个对角轮 ↔ 4 轮之间交替（抬腿确实离地） |
| 切回 `1` | 正常，z 回到 0.663 |

### 6.4 参数缺省值 / 实测与已知问题

**缺省值规则**：FSM 里所有参数都用 `params.Get<T>(key, default)` 读取，`base.yaml` 缺键时
用代码里的默认值，不会崩。当前 `base.yaml` 显式写的值与代码默认值：

| 参数 | `base.yaml` 当前值 | 代码缺省（yaml 缺键时） |
|---|---|---|
| `step_period` | 1.0 | 1.0 |
| `step_lift` | 0.45 | 0.45 |
| `step_stance_drop` | 0.10 | 0.10 |
| `step_calf_ratio` | 2.0 | 2.0 |
| `step_duty` | 0.40 | trot 0.40 / crawl 0.25（上限 0.49） |
| `step_transition` | 0.8 | 0.5 |
| `step_trot` | true | true |
| `step_wheel_kp / step_wheel_kd` | 15 / 2.0 | 15 / 2.0 |
| `step_leg_kp / step_leg_kd` | *（未写）* → 0 / 0 | 0 / 0，即不覆盖 `fixed_kp/kd` |
| `jitter_*` | 见 §6.1 | amp 0.05 / freq 5 / ratio 2 / spread 0 / bias 0 / transition 0.5 |
| `jitter_leg_kp / jitter_leg_kd` | *（未写）* → 0 / 0 | 0 / 0 |
| `step_yaw_rate` | 0.6 | 0.6 |
| `step_turn_half_track` | 0.22 | 常量 `HALF_TRACK = 0.22` |
| `step_turn_wheel_radius` | 0.11 | 常量 `WHEEL_RADIUS = 0.11` |
| `step_turn_wheel_omega` | 0.0（用几何换算） | 0.0 |

**已知问题：当前默认参数偏激进，按 `3` 会被弹翻。**
无头验证（`xvfb-run` + stdin 注入 `0` → `3`，见 §2）实测：进入踏步后约 5 s 翻倒
（`z` 0.651 → 0.792 被弹起，roll → 180°，pitch 31°，x −0.77 m）。原因是**进入瞬间满幅抬腿
的冲击**把机身顶起来，而不是缓慢漂移；抬起一对腿后只剩两条腿支撑，本身就是静态不稳定的
（轮式点接触，没有支撑多边形）。

调参方向（按有效性排序）：

1. 先降 `step_lift`（0.45 → 0.30 → 0.25）并把 `step_stance_drop` 设 0（别再顶高机身）
2. 放慢：`step_period` 1.0 → 2.0–3.0 s、`step_duty` 0.40 → 0.30–0.35
3. 落腿冲击大时把腿增益调软（`step_leg_kp/kd` 例如 45 / 3.0，需同时 > 0 才生效）
4. 轮子制动加强：`step_wheel_kp` 15 → 25

已验证"不倒且基本不漂"的区间：`step_lift` 0.20–0.25、`step_stance_drop` 0、
`step_duty` 0.35、`step_period` 2.0–3.0 s、`step_wheel_kp` 15。

未完成的实验（17:33 那批，已随回退撤销，记录在此备查）：
`step_lift` 0.30 / `step_stance_drop` 0 / `step_duty` 0.35 / `step_period` 2.0 /
`step_wheel_kp` 25 / `step_leg_kp` 45 / `step_leg_kd` 3.0 —— 日志已生成但结果未解析。

---

## 7. 验证结果

协议：出生（0.537，零位姿贴地）→ `0` 起身 → `1` 交 RL → 零指令 6 s → `W` 前进。

| 指标 | 结果 |
|---|---|
| 站立高度 z | **0.660**（解析 FK 0.659） |
| roll / pitch | +0.2° / ≈0° |
| yaw 漂移 | 15 s 约 0.5° |
| 零指令漂移 | ≈0.02 m/s（约 0.2 m / 10 s） |
| `cmd x=0.10` | 直线前进 ≈0.13–0.15 m/s，横向偏移 0.01 m |
| `cmd vyaw=1.0` | 原地转，yaw 线性增加 |
| 起身过程 | 0.537 → 0.660，无侧翻 |

---

## 8. 待核对 / 残留

1. `rl_kp / rl_kd`、`torque_limits`、`action_scale`（0.125 / 0.25 / 5.0）、`clip_actions`
   都是按 go2w / M20 惯例填的，拿到 `robot_lab` 训练配置后需逐项核对。
2. 零指令下仍有约 0.02 m/s 的前向漂移；速度指令 0.10 → 实测 0.13–0.15 m/s，略快于名义值。
3. Passive（`kp=0`）阶段会自然下沉，这是 rl_sar 的固有行为，由 GetUp 拉起。
4. `rl_sdk.hpp` 里的 `std::vector<float> obs_pos_bias / obs_vel_bias` **当前没有任何使用者**：
   它是 17:33 那批实验（在 RL 策略输出上叠加步态、并把偏置从观测里扣掉）留下的成员，
   使用它的 `rl_sim_mujoco.cpp` 代码已随 18:06 回退一起撤销。如果不再做"策略 + 开环步态叠加"
   的方案，可以连同这两个成员一起删掉；要恢复该方案，则在 `RL_Sim::RobotControl()` 的观测
   组装处（`this->obs.dof_pos = ...` 之后）减去这两个数组。
5. 原地踏步（`3`）的稳定性仍未彻底解决：当前 `base.yaml` 的激进参数会翻（§6.4）。
   已验证不倒的抬腿量只有 4–5 cm，与"抬得高"的诉求还有差距；下一步要么加姿态平衡外环，
   要么改成"策略保持机身 + 只叠加小幅抬腿偏置"的方案（即第 4 条那批实验的思路）。
6. MJCF 生成物的蹲姿关键帧（thigh ±1.0 / calf ∓1.9）与 FSM 的 `pre_running_pos`
   （±0.90 / ∓1.60）数值不同（§5），不影响运行，但如果要统一需要改其中一侧。
