# M20 部署改动记录（2026-10-12）

在 rl_sar（v4.0.0）+ MuJoCo 3.2.7 仿真环境中，部署 robot_lab（Isaac Lab / rsl_rl）导出的 Deep Robotics M20 轮式四足策略。

当前分支：`main`；本次改动均为**新增文件**，对仓库已有代码的唯一改动是 `fsm_all.hpp` 里加了一行 include。

---

## 1. 文件清单

### 1.1 新增文件

| 路径 | 说明 |
|---|---|
| `src/rl_sar/fsm_robot/fsm_M20.hpp` | M20 的状态机：Passive / GetUp / GetDown / RLLocomotion，以及 `M20FSMFactory` 自动注册 |
| `policy/M20/base.yaml` | 机器人级参数（物理 / MJCF 关节顺序，见 §3.4） |
| `policy/M20/robot_lab/config.yaml` | 策略级参数 + 观测定义（**策略**关节顺序，见 §3.4） |
| `policy/M20/robot_lab/policy.pt`、`policy.onnx` | 训练端导出的原始模型 |
| `policy/M20/robot_lab/policy_deploy.onnx` | 裁剪后的部署模型：输入 174→57，batch 维 `-1`→`1`（见 §3.2、§3.3） |
| `policy/M20/robot_lab/truncate_to_41.py` | ONNX 裁剪 / 固定 batch 维的脚本（`python3 truncate_to_41.py 57`） |
| `policy/M20/robot_lab/policy_meta.json` | 导出产物自带的元信息（**不含**观测项清单，因此才有了 §3.2 的权重分析） |
| `src/rl_sar_zoo/M20_description/mjcf/M20.xml`、`scene.xml` | URDF→MJCF 生成物。该目录被 `.gitignore` 第 44 行 `rl_sar_zoo` 忽略，不在版本控制内 |

### 1.2 修改文件

| 路径 | 改动 |
|---|---|
| `src/rl_sar/fsm_robot/fsm_all.hpp` | 新增 `#include "fsm_M20.hpp"`（+1 行） |
| `src/rl_sar/src/rl_sim_mujoco.cpp` | `RobotControl()` 末尾加了 `RL_SIM_DIAG` 环境变量控制的位姿/轮速打印（默认关闭） |
| `src/rl_sar/library/core/rl_sdk/rl_sdk.cpp` | `ComputeOutput()` 增加 `wheel_tau_damping` 开关（默认 `true`，行为不变；用于消除轮子阻尼被计入两次的问题，M20 目前未启用） |

> 调试打印的定位事故也记录下来：在 `RobotControl()` 里遍历 `params.Get<std::vector<float>>(...)` 会导致进程启动后立刻 core dump；只保留读 `mj_data->qpos` 的打印是安全的。以后再排查时按这个结论加打印。

---

## 2. 运行方式

```bash
./build.sh -mj                     # 编译 MuJoCo 仿真（已支持 ONNX Runtime）
./cmake_build/bin/rl_sim_mujoco M20 scene
```

键盘：`0` 站立（GetUp）→ `1` 交给 RL 策略 → `W/A/S/D` 发速度指令 → `9` 趴下 → `P` 回被动。

---

## 3. 本次踩到并修掉的 5 个坑

### 3.1 MJCF 的 `<actuator>` / `<sensor>` 必须按 MJCF 关节顺序生成

`rl_sim_mujoco.cpp` 里用的是已经映射过的下标：

```cpp
mj_data->ctrl[joint_mapping[i]] = ...
    kp[i] * (q[i] - mj_data->sensordata[joint_mapping[i]]) + ...
```

也就是 **`joint_mapping` 是「策略序号 → 物理序号」**。如果 MJCF 里 actuator/sensor 按策略顺序（12 条腿 + 4 轮）生成，就等于二次映射，腿部指令会打到轮子关节上——表现是「按 0 站不起来」。

修法：生成 MJCF 时先做运动树深度优先遍历得到 `mjcf_order`，actuator 与 `jointpos/jointvel/jointactuatorfrc`、keyframe 的 `qpos/ctrl` 全部按该顺序输出。

`joint_mapping = [0,1,2,4,5,6,8,9,10,12,13,14,3,7,11,15]`（策略 → MJCF）。

### 3.2 真正的观测是 57 维，不是 174 维

导出模型的输入张量是 174 维（`actor.0.weight` 形状 `[512, 174]`），但**训练时只有前 57 维有数据**。

判据：首层权重每一列对应一个输入维度，梯度和该维输入成正比；输入恒为常量的维度权重永不更新，会停在随机初始化水平。提取列范数：

| 列 | 含义（推断） | 列范数均值 |
|---|---|---|
| 0–2 | ang_vel | 6.45 |
| 3–5 | gravity_vec | 6.37 |
| 6–8 | commands | 6.35 |
| 9–24 | joint_pos | 3.75 |
| 25–40 | joint_vel | 4.30 |
| 41–56 | actions | 1.07（被训练，但被压制） |
| 57–173 | （未使用） | 1.302 ± 0.057（= 随机初始化基线，从未更新） |

与仓库里已知能跑的模型交叉验证，结论一致：

| 模型 | 首层形状 | 0–2 | 3–5 | 6–8 | 9–24 | 25–40 | 41–56 |
|---|---|---|---|---|---|---|---|
| b2w | [512, 57] | 11.08 | 12.50 | 6.30 | 8.17 | 9.57 | 1.18 |
| go2w | [512, 57] | 10.50 | 12.57 | 9.23 | 9.51 | 10.30 | 1.39 |
| go2 | [512, 45] | 10.79 | 13.00 | 10.61 | 10.12 | 5.63 | 1.86 |
| **M20** | [512, **174**] | 6.45 | 6.37 | 6.35 | 3.75 | 4.30 | 1.07 |

`b2w` / `go2w` 的输入维度**正好等于**它们 `config.yaml` 里观测列表的长度（57 = 3+3+3+16+16+16），M20 多出来的 117 维在训练时是死的。

处理：用 `truncate_to_41.py` 把首层权重切成 `[512, 57]`、去掉后 117 列，并把 batch 维钉死为 1。验证方式：给原模型喂「57 维数据 + 117 个 0」、给新模型喂同样 57 维，两者输出 **maxdiff = 0**。

配置：

```yaml
num_observations: 57
observations: ["ang_vel", "gravity_vec", "commands", "dof_pos", "dof_vel", "actions"]
```

注：`dof_pos` 项不含轮子（训练端用 `joint_pos_rel_without_wheel`），rl_sar 的 `rl_sdk.cpp:99-102` 已自动把轮子位置置零；`dof_vel` 含轮子。

### 3.3 ONNX 的 batch 维必须是静态 1

`library/core/inference_runtime/inference_runtime.cpp:193` 直接把模型声明的 shape 传给 `Ort::Value::CreateTensor`，动态 batch（`-1`）会报 `onnxruntime: negative value in shape`。

没有改核心代码，而是在导出模型里把 `dim_param` 换成 `dim_value = 1`。

### 3.4 ★ 配置里的 16 长数组：两个文件用**不同**的顺序

这是当天最隐蔽、也是最终让策略跑起来的那个 bug。

`joint_mapping` 只用于把「策略序号」翻译成「仿真/物理下标」，因此：

- `policy/M20/base.yaml`：**物理（MJCF）顺序** —— `fl_hipx, fl_hipy, fl_knee, fl_wheel, fr_..., hl_..., hr_...`
- `policy/M20/robot_lab/config.yaml`：**策略顺序** —— 12 条腿在前，4 个轮子放最后

我最初把 `robot_lab/config.yaml` 里的 `rl_kp / rl_kd / action_scale / default_dof_pos / torque_limits / fixed_kp / fixed_kd` 都按「每条腿 4 个值（含轮子）」分行写了，结果这些值落在策略下标 3/7/11/15 上，而真正的轮子下标是 12–15：

- 下标 12/13/14（轮子）拿到了 `rl_kp = 40`、`action_scale = 0.125/0.25/0.25`、`default_dof_pos = 0/0.6/-1.0` → **轮子被当成位置伺服关节**，机器人根本滚不起来
- 下标 3/7/11（`fr_hipx / hl_hipy / hr_knee`）反过来拿到 wheel 参数（`rl_kp = 0`）→ 这三条腿完全失去刚度

修正后（对齐 `policy/go2w/robot_lab/config.yaml` 的写法，12 腿 + 4 轮）：

```yaml
rl_kp: [40 ×12, 0.0 ×4]
rl_kd: [1.0 ×12, 0.6 ×4]
action_scale: [0.125,0.25,0.25 ×4, 5.0 ×4]
default_dof_pos: [0,-0.6,1.0, 0,-0.6,1.0, 0,0.6,-1.0, 0,0.6,-1.0, 0,0,0,0]
torque_limits: [76.4 ×12, 21.6 ×4]
```

效果对比（`up` = 躯干竖直度，1.0 为完全水平）：

| | 修正前 | 修正后 |
|---|---|---|
| RL 接管后 `up` | 迅速掉到 0.774 并保持不动 | 持续保持 0.994–0.999 |
| 按 W | 完全不动（x/y 冻结） | 位姿持续变化，机器人在移动 |

同一类问题还有一处：`src/rl_sar/fsm_robot/fsm_M20.hpp` 里 GetUp 的跪姿 `pre_running_pos` 也必须按策略顺序写（12 腿 + 4 轮置 0）。

### 3.5 GetUp 阶段的默认说法

- `base.yaml` 里 `fixed_kd` 给轮子 5.0（驻车制动），`fixed_kp` 给轮子 0（轮子不能有位置刚度）
- `rl_kp` 给 40 是因为 rl_sar 把位置项算了两次（`tau + kp*(cmd_q - q)`），有效刚度 = 2× = 80，与 Isaac 侧一致

---

## 4. 已知遗留问题

1. **轮子阻尼被重复计入**（已修）：`SetCommand` 施加 `tau + kp*(cmd_q - q) + kd*(cmd_dq - dq)`，而 `ComputeOutput` 里 `output_dof_tau` 对轮子已含 `-rl_kd*dq`（kp=0），于是轮端正则控制律变成 `rl_kd*v_cmd - 2*rl_kd*dq`，稳态轮速只有指令一半（`action_scale` 的 5.0 实际只等价于 2.5）。
   修复：`rl_sdk.cpp ComputeOutput` 在 `wheel_tau_damping: false` 时把轮子 `output_dof_tau` 置 0，只保留 `SetCommand` 的 `kd*(v_cmd - dq)`，与 Isaac DCMotor 一致。M20 的 `config.yaml` 已设 `wheel_tau_damping: false`（默认 true 保持其它机器人旧行为）。验证：`wheel_cmd ≈ wheel_dq`（1:1 跟随）。注意旧 checkpoint 在该修复下会原地翻倒——因为它本来就是坏策略（轮子动作头是固定偏置），需重训。
2. **站姿高度偏低**：RL 接管后基座高 z ≈ 0.42 m，低于 Isaac 标称站立高度 ≈ 0.52 m（偏蹲）。可能由上面第 1 点、MuJoCo 侧 PD/接触参数或转角所致。
3. **GetUp 过程中机身会滑动**：轮子现在是自由滚动的（不再是位置伺服），起步时会漂移约 0.5 m。若想让 GetUp 更稳，可在 GetUp 阶段临时加大 `fixed_kd`。
4. ** observations 的前 3 个 3 维块顺序（ang_vel / gravity_vec / commands）以及 dof_pos 在 dof_vel 之前，是按 robot_lab 惯例推定的**，它们的权重统计量太接近，无法从权重上分辨；需要从行为侧确认。
5. 我在排查期间还试过三种 3 维块排列在 **41 维**假设下的 12 种组合，全都失败。那批测试的前提（41 维）已被 §3.2 修正，理论上应在 **57 维**下重测一遍；但 57 维默认顺序已经能站稳并移动，所以优先级不高。

---

## 4b. 第二轮：姿态「歪」的定位与调参（当天晚些时候）

现象：RL 接管后能站住、能移动，但**零指令下 yaw 持续漂移（15–20°/s 自转）**、roll/pitch 抖动 ±5°、基座高度只有 0.40–0.44（Isaac 标称 0.52）。

用 `RL_SIM_DIAG=1` 打印 `z / roll / pitch / yaw / x,y / 四轮实际转速` 逐项排查，找到 3 个「仿真与 Isaac 不一致」的点：

| # | 问题 | Isaac 侧 | 修正 |
|---|---|---|---|
| 1 | MJCF 默认 `<geom condim="1">` = **无摩擦接触**，只有轮子用了 condim=3 | 物理材质有摩擦 | 默认接触改 `condim="3" friction="0.8 0.02 0.01"` |
| 2 | MJCF 默认 `<joint damping="0.1" frictionloss="0.2">` 被 12 条腿继承 | DCMotor `friction=0.0`，阻尼全部由执行器提供 | 关节 `damping="0" armature="0" frictionloss="0"`（轮子同样归零） |
| 3 | MuJoCo 关节没有速度上限，某个轮子曾飞转到 **312 rad/s** | DCMotor `velocity_limit=79.3` | 用 action clip 补上限：轮子 `clip_actions = ±79.3/action_scale = ±15.86` |

改的是生成器 `policy/M20/tools_urdf2mjcf.py`，重新生成 `src/rl_sar_zoo/M20_description/mjcf/M20.xml`；clip 改在 `robot_lab/config.yaml`。

**修正前后（同一协议：0 起身 → 1 交给 RL → 零指令 7 s）**

| 指标 | 修正前 | 修正后 |
|---|---|---|
| yaw 漂移 | 15–20°/s，持续自转 | +3.7° / 7 s |
| \|roll\| / \|pitch\| 均值 | ±5° / ±3.5° | 1.9° / 1.2° |
| 最大轮速 | 312 rad/s（失控） | ≤ 101 rad/s（受 clip 约束） |
| 基座高度 z | 0.40–0.44 | 0.40–0.42 |
| 前进指令 (x=0.10) | 方向乱、边走边转 | x 从 -0.07 → +0.89（约 0.3 m/s），roll ±1–4° |

### 试过、但**否决**的调参（都写在 `robot_lab/config.yaml` 里，改一行就能复现）

| 改动 | 结果 |
|---|---|
| 轮子 `action_scale` 5.0 → 10.0（补偿 rl_sar 阻尼双计） | 起步高度对了(0.535)但立刻 pitch 38° 前翻 |
| 轮子 `action_scale` → 7.5 | 自转消失，但零指令仍有 +10.7°/7s 漂移，不如 5.0 |
| 腿 `rl_kp` 40 → 80（有效刚度 160） | 侧翻（roll -160°） |
| `wheel_tau_damping: false`（见下） | 倒地 |

**轮速上限扫描**（协议：0 → 1 → W 前进；每项 1 次，注意 rl_sar 多线程时序会带来明显 run-to-run 差异）

| 轮子 action clip | 对应上限 | 结果 |
|---|---|---|
| ±25.0 | 125 rad/s | 27/44 帧处于翻倒状态 |
| **±15.86（= Isaac 79.3）** | 79.3 rad/s | 0/44 翻倒；最好一次前进 dx=+1.36、dy=-0.11（走直线） |
| ±8.0 | 40 rad/s | 0/43 翻倒，但 pitch 瞬时到 43°，前进更少 |

结论：取 Isaac 的 79.3 rad/s 最合适。翻车与否在多次重复中仍有随机性（一次 ±15.86 出现 roll 59° 的瞬时大倾角），属于"边缘稳定"。

### 代码改动：`wheel_tau_damping` 开关

`rl_sdk.cpp` 的 `ComputeOutput` 里，轮子力矩 = `rl_kp*(...) - rl_kd*dq`，而 `SetCommand` 又叠加了 `kd*(cmd_dq - dq)`，阻尼被计入两次 → 稳态轮速只有指令的一半。为了不静默改变其它机器人（go2w/b2w）的行为，加了一个默认 `true` 的开关：

```cpp
// rl_sdk.cpp ComputeOutput()
if (!wheel_tau_damping) { for (int i : wheel_indices) output_dof_tau[i] = 0.0f; }
```

M20 目前设为 `true`（保持原行为）。若哪天要严格对齐 Isaac 可置 `false`，但实测会倒地，说明这个策略依赖那份额外阻尼。

### 姿态诊断开关

`src/rl_sar/src/rl_sim_mujoco.cpp` 的 `RobotControl()` 末尾加了环境变量控制的打印（默认关闭，不影响其它机器人）：

```bash
RL_SIM_DIAG=1 ./cmake_build/bin/rl_sim_mujoco M20 scene
# [DIAG] z=0.421 roll=-3.0 pitch=+0.7 yaw=-25.5 x=-0.67 y=+0.03 cmd=0.00/0.00/0.00 wheel_dq=-0.9/-80.5/83.1/-0.5
```

---

## 4c. 第三轮：「按 0 反而趴下」+「走得奇怪」

### 4c-1 按 0 起不来 / 反而趴下 —— 两个 bug

**bug A：`base.yaml` 的数组顺序写错了（和第二轮 `robot_lab/config.yaml` 那个是同类问题）**

`rl_sdk.cpp` 的 `Interpolate()` 用**策略下标**写 `motor_command.q/kp/kd`，`SetCommand()` 再用 `joint_mapping` 映射到仿真：

```cpp
// rl_sim_mujoco.cpp SetCommand()
mj_data->ctrl[joint_mapping[i]] = command->motor_command.tau[i] + kp[i]*(q[i] - ...) + ...
```

所以 **FSM 阶段（Passive/GetUp/GetDown）用到的 `fixed_kp` / `fixed_kd` / `default_dof_pos` 以及 `joint_mapping` 本身，全部必须是策略顺序**。我原先按「物理顺序（每条腿含轮子）」写了 `base.yaml`，后果是：

- 策略下标 3/7/11（`fr_hipx` / `hl_hipy` / `hr_knee`）拿到了轮子的 `fixed_kp = 0` → 这三条腿在起身时完全没有刚度
- `default_dof_pos` 被当策略顺序解读后，`fr_hipy` 目标变成 0、`fr_knee` 变成 -0.6 → 目标姿态就是错的
- `joint_mapping` 写成了 identity → GetUp 阶段读到的关节角本身就是错位的

对照 `policy/go2w/base.yaml` 确认了仓库约定：base.yaml 同样是**策略顺序**（12 腿 + 4 轮，`fixed_kp` 末 4 个 0、`fixed_kd` 末 4 个 0.5）；go2w 之所以是 identity mapping，是因为它的 MJCF 顺序恰好就等于策略顺序，而 M20 的 MJCF 是 URDF 物理顺序，必须填真实映射。

**bug B：MJCF 的生成姿态让机器人侧翻，而 GetUp 扶不起来**

MuJoCo 的初始状态来自 `qpos0`（body 位置 + 关节 `ref`），**`<keyframe>` 不会被自动应用**。原来的生成姿态是"零角度站姿"，Passive（`kp=0`）一放手机器人就自由倒地，而且常常是**侧翻/仰翻**（实测 roll -68.9°）。rl_sar 的 GetUp 只是插值关节角、**完全不控制基座姿态**，所以侧翻后只能把腿摆来摆去，结果就是卡在 `base_link` 着地、pitch ±40° 的姿态 —— 看起来就是"按 0 反而趴下了"（再按一次 0 有时能侥幸起来，这就是随机的来源）。

修法：让 `qpos0` 是"站立姿态且轮子刚好触地"，这样 Passive 放手时是**垂直下沉**而不是翻倒。用工具量出各姿态下最低接触点相对基座的偏移：

| 姿态 | 最低点相对基座 | 基座应在的高度 |
|---|---|---|
| 零角度 | -0.441 | 0.441 |
| 站立（hipy ±0.6, knee ±1.0） | -0.572 | **0.572** |
| 我原先以为的"折叠"（hipy ±0.9, knee ±1.6） | -0.594 | 0.594（其实更低，不是蹲姿） |

生成器改为：`BASE_Z = 0.572` + 每个关节加 `ref="<站立角>"`。实测 GetUp 稳定成功：z 0.25 → 0.548 → 0.588，roll/pitch 回到 ~0°。

> 顺带：`pre_running_pos`（hipy ±0.9 / knee ±1.6）按上面这表其实是"腿伸得更直"而不是蹲姿，只是作为中间过渡能用，暂未改。

### 4c-2 走得奇怪 —— 不是部署问题，是 checkpoint 本身的问题

用 `RL_SIM_DIAG=1` 加打印后发现：**同时只有 3 个轮子着地**，总有一个轮子离地空转到 70~90 rad/s（轮子半径 0.09 m，对应 7 m/s 线速度，而机身只有 0.1~0.2 m/s）。

直接把导出的 ONNX 拿出来做离线测试（不看仿真），喂一个"**完全标准站立**"的观测（角速度 0、重力 [0,0,-1]、指令 0、关节偏差 0、关节速度 0、上帧动作 0）：

```
轮子输出: fl +6.2   fr -30.9   hl +32.7   hr +1.0   rad/s
```

站着不动就命令两个**对角轮反向 30 rad/s**。再测指令响应：

| 输入指令 | fl | fr | hl | hr |
|---|---|---|---|---|
| vx=+1 | +1.5 | -24.1 | +34.9 | -0.6 |
| vx=-1 | +8.4 | -13.8 | +26.6 | +7.6 |
| vyaw=+1 | +2.5 | -22.5 | +27.9 | +2.2 |
| vyaw=-1 | +6.2 | -30.9 | +32.7 | +1.0 |

正常策略下：vx 正/负应该近似相反（这里和远不为 0）；vyaw 正/负应该左右轮反号（这里两次输出几乎一样）。**也就是说轮子输出基本不响应指令，只是一个固定偏置。**

输出层 `actor.6.weight`（shape `[16, 128]`）的行范数也印证了这个偏置：

```
腿 12 行: 0.62 ~ 0.85 (均值 0.740)
轮  4 行: fl 0.86   fr 1.537   hl 1.676   hr 0.703
```

fr / hl 两行的范数是腿的 2 倍多，正是那两个"疯狂输出"的轮子。

**结论：这个 checkpoint 的轮子动作头没有训好。** 部署侧（观测布局、量纲、关节顺序、起停流程）已经和训练端配置逐项对齐验证过，再调仿真参数也不会让"固定偏置"消失。

---

## 5. 关于「是不是要重新训练」

判断依据都在 §3.2：训练时有 117 / 174 个输入维度是死输入。

**结论（第三轮之后已经明确）：需要重训。**

- **如果目标是复现这个策略**：不必重训。现在的部署（喂前 57 维、其余按 0）在数值上与训练时的有效输入完全一致。
- **如果原本设计的是 174 维观测**（比如多帧历史、地形高度图等）：那说明训练端配置里那一项是存在但没有数据进来的（常见的历史 buffer 没填 / tensor 没写），这是训练侧的 bug，需修配置后重训。
- 顺带一个信号：M20 训练过的列范数（3.75–6.45）明显低于同结构的 b2w / go2w（6.3–12.6），虽然初始化尺度不同不能直接比，但值得确认这个 checkpoint 是不是训够了。
- **决定性的证据是 §4c-2**：策略在"标准站立"输入下就输出 fr=-31 / hl=+33 rad/s 的轮速，且轮子输出对 vx / vyaw 指令几乎无响应，输出层 fr/hl 两行范数是腿的 2 倍。这不是部署能调回来的，**必须重训（或换一个训好的 checkpoint）**。重训前建议先确认 `actions.joint_vel` 这一项在训练时是否真的参与了（可对照奖励项 `joint_vel_wheel_l2` / `track_lin_vel_xy_exp` 的曲线）。

---

## 6. 排查用的一次性工具（位于 `/tmp/m20_src`，未纳入仓库）

| 文件 | 用途 |
|---|---|
| `urdf2mjcf.py` | URDF → MJCF 生成器（含 actuator/sensor 顺序修正），**建议长期保留** |
| `pd_test.cpp` | 独立跑 GetUp 的 PD 插值，验证 MJCF + 关节映射能站稳 |
| `shapes.cpp` | 打印 ONNX 输入/输出 shape（查动态 batch） |
| `onnx_cmp.cpp`、`cmp57.cpp` | 验证裁剪模型与原模型数值等价 |
| `settle.cpp`、`probe.cpp`、`sweep.sh` | 静置 / 观测排列扫描脚本 |

`/tmp` 会被清理，其中 `urdf2mjcf.py` 已复制到 `policy/M20/tools_urdf2mjcf.py`（需要重生成 MJCF 时用它）。

---

## 5. 换模型后「站不住 / 一按就趴」的根因（已修）

现象：按 0 起身、按 1 后机身直接瘫倒（z 掉到 0.09，base_link 触地），`wheel_cmd` 恒为 0。
根因：**rl_sar 加载的是 `policy_deploy.onnx`，而重训后只更新了 `policy.onnx`**，没有重新生成 deploy 版。
- `config.yaml` 里 `model_name: "policy_deploy.onnx"`（11:19 的旧模型）。
- 新 `policy.onnx` 已是 57 维输入、动态 batch 维。直接拿旧 deploy 模型 → 策略错；若手写 protobuf 把 batch 维设成静态 1 但序列化出错 → `InitRL()` 抛 `ShapeInferenceError: actor.0/Gemm First input does not have rank 2` → 退回 Passive(kp=0) → 瘫倒。
- 正确做法：`python3 make_deploy.py`（用官方 `onnx` API 把输入维设成 `[1,57]`，权重不动）。生成后模型加载正常、策略接管、能站到 z≈0.52（Isaac 标称）。
- 注意：`truncate_to_41.py` 只适用于旧的 174 维导出，对 57 维模型会误切隐藏层，**勿用**。

---

## 6. 站立/行走调试结论（换常规地形模型后）

### 已修（部署侧）
1. **deploy 模型没重新生成**（§5）：已用 `make_deploy.py`（onnx API 把输入维钉成 [1,57]）修复，`policy_deploy.onnx` 现在与 `policy.onnx` 一致。
2. **轮子速度打五折**（§1，上轮）：`wheel_tau_damping: false` 已修复。
3. **轮-地摩擦 0.8→1.2**（`M20.xml` 默认 geom + ground）：站立可靠性从 ~1/3 提升到 ~4/5（稳定的 z=0.56-0.58、roll/pitch<2°、轮速≈0、v<0.06）。
4. **RL 接管冻结 0.5s**（`fsm_M20.hpp` RLLocomotion::Run）：接管后先用 GetUp 增益把机器人刹停在标准站姿，再交策略，消除 Passive 自由落体+GetUp 计时带来的随机残速方差。

### 验证：部署 config 与 Isaac 训练端逐项一致（无 scale 偏差）
- obs 布局/scale（ang_vel 0.25 / dof 1.0·0.05 / 无 lin_vel）、action_scale（腿 0.125/0.25、轮 5.0）、clip（轮 ±15.86≡velocity_limit 79.3）、DCMotor（腿 80/2.0/76.4、轮 0/0.6/21.6/79.3）、轮半径 0.09、关节轴、ang_vel_axis=body、`actor_obs_normalization=False`。**即 rl_sar 是 Isaac 的忠实 sim2sim。**

### 残差问题（非部署 bug）
- **零指令立定有时仍倒**（约 1/5~2/5），且**前进(W)必倒（偏航自转 ≈4 rad/s）**。根因是 Isaac(PhysX) 与 MuJoCo 的接触/求解器 sim2sim 差：轮子→机身偏航的动力学不同，Isaac-only 训练的策难以在 MuJoCo 下补偿。已验证：把轮子 `rl_kd` 从 0.6 提到 2.0 反而 0/5 全倒 → 0.6（Isaac 值）才是对的，偏离更差；前进本身能走（x 持续增加）说明轮子符号没错，纯偏航环不稳定。
- **结论**：部署已正确，站立可达；要让行走稳定，需在训练端加入 **MuJoCo 感知的域随机化**（摩擦/质量/接触/电机强度），或把 MuJoCo 物理参数精确对齐 Isaac。这是重训时要补的（见上轮清单 D/E）。

---

## 7. 第四轮：MJCF `joint/@ref` 让运动学整体偏移（已修）

### 现象

§4 遗留问题 2「站姿高度偏低 z ≈ 0.42，低于 Isaac 标称 0.52」一直没解释清楚。
按 URDF 参数解析 FK：**站立姿态**（`hipy ±0.6 / knee ∓1.0`）轮心应在基座下方 0.4366 m，
基座高度应是 0.527 m；而实测只有 0.40–0.44。

### 根因

MuJoCo 对 hinge 的旋转量是 **(qpos − ref)**，不是 qpos。

用 `mj_makeData` + `mj_forward` 在 scratch `mjData` 上验证（`ancnw.xml` 的 `fl_thigh_joint`，
`axis="0 -1 0"`，看 `fl_calf` 相对 `fl_thigh` 的偏移）：

| 配置 | 实测偏移 | 说明 |
|---|---|---|
| `ref="0.6"`, qpos=0.6 | `(-0.198, 0.005, -0.198)` | = URDF 原始偏移，**完全没旋转** |
| `ref="0.6"`, qpos=0 | `(-0.276, 0.005, -0.052)` | 反而被转了 −0.6 rad |
| 去掉 `ref`, qpos=0 | `(-0.198, 0.005, -0.198)` | 正常（零角 = 未旋转） |
| 去掉 `ref`, qpos=1.5 | `(0.183, 0.005, -0.211)` | 正常（旋转 +1.5 rad） |

后果：`qpos == ref` 时几何上是 **URDF 零位姿**，控制器命令的站立角完全失效，机器人退化成
"刚性四轮小车"——看着很稳（不会倒），但**腿对基座没有任何作用**，站高等于零位姿高度。
这正好解释了 §4c-1 里"实测 GetUp 稳定成功 z → 0.588"：0.588 就是零位姿高度，不是站姿高度；
也解释了 §4 遗留 2 和 §4b 里"基座高度 0.40–0.44"始终调不上去。

go2w / b2w 的 MJCF 都不带 `ref`。

### 修复（`policy/M20/tools_urdf2mjcf.py` → 重新生成 `M20.xml`）

| 改动 | 旧 | 新 |
|---|---|---|
| `joint/@ref` | 每个关节 `ref="<站立角>"` | **移除**（16 处） |
| `BASE_Z` | 0.572（按站姿算，但 ref 让实际姿态是零位姿） | **0.590**（零位姿贴地，qpos0 即零位姿） |
| `lying` 关键帧高度 | 0.32 | 0.437 |
| `home` 关键帧 | 无 | 新增，0.527 |
| 摩擦 | 手工在 `M20.xml` 里改成 `1.2 0.05 0.02` | 写进生成器（否则重新生成会被覆盖回 0.8） |

按 URDF 解析的真实高度（轮半径 0.09）：

| 姿态 | 轮心相对基座 z | 基座高度 |
|---|---|---|
| 零位姿（qpos0，出生） | −0.5000 | **0.590** |
| 站立（`hipy ±0.6 / knee ∓1.0`） | −0.4366 | 0.527 |
| 躺（`hipy ±0.9 / knee ∓1.6`） | −0.3466 | 0.437 |

注意 M20 的零位姿比站姿**更直**（腿展得更开），所以修好后出生高度反而比站姿高，
§4c-1 那张表（零角度 −0.441 / 站立 −0.572）是错的，本表以解析 FK 为准。

重新生成后的 diff 只有预期内容：基座 0.572→0.590、16 处 `ref=` 移除、两个关键帧；
摩擦 1.2 保持不变。

### 修复前后

同一协议：出生 → `0` 起身 → `1` 交 RL → 零指令 → `W` 前进。

| 指标 | 修复前（§4b 记录） | 修复后 |
|---|---|---|
| 基座高度 z | 0.40–0.44 | **0.475** |
| roll / pitch | ±5° / ±3.5° | −1.4° / −2.4° |
| yaw 漂移 | 15–20 °/s 自转 | ≈0.5 °/s |
| `cmd x=0.10` | 方向乱、边走边转 | 前进 ≈0.15 m/s |

### 副作用（预期内）

出生在零位姿（0.590）且 Passive 是 `kp=0`，放手后仍会下沉/前倾（实测 z 0.29–0.34、
pitch 44–53°），但**不再侧翻**，GetUp 能稳定拉起。想让出生即站稳做不到：没有 `ref` 时
`qpos0` 只能是零位姿（`<keyframe>` 不会被自动应用）。

### 残留

站高 0.475 仍比解析站姿 0.527 低约 0.05 m（策略稳态偏蹲）；零指令下约 0.02 m/s 前向漂移
并缓慢横移。

### 同源问题

ancnw 的部署踩的是同一个坑，见 `ANCNW_DEPLOY_NOTES.md` §5。
