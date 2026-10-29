/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

// ============================================================================
//  rl_sdk —— 所有机器人共用的 RL 控制内核(与具体硬件无关)
//
//  · 本文件定义基类 RL: 加载策略模型 / 组装观测 / 推理 / 解码动作 / 输出电机指令。
//    这套流程对真机与仿真完全一致, 因此各机型共享同一份实现。
//  · 各机型(ancnw / go2 / g1 ...)的差异, 由 rl_real_*.cpp、rl_sim_*.cpp 继承 RL
//    后重写纯虚函数实现:
//       Forward()     跑模型推理(进程内调用, 不是跨进程/跨板通信)
//       GetState()    从总线/传感器读取真实状态
//       SetCommand()  把电机指令下发到总线
//  · 部署拓扑: RL 模型与 rl_sdk 必须位于同一块机载主板(同一进程);
//    电机/驱动器侧的 MCU 只做高频内环伺服, 不运行策略。
//  · 两份配置(均从编译期宏 POLICY_DIR 指向的绝对路径加载):
//       policy/<robot>/base.yaml             FSM 增益、几何、关节映射
//       policy/<robot>/<cfg>/config.yaml     观测/动作缩放、增益、模型名(InitRL 读取)
// ============================================================================

#ifndef RL_SDK_HPP
#define RL_SDK_HPP

#include <iostream>
#include <string>
#include <exception>
#include <unistd.h>
#include <algorithm>
#include <tbb/concurrent_queue.h>
#include <vector>
#include <memory>
#include <fstream>
#include <mutex>

#include <yaml-cpp/yaml.h>
#include "fsm.hpp"
#include "observation_buffer.hpp"
#include "vector_math.hpp"
#include "inference_runtime.hpp"
#include "logger.hpp"
#include "motion_loader.hpp"

// 下发给机器人的电机指令(每个控制周期一份)
template <typename T>
struct RobotCommand
{
    // 每个关节一套 "PD + 前馈": 目标位置 / 目标速度 / 前馈力矩 / 刚度 / 阻尼
    // ※ 所有数组按 "policy 顺序" 排列, 由 SetCommand() 经 joint_mapping 换算到实际关节序
    struct MotorCommand
    {
        std::vector<int> mode;   // 电机模式字(多数机型未使用, 保留扩展)
        std::vector<T> q;        // 目标位置 (rad)
        std::vector<T> dq;       // 目标速度 (rad/s) —— 轮式关节走速度环
        std::vector<T> tau;      // 前馈力矩 (N*m)
        std::vector<T> kp;       // 位置刚度
        std::vector<T> kd;       // 阻尼

        void resize(size_t num_joints)
        {
            mode.resize(num_joints, 0);
            q.resize(num_joints, 0.0f);
            dq.resize(num_joints, 0.0f);
            tau.resize(num_joints, 0.0f);
            kp.resize(num_joints, 0.0f);
            kd.resize(num_joints, 0.0f);
        }
    } motor_command;
};

// 从机器人读回的状态(由各机型的 GetState() 填充)
template <typename T>
struct RobotState
{
    // 机体 IMU: 姿态 + 角速度 + 线加速度
    struct IMU
    {
        std::vector<T> quaternion = {1.0f, 0.0f, 0.0f, 0.0f}; // 姿态四元数, 顺序 w,x,y,z(标量在前);{1,0,0,0}=水平无旋转
        std::vector<T> gyroscope = {0.0f, 0.0f, 0.0f};        // 角速度 (rad/s), 机体坐标系
        std::vector<T> accelerometer = {0.0f, 0.0f, 0.0f};    // 线加速度 (m/s^2)
    } imu;

    // 关节状态(按 policy 顺序)
    struct MotorState
    {
        std::vector<T> q;        // 关节位置 (rad)
        std::vector<T> dq;       // 关节速度 (rad/s)
        std::vector<T> ddq;      // 关节加速度(部分机型回填, 常为空)
        std::vector<T> tau_est;  // 估计力矩 (N*m), 用于 CSV 日志与力矩保护
        std::vector<T> cur;      // 电流(部分机型回填)

        void resize(size_t num_joints)
        {
            q.resize(num_joints, 0.0f);
            dq.resize(num_joints, 0.0f);
            ddq.resize(num_joints, 0.0f);
            tau_est.resize(num_joints, 0.0f);
            cur.resize(num_joints, 0.0f);
        }
    } motor_state;
};

// 输入设备枚举。实际"键 -> 状态"的绑定在各机型的 fsm_*.hpp 的 CheckChange() 里完成,
// 这里的注释只是推荐约定, 不是强制映射。
namespace Input
{
    // 推荐: Num0-起身 Num9-下趴 N-切换导航模式
    //       R-仿真复位 Enter-仿真开关
    //       M-电机使能 K-电机失能 P-电机趴下
    //       Num1-基础行走 Num2~Num8-技能(7 个)
    //       WS-X轴(前后) AD-Y轴(左右) QE-偏航 Space-清零
    enum class Keyboard
    {
        None = 0,
        A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        Space, Enter, Escape,
        Up, Down, Left, Right
    };

    // 推荐: A-起身 B-下趴 X-切换导航模式 Y-无
    //       RB_Y-仿真复位 RB_X-仿真开关
    //       LB_A-电机使能 LB_B-电机失能 LB_X-电机趴下
    //       RB_DPadUp-基础行走 RB_DPadOthers/LB_DPadOthers-技能(7 个)
    //       LY-X轴(前后) LX-Y轴(左右) RX-偏航
    // 注: LB_*/RB_* 表示"按住肩键 + 某键"的组合键
    enum class Gamepad
    {
        None = 0,
        A, B, X, Y, LB, RB, LStick, RStick, DPadUp, DPadDown, DPadLeft, DPadRight,
        LB_A, LB_B, LB_X, LB_Y, LB_LStick, LB_RStick, LB_DPadUp, LB_DPadDown, LB_DPadLeft, LB_DPadRight,
        RB_A, RB_B, RB_X, RB_Y, RB_LStick, RB_RStick, RB_DPadUp, RB_DPadDown, RB_DPadLeft, RB_DPadRight,
        LB_RB
    };
}

// 用户输入状态: 按键/手柄 + 期望速度指令
struct Control
{
    // 当前键与上一次的键: SetKeyboard 只在"键发生变化"时才更新 last,
    // 因此 FSM 可用 "current != last" 判定这是一次新的按键事件(边沿触发)
    Input::Keyboard current_keyboard = Input::Keyboard::None, last_keyboard = Input::Keyboard::None;
    Input::Gamepad current_gamepad = Input::Gamepad::None, last_gamepad = Input::Gamepad::None;

    // 期望速度指令(会作为 commands 观测喂给策略)
    float x = 0.0f;    // 前后: + 向前
    float y = 0.0f;    // 左右: + 向左
    float yaw = 0.0f;  // 偏航角速度: + 逆时针
    bool navigation_mode = false;   // 导航模式开关(N 键 / 手柄 X)

    void SetKeyboard(Input::Keyboard keyboad)
    {
        if (current_keyboard != keyboad)
        {
            last_keyboard = current_keyboard;
            current_keyboard = keyboad;
        }
    }

    void SetGamepad(Input::Gamepad gamepad)
    {
        if (current_gamepad != gamepad)
        {
            last_gamepad = current_gamepad;
            current_gamepad = gamepad;
        }
    }

    // 每个控制周期末调用: 把 current 归位, 使下一次 SetKeyboard 的"变化"才被识别为新事件;
    // 手柄直接置 None(手柄事件天然是脉冲式的, 不做边沿保持)
    void ClearInput()
    {
        current_keyboard = last_keyboard;
        current_gamepad = Input::Gamepad::None;
    }
};

// YAML 配置读取器: 把 config.yaml / base.yaml 的键值挂到 config_node 上按需取用
struct YamlParams
{
    YAML::Node config_node;

    // 按键取值; 键不存在时返回 default_value
    // 警告: 对 vector 等容器, 必须先存到变量再用迭代器/引用:
    //   ✓ auto vec = params.Get<std::vector<int>>("key"); vec.begin()
    //   ✗ params.Get<std::vector<int>>("key").begin()  // 悬垂引用!
    template<typename T>
    T Get(const std::string& key, const T& default_value = T()) const
    {
        if (config_node[key])
        {
            return config_node[key].as<T>();
        }
        return default_value;
    }

    bool Has(const std::string& key) const
    {
        return config_node[key].IsDefined();
    }
};

// 观测量的原始容器: 由各机型的 GetState()/RunModel() 填充原始值,
// 再由 ComputeObservation() 按 config.yaml 的 observations 列表挑选、缩放后拼成网络输入。
// 注意: 这里的量是"原始物理量", 尚未乘 *_scale, 也不是最终喂给网络的向量。
template <typename T>
struct Observations
{
    std::vector<T> lin_vel;       // 线速度 (m/s),  3
    std::vector<T> ang_vel;       // 角速度 (rad/s), 3
    std::vector<T> gravity_vec;   // 重力方向向量,   3(机体系, 常为 {0,0,-1})
    std::vector<T> commands;      // 速度指令 [x, y, yaw], 3
    std::vector<T> base_quat;     // 基座四元数 w,x,y,z, 4
    std::vector<T> dof_pos;       // 关节位置 (rad), num_of_dofs
    std::vector<T> dof_vel;       // 关节速度 (rad/s), num_of_dofs
    std::vector<T> actions;       // 上一帧动作(网络的原始输出), num_of_dofs
};

class RL
{
public:
    RL() {};
    ~RL() {};

    YamlParams params;                 // config.yaml 的参数读取器
    Observations<float> obs;           // 观测原始量容器
    std::vector<int> obs_dims;         // 每个观测块的维度(调试/历史缓冲用)

    RobotState<float> robot_state;     // 当前机器人状态(GetState 填充)
    RobotCommand<float> robot_command; // 当前电机指令(FSM 写入, SetCommand 下发)

    // 模型输出 -> 状态机的无锁桥接队列(生产者: 推理线程 RunModel/Forward; 消费者: FSM 的 RLControl)
    // 之所以用队列而不是直接赋值, 是因为推理与 FSM 分处不同线程, 且频率可能不一致
    tbb::concurrent_queue<std::vector<float>> output_dof_pos_queue;
    tbb::concurrent_queue<std::vector<float>> output_dof_vel_queue;
    tbb::concurrent_queue<std::vector<float>> output_dof_tau_queue;

    FSM fsm;                             // 有限状态机(Passive / GetUp / RL / ...)
    RobotState<float> start_state;       // 进入某状态时记录的起始状态(GetDown 靠它回退)
    RobotState<float> now_state;         // FSM 当前使用的状态快照
    bool rl_init_done = false;           // RL 初始化完成标志: 未完成时 RunModel 不跑推理

    // 某些状态(如踏步/抖动)会在策略目标之上额外叠加一段运动。
    // 这里记录该叠加量, 在送进策略前从关节位置/速度观测中减掉,
    // 使策略"看不见"这段附加运动, 也就不会与之对抗。为空 = 不隐藏。
    std::vector<float> obs_pos_bias;
    std::vector<float> obs_vel_bias;

    // ===== 初始化 =====
    void InitObservations();                     // 观测清零, 并按 default_* 填初值
    void InitOutputs();                          // 输出缓冲清零, 目标位置置为 default_dof_pos
    void InitControl();                          // 速度指令 x/y/yaw 清零
    void InitRL(std::string robot_config_path);  // 总入口: 读 config.yaml + 建观测 + 加载模型
    void InitJointNum(size_t num_joints);        // 按 num_of_dofs 给各状态/指令数组定长

    // ===== RL 主流程(通用实现) =====
    // Forward(): 纯虚 —— 由各机型实现"调用模型"这一句(model->forward / model->infer)
    virtual std::vector<float> Forward() = 0;
    // ComputeObservation(): 按 config.yaml 的 observations 列表拼出网络输入向量(含缩放与 clip_obs)
    std::vector<float> ComputeObservation();
    // GetState()/SetCommand(): 纯虚 —— 硬件收发, 由 rl_real_*.cpp / rl_sim_*.cpp 实现
    virtual void GetState(RobotState<float> *state) = 0;
    virtual void SetCommand(const RobotCommand<float> *command) = 0;
    // StateController(): 把 state/command 注入各 FSM 状态, 驱动 fsm.Run(), 再处理键盘速度指令
    void StateController(const RobotState<float> *state, RobotCommand<float> *command);
    // ComputeOutput(): 把网络输出的 action 解码成 目标位置/速度/力矩(乘 action_scale + PD)
    void ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau);

    // ===== 配置读取 =====
    // 从 POLICY_DIR/<file_path>/<file_name> 加载 YAML, 并把顶层键值挂到 params
    void ReadYaml(const std::string& file_path, const std::string& file_name);

    // ===== CSV 离线日志(编译期开 CSV_LOGGER 才有调用) =====
    // 记录 计算力矩 / 估计力矩 / 实际位置 / 目标位置 / 速度, 用于离线分析跟踪效果
    std::string csv_filename;
    void CSVInit(std::string robot_name);
    void CSVLogger(const std::vector<float> &torque, const std::vector<float> &tau_est, const std::vector<float> &joint_pos, const std::vector<float> &joint_pos_target, const std::vector<float> &joint_vel);

    // ===== 输入 =====
    Control control;                  // 按键/手柄 + 期望速度
    void KeyboardInterface();         // 非阻塞读键盘, 把字符映射成 Input::Keyboard

    // ===== 观测历史缓冲 =====
    // 当 config.yaml 的 observations_history 非空时启用: 堆叠多帧观测作为网络输入
    ObservationBuffer history_obs_buf;
    std::vector<float> history_obs;

    // ===== 其它 =====
    int motiontime = 0;                        // 主循环计数
    std::string robot_name, config_name;       // 机器人名与配置名(决定模型与配置路径)
    bool simulation_running = true;            // 仿真运行标志
    std::string ang_vel_axis = "body";         // 角速度所在坐标系: "body" 机体系 / "world" 世界系
                                               // ※ 真机与 Isaac 训练均为 body, 不可随意改
    unsigned long long episode_length_buf = 0; // 当前回合已跑步数(用于 phase 类观测)
    float motion_length = 0.0;                 // 动作总时长(mimic/dance 任务用)
    int InverseJointMapping(int idx) const;    // 实际关节序 -> policy 序 的反查

    // 动作/模仿任务(mimic、dance)的动作加载器
    std::unique_ptr<MotionLoader> motion_loader;

    // ===== 保护函数 =====
    void TorqueProtect(const std::vector<float> &origin_output_dof_tau);                       // 力矩超限时告警
    void AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold);  // 俯仰/横滚超限切 Passive

    // ===== RL 模型 =====
    // 进程内加载的推理引擎(policy.pt / policy.onnx), 由 InitRL 的 load_model 创建;
    // Forward() 中的 model->forward(...) 就是"真机与 RL 模型"的唯一交接点
    std::unique_ptr<InferenceRuntime::Model> model;

    // 最近一次模型解码输出(力矩 / 目标位置 / 目标速度)
    std::vector<float> output_dof_tau;
    std::vector<float> output_dof_pos;
    std::vector<float> output_dof_vel;

    // 保护 model 与推理过程, 避免热重载模型时与推理线程竞争
    std::mutex model_mutex;
};

// RL 版 FSM 状态基类: 各机型的 fsm_*.hpp 里所有状态都继承它,
// 从而获得"读机器人状态"和"写电机指令"的能力。
class RLFSMState : public FSMState
{
public:
    RLFSMState(RL& rl, const std::string& name)
        : FSMState(name), rl(rl), fsm_state(nullptr), fsm_command(nullptr) {}

    RL& rl;                                  // 反向引用宿主 RL 对象(取参数、取状态、取模型输出)
    const RobotState<float> *fsm_state;      // 当前机器人状态(由 StateController 每周期注入)
    RobotCommand<float> *fsm_command;        // 本状态要写的电机指令(由 StateController 每周期注入)

    // 位置插值工具: 把关节从 start_pos 平滑过渡到 target_pos
    //   percent        进度 [0,1], 由调用方持有(状态成员), 逐帧递增
    //   duration_seconds  过渡时长(秒), 内部按 dt 折算成帧数
    //   use_fixed_gains   true 用 fixed_kp/fixed_kd(FSM 动作), false 用 rl_kp/rl_kd(策略增益)
    //   返回值: true  = 仍在过渡中(调用方应 return, 跳过本帧其余逻辑)
    //           false = 已到达目标(percent>=1), 可继续执行后续逻辑
    bool Interpolate(
        float& percent,
        const std::vector<float>& start_pos,
        const std::vector<float>& target_pos,
        float duration_seconds,
        const std::string& description = "",
        bool use_fixed_gains = true
    );

    // 从 output_*_queue 取出模型输出(位置/速度), 配上 rl_kp/rl_kd 写进 fsm_command
    // 即 RL 状态下"把策略输出变成电机指令"的最后一步
    void RLControl();
};

#endif // RL_SDK_HPP
