/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 * 这是一个轮腿四足机器人的编排层,提供了「趴下 / 起身 / 下趴 / 原地抖动 / 原地踏步 / 
 * 原地踏步转圈 / RL 策略行走」七种模式,机械腿用位置环做运动学踏步,轮子用位置环兼当刹车与差速转向,
 * 所有模式可通过键盘或手柄实时切换
 */

// ============================================================================
//  关节顺序 (policy order, 共 16 DOF)
//    [0..11]  4 条腿 x 3 关节: fl(0 hip,1 thigh,2 calf) fr(3,4,5)
//             hl(6,7,8) hr(9,10,11)        前腿 leg<2, 后腿 leg>=2, 折叠方向相反
//    [12..15] 4 个轮子:      wheel[leg] = 12 + leg
//  ※ 这是策略/FSM 内部的 "policy 顺序", 与 URDF 实际关节顺序不同,
//    二者通过 base.yaml 的 joint_mapping 映射.
//
//  状态机生命周期(每个状态都重写 4 个虚函数):
//    Enter()       进入状态时调用一次(重置插值进度、记录起始姿态)
//    Run()         每个控制周期(dt)调用, 计算并写入 fsm_command(电机指令)
//    Exit()        离开状态时调用一次
//    CheckChange() 每个周期调用, 返回目标状态名; 返回自身 state_name_ = "停留"
//
//  Interpolate(percent, from, to, dur, ...) 工具:
//    推进 percent 并在"仍在插值中"时返回 true —— 此时 Run 应 return 跳过其余逻辑;
//    到达目标(percent>=1)时返回 false. 故常见写法: if (Interpolate(...)) return;
//
//  键位映射(键盘 / 手柄):
//    0 / A      -> GetUp      1 / RB_DPadUp   -> RLLocomotion
//    2 / RB_DPadRight -> Jitter   3 / RB_DPadLeft -> StepInPlace
//    4 / RB_DPadDown  -> StepTurn   9 / B -> GetDown   P / LB_X -> Passive
// ============================================================================

#ifndef ANCNW_FSM_HPP
#define ANCNW_FSM_HPP

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "fsm.hpp"
#include "rl_sdk.hpp"

//状态类 + 工具函数
namespace ancnw_fsm
{

// 摆动参数 s 每弧度对应的轮子抬升高度, 基于 URDF 在站姿附近做数值 IK 测得
// (s=0.25 -> 4.4 cm, s=0.35 -> 7.1 cm). 仅用于屏幕打印, 步态本身是纯运动学.
static constexpr float LIFT_M_PER_RAD = 0.203f;

// 轮子几何取自 URDF / 站姿正运动学: 轮半径 0.11 m, 半轮距(轮心 |y|) 0.22 m.
// 故机身以偏航率 w 做滑移转向时, 所需差速轮速 = w * HALF_TRACK / WHEEL_RADIUS = 2w.
static constexpr float WHEEL_RADIUS = 0.11f;
static constexpr float HALF_TRACK = 0.22f;

// 两个踏步状态共用的步态生成器. 按 policy 顺序填充 target[].
// 轮子指令为 wheel_ref + wheel_dir[leg] * wheel_spin, 因此:
//   wheel_dir = {0,0,0,0}                 -> 轮子锁住, 机身不会漂移
//   wheel_dir = {+1,-1,+1,-1} (左右) -> 差速驱动, 原地转向
// swing_leg / swing_peak 记录哪条(对)腿离地以及抬多高.
inline void BuildStepTargets(RL &rl,
                             float gait_time,
                             const std::vector<float> &wheel_ref,
                             const float wheel_dir[4],
                             float wheel_spin,
                             std::vector<float> &target,
                             int &swing_leg,
                             float &swing_peak,
                             bool &trot)
{
    const float period = rl.params.Get<float>("step_period", 1.0f);
    // 在前 step_ramp 秒内把摆动幅度平滑 ramp 进来. 从 RL 进入踏步时 gait_time 从 ~0 开始,
    // 否则腿会在第一帧就猛地抬到满幅, 把支撑腿拽走 -> 一按键机身就往前栽.
    const float ramp_time = rl.params.Get<float>("step_ramp", 1.0f);
    const float ramp = (ramp_time > 0.0f) ? std::min(1.0f, gait_time / ramp_time) : 1.0f;
    const float lift = rl.params.Get<float>("step_lift", 0.30f) * ramp;
    const float stance_drop = rl.params.Get<float>("step_stance_drop", 0.10f);
    const float calf_ratio = rl.params.Get<float>("step_calf_ratio", 2.0f);
    trot = rl.params.Get<bool>("step_trot", true);

    float phase = std::fmod(gait_time, period) / period;   // 取一个周期内的 0..1
    if (phase < 0.0f) phase += 1.0f;

    float offsets[4];
    float duty;
    const float duty_cfg = rl.params.Get<float>("step_duty", 0.0f);
    if (trot)
    {
        // 对角配对: fl+hr 与 fr+hl. 两条着地轮位于过质心的一条线上,
        // 因此离地的那对真正离地(已验证: 接触在 2 个对角轮 与 4 轮之间交替).
        offsets[0] = 0.00f; offsets[3] = 0.00f;
        offsets[1] = 0.50f; offsets[2] = 0.50f;
        duty = (duty_cfg > 0.0f) ? duty_cfg : 0.40f;
    }
    else
    {
        // 四拍爬行: 每次仅一条腿离地, 三条始终支撑
        offsets[2] = 0.00f;   // hl 左后
        offsets[0] = 0.25f;   // fl 左前
        offsets[3] = 0.50f;   // hr 右后
        offsets[1] = 0.75f;   // fr 右前
        duty = (duty_cfg > 0.0f) ? duty_cfg : 0.25f;
    }
    duty = std::min(duty, 0.49f);

    // 摆动窗口内 0 -> 1 -> 0 的平滑凸起
    auto swing = [&](int leg) -> float
    {
        float x = phase - offsets[leg];
        x -= std::floor(x);
        if (x >= duty) return 0.0f;
        return 0.5f * (1.0f - std::cos(2.0f * 3.14159265f * x / duty));
    };

    target = rl.params.Get<std::vector<float>>("default_dof_pos");
    const int thigh_idx[4] = {1, 4, 7, 10};   // fl, fr, hl, hr (policy 顺序)
    const int calf_idx[4] = {2, 5, 8, 11};
    swing_leg = -1;
    swing_peak = 0.0f;
    for (int leg = 0; leg < 4; ++leg)
    {
        float sw = swing(leg);
        // sw = 1 -> 腿离地(抬起); sw = 0 -> 腿着地并略微伸展,
        // 从而抬高机身, 给另一侧让出离地间隙
        float s = lift * sw - stance_drop * (1.0f - sw);
        float sign = (leg < 2) ? -1.0f : 1.0f;   // 前腿折叠方向相反
        target[thigh_idx[leg]] += sign * s;
        target[calf_idx[leg]] -= sign * calf_ratio * s;
        if (s > swing_peak)
        {
            swing_peak = s;
            swing_leg = leg;
        }
    }

    for (int leg = 0; leg < 4; ++leg)
    {
        const int w = 12 + leg;
        target[w] = ((size_t)w < wheel_ref.size() ? wheel_ref[w] : 0.0f) + wheel_dir[leg] * wheel_spin;
    }
}

// 把踏步目标写入电机指令: 腿用 fixed_kp/fixed_kd
// (当 step_leg_kp/step_leg_kd 都 > 0 时改用它们); 轮子用位置环
// (step_wheel_kp/kd), 目标锁住时相当于驻车制动.
inline void WriteStepCommand(RL &rl, RobotCommand<float> *cmd, const std::vector<float> &target)
{
    auto kp = rl.params.Get<std::vector<float>>("fixed_kp");
    auto kd = rl.params.Get<std::vector<float>>("fixed_kd");
    const float leg_kp = rl.params.Get<float>("step_leg_kp", 0.0f);
    const float leg_kd = rl.params.Get<float>("step_leg_kd", 0.0f);
    if (leg_kp > 0.0f && leg_kd > 0.0f)
    {
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            kp[i] = leg_kp;
            kd[i] = leg_kd;
        }
    }
    const float wheel_kp = rl.params.Get<float>("step_wheel_kp", 15.0f);
    const float wheel_kd = rl.params.Get<float>("step_wheel_kd", 2.0f);
    for (int w : rl.params.Get<std::vector<int>>("wheel_indices"))
    {
        kp[w] = wheel_kp;
        kd[w] = wheel_kd;
    }
    for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
    {
        cmd->motor_command.q[i] = target[i];
        cmd->motor_command.dq[i] = 0.0f;
        cmd->motor_command.kp[i] = kp[i];
        cmd->motor_command.kd[i] = kd[i];
        cmd->motor_command.tau[i] = 0.0f;
    }
}

class RLFSMStatePassive : public RLFSMState
{
public:
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    // 进入 Passive 时记录的轮子角度; 当 passive_wheel_kp > 0 时
    // 会把它锁住, 使机器人趴着时不会慢慢溜走
    std::vector<float> wheel_lock_pos;

    void Enter() override
    {
        wheel_lock_pos.clear();
        std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' (Keyboard) or 'A' (Gamepad) to switch to RLFSMStateGetUp." << std::endl;
    }

    void Run() override
    {
        // 用位置环保持趴下姿态(零位, 也是 qpos0 / GetDown 插值的回退姿态).
        // 原来只用阻尼(kp=0)时腿托不住机身: 会下沉并在启动后慢慢后倒.
        // 在 base.yaml 设 passive_hold: false 可恢复旧的纯阻尼行为.
        const bool hold = rl.params.Get<bool>("passive_hold", true);
        const float passive_kp = rl.params.Get<float>("passive_kp", 50.0f);
        const float passive_kd = rl.params.Get<float>("passive_kd", 2.0f);
        auto wheels = rl.params.Get<std::vector<int>>("wheel_indices");

        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            bool is_wheel = false;
            for (int w : wheels)
            {
                if (w == i) { is_wheel = true; break; }
            }
            if (is_wheel)
            {
                // 速度关节: 默认仅阻尼(kd 8). 即便如此机身仍会在 25 s 内
                // 悄悄溜走约 0.6 m, 因此可选的驻车制动锁住进入 Passive 时记录的轮角.
                const float wheel_kp = rl.params.Get<float>("passive_wheel_kp", 0.0f);
                if (wheel_kp > 0.0f)
                {
                    if (wheel_lock_pos.empty())
                    {
                        wheel_lock_pos = fsm_state->motor_state.q;
                    }
                    fsm_command->motor_command.q[i] = wheel_lock_pos[i];
                    fsm_command->motor_command.dq[i] = 0;
                    fsm_command->motor_command.kp[i] = wheel_kp;
                    fsm_command->motor_command.kd[i] = rl.params.Get<float>("passive_wheel_kd", 2.0f);
                    fsm_command->motor_command.tau[i] = 0;
                }
                else
                {
                    fsm_command->motor_command.dq[i] = 0;
                    fsm_command->motor_command.kp[i] = 0;
                    fsm_command->motor_command.kd[i] = 8;
                    fsm_command->motor_command.tau[i] = 0;
                }
            }
            else if (hold && passive_kp > 0.0f)
            {
                fsm_command->motor_command.q[i] = 0.0f;   // 零位 == 趴下姿态
                fsm_command->motor_command.dq[i] = 0;
                fsm_command->motor_command.kp[i] = passive_kp;
                fsm_command->motor_command.kd[i] = passive_kd;
                fsm_command->motor_command.tau[i] = 0;
            }
            else
            {
                fsm_command->motor_command.dq[i] = 0;
                fsm_command->motor_command.kp[i] = 0;
                fsm_command->motor_command.kd[i] = 8;
                fsm_command->motor_command.tau[i] = 0;
            }
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateGetUp : public RLFSMState
{
public:
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    float percent_pre_getup = 0.0f;
    float percent_getup = 0.0f;
    // 起身前的蜷曲姿态, POLICY 顺序:
    // 12 个腿关节 (fl_hip, fl_thigh, fl_calf, fr_..., hl_..., hr_...) + 4 个轮子
    std::vector<float> pre_running_pos = {
        0.00,  0.90, -1.60, 0.00,  0.90, -1.60,
        0.00, -0.90,  1.60, 0.00, -0.90,  1.60,
        0.00,  0.00,  0.00, 0.00
    };
    bool stand_from_passive = true;

    void Enter() override
    {
        percent_pre_getup = 0.0f;
        percent_getup = 0.0f;
        if (rl.fsm.previous_state_->GetStateName() == "RLFSMStatePassive")
        {
            stand_from_passive = true;
        }
        else
        {
            stand_from_passive = false;
        }
        rl.now_state = *fsm_state;
        rl.start_state = rl.now_state;
    }

    void Run() override
    {
        // stand_from_passive=true 表示刚从 Passive(趴姿/零位)起身: 必须先蜷到
        // pre_running_pos 再升到站姿(两段, 避免从趴姿直接拉到站姿扯坏腿);
        // 若来自 GetDown 等已站姿状态, 则直接插值到站姿(一段).
        if (stand_from_passive)
        {
            if (Interpolate(percent_pre_getup, rl.now_state.motor_state.q, pre_running_pos, 1.0f, "Pre Getting up", true)) return;
            if (Interpolate(percent_getup, pre_running_pos, rl.params.Get<std::vector<float>>("default_dof_pos"), 2.0f, "Getting up", true)) return;
        }
        else
        {
            if (Interpolate(percent_getup, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 1.0f, "Getting up", true)) return;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        if (percent_getup >= 1.0f)
        {
            if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
            {
                return "RLFSMStateRLLocomotion";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
            {
                return "RLFSMStateJitter";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
            {
                return "RLFSMStateStepInPlace";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num4 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
            {
                return "RLFSMStateStepTurn";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
            {
                return "RLFSMStateGetDown";
            }
        }
        return state_name_;
    }
};

// 原地抖动("原地抖动"): 保持站姿, 但在每个腿关节上叠加一个小幅高频正弦,
// 使整个机身颤动而不位移. 这与踏步是*不同*的模式: 轮子从不离地,
// 幅度约 1 cm, 频率为几 Hz.
//
// 振荡沿腿抬升轮子所用的同一方向施加(dthigh = -s, dcalf = +2s 对应前腿,
// 后腿镜像), 因此可见运动是机身上下颠动.
class RLFSMStateJitter : public RLFSMState
{
public:
    RLFSMStateJitter(RL *rl) : RLFSMState(*rl, "RLFSMStateJitter") {}

    float percent_transition = 0.0f;
    float jitter_time = 0.0f;
    std::vector<float> wheel_locked_pos;

    void Enter() override
    {
        percent_transition = 0.0f;
        jitter_time = 0.0f;
        rl.now_state = *fsm_state;
        wheel_locked_pos = rl.now_state.motor_state.q;
        std::cout << LOGGER::NOTE << "Entered jitter mode (shake in place). '3' step, '4' step+turn, '1' RL, '0' GetUp, '9' GetDown, 'P' passive." << std::endl;
    }

    void Run() override
    {
        if (Interpolate(percent_transition, rl.now_state.motor_state.q,
                        rl.params.Get<std::vector<float>>("default_dof_pos"),
                        rl.params.Get<float>("jitter_transition", 0.5f),
                        "Jitter", true))
        {
            return;
        }

        const float amp = rl.params.Get<float>("jitter_amp", 0.05f);          // 大腿上的弧度
        const float freq = rl.params.Get<float>("jitter_freq", 5.0f);         // Hz
        const float spread = rl.params.Get<float>("jitter_phase_spread", 0.0f); // 0 = 所有腿同相
        const float bias = rl.params.Get<float>("jitter_bias", 0.0f);         // >0: 只抬不压
        const float calf_ratio = rl.params.Get<float>("jitter_calf_ratio", 2.0f);

        jitter_time += rl.params.Get<float>("dt");
        const float omega = 2.0f * 3.14159265f * freq;

        auto target = rl.params.Get<std::vector<float>>("default_dof_pos");
        const int thigh_idx[4] = {1, 4, 7, 10};   // fl, fr, hl, hr (policy 顺序)
        const int calf_idx[4] = {2, 5, 8, 11};
        for (int leg = 0; leg < 4; ++leg)
        {
            float phase = omega * jitter_time + 2.0f * 3.14159265f * spread * static_cast<float>(leg);
            float s = amp * (bias + (1.0f - bias) * std::sin(phase));
            float sign = (leg < 2) ? -1.0f : 1.0f;
            target[thigh_idx[leg]] += sign * s;
            target[calf_idx[leg]] -= sign * calf_ratio * s;
        }

        auto kp = rl.params.Get<std::vector<float>>("fixed_kp");
        auto kd = rl.params.Get<std::vector<float>>("fixed_kd");
        const float leg_kp = rl.params.Get<float>("jitter_leg_kp", 0.0f);
        const float leg_kd = rl.params.Get<float>("jitter_leg_kd", 0.0f);
        if (leg_kp > 0.0f && leg_kd > 0.0f)
        {
            for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
            {
                kp[i] = leg_kp;
                kd[i] = leg_kd;
            }
        }

        const float wheel_kp = rl.params.Get<float>("step_wheel_kp", 15.0f);
        const float wheel_kd = rl.params.Get<float>("step_wheel_kd", 2.0f);
        for (int w : rl.params.Get<std::vector<int>>("wheel_indices"))
        {
            kp[w] = wheel_kp;
            kd[w] = wheel_kd;
            target[w] = wheel_locked_pos[w];
        }

        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            fsm_command->motor_command.q[i] = target[i];
            fsm_command->motor_command.dq[i] = 0.0f;
            fsm_command->motor_command.kp[i] = kp[i];
            fsm_command->motor_command.kd[i] = kd[i];
            fsm_command->motor_command.tau[i] = 0.0f;
        }

        std::ostringstream msg;
        msg << "Jitter in place amp:" << std::fixed << std::setprecision(2) << amp << " rad ("
            << std::setprecision(1) << (amp * LIFT_M_PER_RAD * 100.0f) << " cm) freq:" << freq << " Hz";
        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << msg.str() << std::flush;
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
        {
            return "RLFSMStateStepInPlace";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num4 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            return "RLFSMStateStepTurn";
        }
        return state_name_;
    }
};

// 原地踏步("原地踏步"): 腿轮流抬起再落下, 而机身原地不动. 轮子是速度关节,
// 因此被制动(kp=0, dq=0), 只有摆动腿这一处接触会离地.
//
// 抬升是纯竖直的轮子运动: 在站姿附近解腿 IK 得 前腿 dthigh = -s, dcalf = +2s,
// 后腿取镜像(每弧度 s 提供 0.203 m 离地间隙).
// 键盘 '3' = 原地踏步(轮子锁住, 不漂移);
// 键盘 '4' = 同样步态但机身绕自身轴滑移转向.
class RLFSMStateStepInPlace : public RLFSMState
{
public:
    RLFSMStateStepInPlace(RL *rl) : RLFSMState(*rl, "RLFSMStateStepInPlace") {}

    float percent_transition = 0.0f;
    float gait_time = 0.0f;
    std::vector<float> wheel_locked_pos;

    void Enter() override
    {
        percent_transition = 0.0f;
        gait_time = 0.0f;
        rl.now_state = *fsm_state;
        // 记住轮子当前角度, 以便锁在那里(= 驻车制动)
        wheel_locked_pos = rl.now_state.motor_state.q;
        std::cout << LOGGER::NOTE << "Entered step-in-place mode (legs lift, body stays). '4' step+turn, '2' jitter, '1' RL, '0' GetUp, '9' GetDown, 'P' passive." << std::endl;
    }

    void Run() override
    {
        // 先从上电进入时的姿态混合到站姿
        if (Interpolate(percent_transition, rl.now_state.motor_state.q,
                        rl.params.Get<std::vector<float>>("default_dof_pos"),
                        rl.params.Get<float>("step_transition", 0.5f),
                        "Step in place", true))
        {
            return;
        }

        gait_time += rl.params.Get<float>("dt");

        // 轮子锁在进该状态时的角度: 机身不会漂移
        const float hold[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        std::vector<float> target;
        int swing_leg = -1;
        float swing_peak = 0.0f;
        bool trot = true;
        BuildStepTargets(rl, gait_time, wheel_locked_pos, hold, 0.0f, target, swing_leg, swing_peak, trot);
        WriteStepCommand(rl, fsm_command, target);

        const float lift = rl.params.Get<float>("step_lift", 0.45f);
        const float stance_drop = rl.params.Get<float>("step_stance_drop", 0.10f);
        const float period = rl.params.Get<float>("step_period", 1.0f);

        static const char *leg_name[4] = {"fl", "fr", "hl", "hr"};
        std::string up_label = "-";
        if (swing_leg >= 0)
        {
            up_label = trot ? ((swing_leg == 0 || swing_leg == 3) ? "fl+hr" : "fr+hl")
                            : leg_name[swing_leg];
        }
        std::ostringstream msg;
        msg << "Step in place [" << (trot ? "trot" : "crawl") << "] clearance:" << std::fixed
            << std::setprecision(1) << ((lift + stance_drop) * LIFT_M_PER_RAD * 100.0f) << " cm  up:"
            << up_label << " (" << std::setprecision(1)
            << ((swing_peak + stance_drop) * LIFT_M_PER_RAD * 100.0f) << " cm now)  period:" << period << " s";
        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << msg.str() << std::flush;
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
        {
            return "RLFSMStateJitter";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num4 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            return "RLFSMStateStepTurn";
        }
        return state_name_;
    }
};

// 原地踏步同时原地转向("原地踏步转圈"), 键盘 '4'.
// 与 RLFSMStateStepInPlace 步态相同, 但不是锁轮子, 而是左右两侧互相对顶(滑移转向):
// 左轮 +w, 右轮 -w. 合力抵消, 偏航力矩不抵消, 因此机身绕自身轴旋转.
// 轮半径 0.11 m, 半轮距 0.22 m 时, 机身偏航率 w 需要轮速 w * 0.22 / 0.11 = 2w.
class RLFSMStateStepTurn : public RLFSMState
{
public:
    RLFSMStateStepTurn(RL *rl) : RLFSMState(*rl, "RLFSMStateStepTurn") {}

    float percent_transition = 0.0f;
    float gait_time = 0.0f;
    float wheel_spin = 0.0f;        // 累计轮子转角, 单位 rad
    std::vector<float> wheel_ref_pos;

    void Enter() override
    {
        percent_transition = 0.0f;
        gait_time = 0.0f;
        wheel_spin = 0.0f;
        rl.now_state = *fsm_state;
        wheel_ref_pos = rl.now_state.motor_state.q;
        std::cout << LOGGER::NOTE << "Entered step-turn mode (step in place while spinning). '3' step only, '2' jitter, '1' RL, '0' GetUp, '9' GetDown, 'P' passive." << std::endl;
    }

    void Run() override
    {
        if (Interpolate(percent_transition, rl.now_state.motor_state.q,
                        rl.params.Get<std::vector<float>>("default_dof_pos"),
                        rl.params.Get<float>("step_transition", 0.5f),
                        "Step turn", true))
        {
            return;
        }

        const float dt = rl.params.Get<float>("dt");
        gait_time += dt;

        const float yaw_rate = rl.params.Get<float>("step_yaw_rate", 0.6f);          // rad/s, + = 逆时针
        const float half_track = rl.params.Get<float>("step_turn_half_track", HALF_TRACK);
        const float radius = rl.params.Get<float>("step_turn_wheel_radius", WHEEL_RADIUS);
        float wheel_omega = rl.params.Get<float>("step_turn_wheel_omega", 0.0f);     // >0 时覆盖几何推算值
        if (wheel_omega == 0.0f)
        {
            wheel_omega = yaw_rate * half_track / radius;
        }
        wheel_spin += wheel_omega * dt;

        // 轮角为正时驱动轮子向后转, 因此左 + / 右 -
        // 表示左退右进 = 逆时针(yaw +)
        const float dir[4] = {+1.0f, -1.0f, +1.0f, -1.0f};   // fl, fr, hl, hr
        std::vector<float> target;
        int swing_leg = -1;
        float swing_peak = 0.0f;
        bool trot = true;
        BuildStepTargets(rl, gait_time, wheel_ref_pos, dir, wheel_spin, target, swing_leg, swing_peak, trot);
        WriteStepCommand(rl, fsm_command, target);

        const float lift = rl.params.Get<float>("step_lift", 0.45f);
        const float stance_drop = rl.params.Get<float>("step_stance_drop", 0.10f);
        std::string up_label = "-";
        if (swing_leg >= 0)
        {
            up_label = trot ? ((swing_leg == 0 || swing_leg == 3) ? "fl+hr" : "fr+hl") : "-";
        }
        std::ostringstream msg;
        msg << "Step turn [" << (trot ? "trot" : "crawl") << "] clearance:" << std::fixed
            << std::setprecision(1) << ((lift + stance_drop) * LIFT_M_PER_RAD * 100.0f) << " cm  up:"
            << up_label << "  yaw:" << std::setprecision(2) << yaw_rate << " rad/s (wheel "
            << wheel_omega << " rad/s)";
        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << msg.str() << std::flush;
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
        {
            return "RLFSMStateJitter";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
        {
            return "RLFSMStateStepInPlace";
        }
        return state_name_;
    }
};

class RLFSMStateGetDown : public RLFSMState
{
public:
    RLFSMStateGetDown(RL *rl) : RLFSMState(*rl, "RLFSMStateGetDown") {}

    float percent_getdown = 0.0f;

    void Enter() override
    {
        percent_getdown = 0.0f;
        rl.now_state = *fsm_state;
    }

    void Run() override
    {
        // 回到进入 GetUp 时记录的起始姿态 rl.start_state
        // (若当初从 Passive 起身, 则回到趴姿; 从其余站姿态进入则回到该姿态)
        Interpolate(percent_getdown, rl.now_state.motor_state.q, rl.start_state.motor_state.q, 2.0f, "Getting down", true);
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X || percent_getdown >= 1.0f)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateRLLocomotion : public RLFSMState
{
public:
    RLFSMStateRLLocomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRLLocomotion") {}

    float percent_transition = 0.0f;

    void Enter() override
    {
        percent_transition = 0.0f;
        rl.episode_length_buf = 0;

        // 从 yaml 读取参数
        rl.config_name = "robot_lab";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // 故意不做起身->站姿的位置过渡: 进入 RL 后由策略直接输出力矩,
        // 若先做位置插值会与策略首帧指令"打架"造成起手抖动, 故直接进入 RLControl().
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::flush;
        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
        {
            return "RLFSMStateJitter";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
        {
            return "RLFSMStateStepInPlace";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num4 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            return "RLFSMStateStepTurn";
        }
        return state_name_;
    }
};

} // namespace ancnw_fsm

//状态工厂, 把状态名映射到具体的状态类
class AncnwFSMFactory : public FSMFactory
{
public:
    AncnwFSMFactory(const std::string& initial) : initial_state_(initial) {}
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<ancnw_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<ancnw_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<ancnw_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateJitter")
            return std::make_shared<ancnw_fsm::RLFSMStateJitter>(rl);
        else if (state_name == "RLFSMStateStepInPlace")
            return std::make_shared<ancnw_fsm::RLFSMStateStepInPlace>(rl);
        else if (state_name == "RLFSMStateStepTurn")
            return std::make_shared<ancnw_fsm::RLFSMStateStepTurn>(rl);
        else if (state_name == "RLFSMStateRLLocomotion")
            return std::make_shared<ancnw_fsm::RLFSMStateRLLocomotion>(rl);
        return nullptr;
    }
    std::string GetType() const override { return "ancnw"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",        // 趴下待机:位置环保持趴姿,轮子可选驻车制动,等按 0/A 起身
            "RLFSMStateGetUp",          // 起身:先蜷到 pre_running_pos 再升到站姿(两段插值),起身完成后原地保持站立
            "RLFSMStateGetDown",        // 下趴:从当前姿态插值回到进入 GetUp 时记录的起始姿态,完成后自动转 Passive
            "RLFSMStateJitter",         // 原地抖动:保持站姿,腿关节叠加小幅高频正弦,整机原地颤动、轮子不离地
            "RLFSMStateStepInPlace",    // 原地踏步:腿轮流抬起落下,轮子锁住防漂移,机身原地不动
            "RLFSMStateStepTurn",       // 原地踏步转圈:同上步态,左右轮差速对顶(滑移转向),边踏步边原地旋转
            "RLFSMStateRLLocomotion"    // RL 策略行走:加载 policy,由神经网络输出力矩,支持 WASD 速度指令
        };
    }
    std::string GetInitialState() const override { return initial_state_; }
private:
    std::string initial_state_;
};

//注册到全局 FSM 工厂表
REGISTER_FSM_FACTORY(AncnwFSMFactory, "RLFSMStatePassive")

#endif // ANCNW_FSM_HPP
