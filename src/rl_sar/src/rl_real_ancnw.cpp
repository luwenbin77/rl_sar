/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * 自研 "ancnw" 轮腿四足机器人（RK3588 + CAN-FD）真机入口程序。
 *
 * ===== 仍需完成 =====
 *   ReadIMU()  —— 接入真实 IMU（重力向量 + 角速度）。策略依赖这两项，
 *                 当前占位返回单位四元数与零角速度，真机上无法正常工作。
 *   base.yaml 中的 motor_p_max / v_max / t_max / kp_max / kd_max 必须与
 *   实际模组型号一致（默认值按 P55 模组）。
 *   can_ids / can_interfaces 的接线必须与真实硬件拓扑一致（每条总线 <=6 个电机，
 *   且同一总线上 CAN-ID 唯一）。
 */
#include "rl_real_ancnw.hpp"

#include <cstdint>
#include <cstring>
#include <cctype>
#include <csignal>
#include <iostream>
#include <unistd.h>

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------
// 把数值限制在 [lo, hi] 区间
static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// 按逗号分割字符串（忽略空白），用于解析 can_interfaces
static std::vector<std::string> SplitComma(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else if (!std::isspace((unsigned char)c)) cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// 编码单个关节的 10 字节 MIT 指令槽位（小端，0x8000 表示 0）
static void PackJoint(uint8_t* b, float q, float dq, float kp, float kd, float tau,
                      float p_max, float v_max, float t_max, float kp_max, float kd_max) {
    int32_t q_raw   = (int32_t)clampf((q   / p_max) * 32768.0f + 32768.0f, 0.0f, 65535.0f);
    int32_t dq_raw  = (int32_t)clampf((dq  / v_max) * 32768.0f + 32768.0f, 0.0f, 65535.0f);
    int32_t kp_raw  = (int32_t)clampf((kp  / kp_max) * 65535.0f,           0.0f, 65535.0f);
    int32_t kd_raw  = (int32_t)clampf((kd  / kd_max) * 65535.0f,           0.0f, 65535.0f);
    int32_t tau_raw = (int32_t)clampf((tau / t_max) * 32768.0f + 32768.0f, 0.0f, 65535.0f);
    auto put = [&](int o, int32_t v) { b[o] = (uint8_t)(v & 0xFF); b[o + 1] = (uint8_t)((v >> 8) & 0xFF); };
    put(0, q_raw); put(2, dq_raw); put(4, kp_raw); put(6, kd_raw); put(8, tau_raw);
}

// ---------------------------------------------------------------------------
// 信号处理：Ctrl+C 退出
// ---------------------------------------------------------------------------
volatile sig_atomic_t g_shutdown_requested = 0;
void signalHandler(int /*signum*/) {
    std::cout << LOGGER::INFO << "收到退出信号，正在关闭..." << std::endl;
    g_shutdown_requested = 1;
}

RL_Real::RL_Real(int argc, char **argv) {
    // 至少需要一个 CAN 接口参数；也可在 base.yaml 的 can_interfaces 中指定多条总线
    if (argc < 2) {
        std::cout << LOGGER::ERROR << "用法: " << argv[0]
                  << " <can接口> [can接口2 ...]  （或在 base.yaml 中设置 can_interfaces）"
                  << std::endl;
        throw std::runtime_error("参数错误");
    }
    std::string default_if = argv[1];

    // 读取参数并加载对应的 FSM
    this->robot_name = "ancnw";
    this->ReadYaml(this->robot_name, "base.yaml");

    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name)) {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr) this->fsm = *fsm_ptr;
    } else {
        std::cout << LOGGER::ERROR << "[FSM] 未注册机器人: " << this->robot_name << std::endl;
    }

    // 读取自由度、映射、CAN-ID 与电机极限
    this->num_dofs_ = this->params.Get<int>("num_of_dofs");
    this->joint_mapping_ = this->params.Get<std::vector<int>>("joint_mapping");
    this->motor_canid_ = this->params.Get<std::vector<int>>("can_ids");
    this->p_max_  = this->params.Get<float>("motor_p_max");
    this->v_max_  = this->params.Get<float>("motor_v_max");
    this->t_max_  = this->params.Get<float>("motor_t_max");
    this->kp_max_ = this->params.Get<float>("motor_kp_max");
    this->kd_max_ = this->params.Get<float>("motor_kd_max");

    // 校验 can_ids 长度
    if ((int)this->motor_canid_.size() != this->num_dofs_) {
        std::cout << LOGGER::ERROR << "can_ids 长度 (" << this->motor_canid_.size()
                  << ") 与 num_of_dofs (" << this->num_dofs_ << ") 不一致" << std::endl;
        throw std::runtime_error("can_ids 错误");
    }

    // 分配总线：优先用 base.yaml 的 can_interfaces，否则全部挂到默认接口
    std::string ifstr = this->params.Get<std::string>("can_interfaces");
    auto ifs = SplitComma(ifstr);
    if (ifs.empty()) {
        ifs = { default_if };
    }
    if (ifs.size() == 1) {
        this->motor_if_.assign(this->num_dofs_, ifs[0]);
    } else if ((int)ifs.size() == this->num_dofs_) {
        this->motor_if_ = ifs;
    } else {
        std::cout << LOGGER::ERROR << "can_interfaces 有 " << ifs.size()
                  << " 项，应为 1 或 " << this->num_dofs_ << std::endl;
        throw std::runtime_error("can_interfaces 错误");
    }

    // 建立 (接口, CAN-ID) -> 物理电机 的反查表，并为每条总线创建通信对象
    for (int p = 0; p < this->num_dofs_; ++p) {
        int c = this->motor_canid_[p];
        if (c < 1 || c > 6) {
            std::cout << LOGGER::ERROR << "电机 " << p << " 的 can_id " << c << " 超出 1..6 范围" << std::endl;
            throw std::runtime_error("can_id 错误");
        }
        auto key = std::make_pair(this->motor_if_[p], c);
        if (this->bus_to_phys_.count(key)) {
            std::cout << LOGGER::ERROR << "总线上出现重复的 (总线, can_id) " << this->motor_if_[p]
                      << ":" << c << std::endl;
            throw std::runtime_error("同一总线上 can_id 重复");
        }
        this->bus_to_phys_[key] = p;
        if (this->cans_.find(this->motor_if_[p]) == this->cans_.end()) {
            this->cans_[this->motor_if_[p]] = std::make_unique<CanFdComm>(this->motor_if_[p], true);
        }
    }
    // 打开所有总线
    for (auto& kv : this->cans_) {
        if (!kv.second->open()) {
            std::cout << LOGGER::ERROR << "[CAN] 打开 " << kv.first << " 失败" << std::endl;
            throw std::runtime_error("CAN 打开失败");
        }
    }

    // 初始化电机状态缓冲
    this->motor_q_.assign(this->num_dofs_, 0.0f);
    this->motor_dq_.assign(this->num_dofs_, 0.0f);
    this->motor_tau_.assign(this->num_dofs_, 0.0f);

    this->InitJointNum(this->num_dofs_);
    this->InitOutputs();
    this->InitControl();

    // 每条总线启动一个接收线程
    this->running_ = true;
    for (auto& kv : this->cans_) {
        this->recv_threads_.emplace_back(&RL_Real::RecvThreadFunc, this, kv.first);
    }

    // 使能所有关节（Passive 状态用 kp=0 -> 阻尼，可安全保持）
    this->SendEnable(true);

    // 启动三个控制循环
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Real::RunModel, this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();
}

RL_Real::~RL_Real() {
    this->running_ = false;
    // 等待接收线程结束
    for (auto& t : this->recv_threads_) if (t.joinable()) t.join();
    this->SendEnable(false);   // 退出时失能电机
    if (this->loop_keyboard) this->loop_keyboard->shutdown();
    if (this->loop_control) this->loop_control->shutdown();
    if (this->loop_rl) this->loop_rl->shutdown();
    std::cout << LOGGER::INFO << "RL_Real 退出" << std::endl;
}

// 单条总线的状态接收线程
void RL_Real::RecvThreadFunc(const std::string& ifname) {
    struct canfd_frame frame;
    while (this->running_) {
        auto it = this->cans_.find(ifname);
        if (it == this->cans_.end()) break;
        if (it->second->recv(frame, 1)) {
            int canid = frame.can_id & 0x7FF;            // 关节 CAN-ID（1..6）
            if (canid < 1 || canid > 6) continue;
            auto pit = this->bus_to_phys_.find({ifname, canid});
            if (pit == this->bus_to_phys_.end()) continue;
            int p = pit->second;
            // 解析回报帧中的位置/速度/力矩（小端，0x8000 表示 0）
            int16_t q_raw   = (int16_t)(frame.data[2] | (frame.data[3] << 8));
            int16_t dq_raw  = (int16_t)(frame.data[4] | (frame.data[5] << 8));
            int16_t tau_raw = (int16_t)(frame.data[6] | (frame.data[7] << 8));
            float q   = (q_raw   - 32768) * this->p_max_ / 32768.0f;
            float dq  = (dq_raw  - 32768) * this->v_max_ / 32768.0f;
            float tau = (tau_raw - 32768) * this->t_max_ / 32768.0f;
            {
                std::lock_guard<std::mutex> lk(this->state_mutex_);
                this->motor_q_[p] = q;
                this->motor_dq_[p] = dq;
                this->motor_tau_[p] = tau;
            }
        }
    }
}

// 读取机器人状态：把物理电机状态按 joint_mapping 填回策略顺序，并读取 IMU
void RL_Real::GetState(RobotState<float> *state) {
    std::lock_guard<std::mutex> lk(this->state_mutex_);
    for (int i = 0; i < this->num_dofs_; ++i) {
        int p = this->joint_mapping_[i];   // 策略索引 -> 物理索引
        state->motor_state.q[i] = this->motor_q_[p];
        state->motor_state.dq[i] = this->motor_dq_[p];
        state->motor_state.tau_est[i] = this->motor_tau_[p];
    }
    this->ReadIMU(state);
}

// 下发命令
void RL_Real::SetCommand(const RobotCommand<float> *command) {
    this->PackAndSend(command);
}

// MIT 模式打包并发送：按总线分组，每条总线一帧（CAN-ID 0x000）
void RL_Real::PackAndSend(const RobotCommand<float> *command) {
    // 把策略顺序的命令打散到物理电机顺序的数组
    std::vector<float> q_phys(this->num_dofs_, 0.0f), dq_phys(this->num_dofs_, 0.0f);
    std::vector<float> kp_phys(this->num_dofs_, 0.0f), kd_phys(this->num_dofs_, 0.0f);
    std::vector<float> tau_phys(this->num_dofs_, 0.0f);
    for (int i = 0; i < this->num_dofs_; ++i) {
        int p = this->joint_mapping_[i];
        q_phys[p]   = command->motor_command.q[i];
        dq_phys[p]  = command->motor_command.dq[i];
        kp_phys[p]  = command->motor_command.kp[i];
        kd_phys[p]  = command->motor_command.kd[i];
        tau_phys[p] = command->motor_command.tau[i];
        tau_phys[p] = clampf(tau_phys[p], -this->t_max_, this->t_max_);   // 硬限制电机力矩
    }
    // 每条总线构造一帧（CAN-ID 0x000），槽位按关节 CAN-ID 寻址
    for (auto& kv : this->cans_) {
        struct canfd_frame f;
        std::memset(&f, 0, sizeof(f));
        f.can_id = 0x000;
        f.len = 64;
        f.data[0] = 0x00; f.data[1] = 0x01;        // MSGID 0x0100
        for (int p = 0; p < this->num_dofs_; ++p) {
            if (this->motor_if_[p] != kv.first) continue;
            int off = 2 + (this->motor_canid_[p] - 1) * 10;
            PackJoint(&f.data[off], q_phys[p], dq_phys[p], kp_phys[p], kd_phys[p], tau_phys[p],
                      this->p_max_, this->v_max_, this->t_max_, this->kp_max_, this->kd_max_);
        }
        kv.second->send(f);
    }
}

// 使能 / 失能所有关节
void RL_Real::SendEnable(bool enable) {
    uint8_t code = enable ? 0xFC : 0xFD;
    for (auto& kv : this->cans_) {
        struct canfd_frame f;
        std::memset(&f, 0, sizeof(f));
        f.can_id = 0x000;
        f.len = 64;
        f.data[0] = 0x00; f.data[1] = 0x01;        // MSGID 0x0100
        for (int p = 0; p < this->num_dofs_; ++p) {
            if (this->motor_if_[p] != kv.first) continue;
            int off = 2 + (this->motor_canid_[p] - 1) * 10;
            for (int k = 0; k < 9; ++k) f.data[off + k] = 0xFF;
            f.data[off + 9] = code;                 // 使能 / 失能指令码
        }
        kv.second->send(f);
    }
    std::cout << LOGGER::INFO << (enable ? "电机已使能" : "电机已失能") << std::endl;
}

// TODO: 读取 RK3588 上的真实 IMU（板载 ICM/BMI 经 I2C/SPI，或外接 IMU）。
// 策略需要重力向量和角速度，上真机前必须填充此处。
void RL_Real::ReadIMU(RobotState<float> *state) {
    // 占位：单位四元数 + 零角速度（策略无法正确工作！）
    state->imu.quaternion = {1.0f, 0.0f, 0.0f, 0.0f};
    state->imu.gyroscope = {0.0f, 0.0f, 0.0f};
}

// 控制循环：读取状态 -> 状态机 -> 下发命令
void RL_Real::RobotControl() {
    this->GetState(&this->robot_state);
    this->StateController(&this->robot_state, &this->robot_command);
    this->control.ClearInput();
    this->SetCommand(&this->robot_command);
}

// 策略推理循环
void RL_Real::RunModel() {
    if (this->rl_init_done) {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands[0] = this->control.x;
        this->obs.commands[1] = this->control.y;
        this->obs.commands[2] = this->control.yaw;
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (!this->output_dof_pos.empty()) output_dof_pos_queue.push(this->output_dof_pos);
        if (!this->output_dof_vel.empty()) output_dof_vel_queue.push(this->output_dof_vel);
        if (!this->output_dof_tau.empty()) output_dof_tau_queue.push(this->output_dof_tau);
    }
}

std::vector<float> RL_Real::Forward() {
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        std::cout << LOGGER::WARNING << "模型正在重新初始化，使用上一帧动作" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (!this->params.Get<std::vector<int>>("observations_history").empty()) {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    } else {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() &&
        !this->params.Get<std::vector<float>>("clip_actions_lower").empty()) {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"),
                     this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    return actions;
}

int main(int argc, char **argv) {
    signal(SIGINT, signalHandler);
    RL_Real rl_sar(argc, argv);
    while (!g_shutdown_requested) { sleep(1); }
    std::cout << LOGGER::INFO << "正在退出..." << std::endl;
    return 0;
}
