/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * 自研 "ancnw" 轮腿四足机器人的真机入口程序。
 * 硬件平台：RK3588 主板 + CAN-FD 关节模组（参见《关节模组使用手册V1.2》）。
 *
 * 这里实现的协议为 MIT 阻抗控制模式：
 *   - 主机发送帧：CAN-ID 0x000，64 字节 FD 帧；最多 6 个关节打包为 10 字节的槽位，
 *     按关节自身的 CAN-ID（1..6）排列。
 *   - 每个关节槽位（10 字节，小端，0x8000 表示 0）：
 *        [位置 2][速度 2][Kp 2][Kd 2][力矩 2]
 *     位置/速度/力矩：0..65535 映射到 [-最大值, +最大值]；Kp/Kd：0..65535 映射到 [0, KP/KD_MAX]
 *   - 关节回报帧：12 字节，仲裁 ID 即该关节的 CAN-ID：
 *        [MSGID 2][位置 2][速度 2][力矩 2][温度 1][状态 1][电压 1][保留 1]
 *   - 使能：第 11 字节 == 0xFC；失能：== 0xFD；MSGID == 0x0100
 *
 * 控制主循环（GetState -> StateController(FSM) -> SetCommand）继承自 rl_sdk，
 * 这里只实现了 GetState/SetCommand 以及 CAN-FD 传输层。
 * ReadIMU() 目前仍是占位实现——上真机前必须接上真实 IMU。
 */
#pragma once

#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include "rl_sdk.hpp"
#include "canfd_comm.hpp"
#include "loop.hpp"

// ancnw 真机控制类
class RL_Real : public RL {
public:
    RL_Real(int argc, char **argv);
    ~RL_Real();

    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;

    void RunModel();
    void RobotControl();

private:
    void RecvThreadFunc(const std::string& ifname);   // 每条总线一个 CAN 状态接收线程
    void PackAndSend(const RobotCommand<float> *command);  // MIT 模式发送
    void SendEnable(bool enable);    // 0xFC 使能 / 0xFD 失能
    void ReadIMU(RobotState<float> *state);   // TODO: 接入真实 IMU

    std::map<std::string, std::unique_ptr<CanFdComm>> cans_;  // 接口名 -> 通信对象
    std::vector<std::thread> recv_threads_;   // 接收线程
    std::atomic<bool> running_{false};        // 运行标志
    std::mutex state_mutex_;                  // 状态互斥锁

    std::vector<float> motor_q_, motor_dq_, motor_tau_;   // 按物理电机序号索引
    std::vector<int> joint_mapping_;          // 策略索引 -> 物理索引
    std::vector<std::string> motor_if_;       // 物理电机 -> 所在总线接口
    std::vector<int> motor_canid_;            // 物理电机 -> CAN-ID（1..6）
    std::map<std::pair<std::string,int>, int> bus_to_phys_;  // (接口, CAN-ID) -> 物理电机序号

    // 电机极限（取自 base.yaml，必须与模组型号一致）
    float p_max_ = 12.5f, v_max_ = 65.0f, t_max_ = 30.0f;
    float kp_max_ = 500.0f, kd_max_ = 5.0f;
    int num_dofs_ = 0;

    std::shared_ptr<LoopFunc> loop_keyboard, loop_control, loop_rl;  // 三个控制循环
};
