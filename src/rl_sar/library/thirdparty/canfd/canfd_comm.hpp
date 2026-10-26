/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * 极简的 Linux SocketCAN / CAN-FD 封装，供 rl_real_ancnw 使用。
 *
 * 运行程序前必须先把 CAN 接口拉起，例如：
 *   sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
 *   sudo ip link set can0 up
 * （或者在 RK3588 上通过 /etc/network/interfaces / systemd 配置开机自启）
 */
#pragma once

#include <string>
#include <linux/can.h>

// CAN-FD 通信封装类
class CanFdComm {
public:
    // ifname: 网络接口名（如 "can0"）；fd: 是否启用 CAN-FD
    explicit CanFdComm(const std::string& ifname, bool fd = true);
    ~CanFdComm();

    bool open();   // 打开并绑定接口
    void close();  // 关闭套接字

    // 非阻塞接收。timeout_ms 内无帧则返回 false。
    bool recv(struct canfd_frame& frame, int timeout_ms = 0);

    // 发送一帧（标准帧或 FD 帧）。出错返回 false。
    bool send(const struct canfd_frame& frame);

    bool isOpen() const { return sock_ >= 0; }
    int  sock()   const { return sock_; }

private:
    std::string ifname_;  // 接口名
    bool        fd_;      // 是否 CAN-FD
    int         sock_ = -1;  // 套接字描述符
};
