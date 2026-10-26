/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */
#include "canfd_comm.hpp"

#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <net/if.h>
#include <linux/can/raw.h>
#include <iostream>

// 构造：记录接口名与是否启用 FD
CanFdComm::CanFdComm(const std::string& ifname, bool fd) : ifname_(ifname), fd_(fd) {}

// 析构：关闭套接字
CanFdComm::~CanFdComm() { close(); }

// 打开 CAN 套接字并绑定到指定接口
bool CanFdComm::open() {
    // 创建原始 CAN 套接字
    sock_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sock_ < 0) {
        std::cerr << "[CAN] socket() 失败: " << strerror(errno) << std::endl;
        return false;
    }
    // 启用 CAN-FD 帧（失败则退回经典 CAN）
    if (fd_) {
        int enable = 1;
        if (setsockopt(sock_, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &enable, sizeof(enable)) < 0) {
            std::cerr << "[CAN] 警告: 设置 CAN_RAW_FD_FRAMES 失败（退回经典 CAN）: "
                      << strerror(errno) << std::endl;
        }
    }
    // 通过接口名获取索引
    struct ifreq ifr {};
    std::strncpy(ifr.ifr_name, ifname_.c_str(), IFNAMSIZ - 1);
    if (ioctl(sock_, SIOCGIFINDEX, &ifr) < 0) {
        std::cerr << "[CAN] ioctl(SIOCGIFINDEX) 获取 " << ifname_ << " 失败: "
                  << strerror(errno) << std::endl;
        close();
        return false;
    }
    // 绑定到该接口
    struct sockaddr_can addr {};
    addr.can_family  = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(sock_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[CAN] bind() 失败: " << strerror(errno) << std::endl;
        close();
        return false;
    }
    // 设为非阻塞
    int flags = fcntl(sock_, F_GETFL, 0);
    fcntl(sock_, F_SETFL, flags | O_NONBLOCK);
    std::cout << "[CAN] 已打开 " << ifname_ << " (fd_mode=" << (fd_ ? "开" : "关") << ")" << std::endl;
    return true;
}

// 关闭套接字
void CanFdComm::close() {
    if (sock_ >= 0) {
        ::close(sock_);
        sock_ = -1;
    }
}

// 接收一帧（带可选超时）
bool CanFdComm::recv(struct canfd_frame& frame, int timeout_ms) {
    if (timeout_ms > 0) {
        fd_set rset;
        FD_ZERO(&rset);
        FD_SET(sock_, &rset);
        struct timeval tv { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
        if (select(sock_ + 1, &rset, nullptr, nullptr, &tv) <= 0) return false;
    }
    ssize_t n = read(sock_, &frame, sizeof(frame));
    return n > 0;
}

// 发送一帧
bool CanFdComm::send(const struct canfd_frame& frame) {
    const size_t len = fd_ ? CANFD_MTU : CAN_MTU;
    ssize_t n = write(sock_, &frame, len);
    if (n < 0) {
        std::cerr << "[CAN] write() 失败: " << strerror(errno) << std::endl;
        return false;
    }
    return true;
}
