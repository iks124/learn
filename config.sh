#!/bin/bash

# OSWorld VNC Setup Script (systemd system services version)
# 运行前提：用 root 执行
# 例如：sudo bash setup_osworld_vnc_system.sh

set -e  # Exit on any error

if [ "$(id -u)" -ne 0 ]; then
    echo "本脚本需要以 root 运行。例如：sudo bash $0"
    exit 1
fi

OSWORLD_USER="user"     # TODO: 如果你的普通用户不是 user，请改成你的用户名
OSWORLD_HOME="/home/${OSWORLD_USER}"

echo "=== OSWorld VNC Setup Script (system services) ==="
echo "Target user: ${OSWORLD_USER}"
echo

#-----------------------------
# 1. 更新系统 & 安装依赖
#-----------------------------
echo "[1/5] 更新系统包索引..."
apt update

echo "[2/5] 安装依赖包..."
apt install -y \
    x11vnc \
    xserver-xorg-video-dummy \
    python3 \
    python3-pip \
    python3-tk \
    python3-dev \
    gnome-screenshot \
    wmctrl \
    ffmpeg \
    socat \
    xclip \
    git

echo "安装 noVNC (snap)..."
snap install novnc

echo "再次确认 dummy 视频驱动安装..."
apt install -y xserver-xorg-video-dummy

#-----------------------------
# 2. 配置 Xorg dummy 显示
#-----------------------------
echo "[3/5] 创建 /etc/X11/xorg.conf (dummy 显示)..."

cat >/etc/X11/xorg.conf << 'EOF'
Section "Device"
    Identifier "DummyDevice"
    Driver "dummy"
    VideoRam 256000
EndSection

Section "Monitor"
    Identifier "DummyMonitor"
    HorizSync 28.0-80.0
    VertRefresh 48.0-75.0
    Modeline "1920x1080" 172.80 1920 2048 2248 2576 1080 1083 1088 1120
EndSection

Section "Screen"
    Identifier "DummyScreen"
    Device "DummyDevice"
    Monitor "DummyMonitor"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1920x1080"
    EndSubSection
EndSection
EOF

#-----------------------------
# 3. 创建 OSWorld 目录
#-----------------------------
echo "[4/5] 创建 OSWorld server 目录..."

mkdir -p "${OSWORLD_HOME}/server"
chown -R "${OSWORLD_USER}:${OSWORLD_USER}" "${OSWORLD_HOME}/server" || true

#-----------------------------
# 4. 创建 systemd system 服务
#-----------------------------
echo "[5/5] 创建 systemd system 服务..."

##########################
# 4.1 Xorg dummy service #
##########################
cat >/etc/systemd/system/osworld-xorg.service << 'EOF'
[Unit]
Description=OSWorld Xorg Dummy Display
After=systemd-logind.service network.target
Conflicts=getty@tty7.service

[Service]
Type=simple
# 使用 dummy 配置启动 Xorg，display :0
ExecStart=/usr/bin/X :0 -config /etc/X11/xorg.conf -nolisten tcp -noreset
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

##########################
# 4.2 x11vnc service     #
##########################
cat >/etc/systemd/system/osworld-x11vnc.service << 'EOF'
[Unit]
Description=OSWorld x11vnc Server
After=osworld-xorg.service
Requires=osworld-xorg.service

[Service]
Type=simple
# 不设置密码，实验环境 OK，生产环境请加 -passwd 或 -rfbauth
ExecStart=/usr/bin/x11vnc \
    -display :0 \
    -rfbport 5900 \
    -forever \
    -shared \
    -noxdamage \
    -noxfixes \
    -noxrandr \
    -auth guess
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

##########################
# 4.3 noVNC service      #
##########################
cat >/etc/systemd/system/osworld-novnc.service << 'EOF'
[Unit]
Description=OSWorld noVNC Server
After=osworld-x11vnc.service network-online.target
Requires=osworld-x11vnc.service

[Service]
Type=simple
ExecStart=/snap/bin/novnc --vnc localhost:5900 --listen 5910
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

#-----------------------------
# 5. 启用 & 启动服务
#-----------------------------
echo "重载 systemd 配置..."
systemctl daemon-reload

echo "启用并启动 Xorg dummy..."
systemctl enable --now osworld-xorg.service

echo "启用并启动 x11vnc..."
systemctl enable --now osworld-x11vnc.service

echo "启用并启动 noVNC..."
systemctl enable --now osworld-novnc.service

echo
echo "=== OSWorld VNC setup (system services) 完成 ==="
echo "现在你可以通过浏览器访问: http://<服务器IP>:5910/"
echo "⚠ 注意：当前 x11vnc 没有密码，只适合内网 / 实验环境使用。"
