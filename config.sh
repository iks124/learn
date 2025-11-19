#!/bin/bash

# OSWorld VNC Setup Script
# Run this script on a fresh Ubuntu system

set -e  # Exit on any error

echo "=== OSWorld VNC Setup Script ==="
echo "Setting up VNC and OSWorld server..."

# Update system
echo "Updating system packages..."
sudo apt update

# Install required packages
echo "Installing required packages..."
sudo apt install -y \
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

# Install noVNC
echo "Installing noVNC..."
sudo snap install novnc

# Install dummy video driver (already included above, but ensuring it's there)
echo "Ensuring dummy video driver is installed..."
sudo apt-get install -y xserver-xorg-video-dummy

# Create X11 configuration
echo "Creating X11 configuration..."
sudo tee /etc/X11/xorg.conf > /dev/null << 'EOF'
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

# Create systemd user directory if it doesn't exist
mkdir -p ~/.config/systemd/user

# Create x11vnc service
echo "Creating x11vnc service..."
tee ~/.config/systemd/user/x11vnc.service > /dev/null << 'EOF'
[Unit]
Description=X11 VNC Server
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=simple
ExecStart=/bin/bash -c 'sleep 5 && sudo chown $USER:$USER /tmp/.X11-unix/X0 2>/dev/null || true && x11vnc -display :0 -rfbport 5900 -forever -shared -noxdamage -noxfixes -noxrandr'
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
EOF

# Create noVNC service
echo "Creating noVNC service..."
tee ~/.config/systemd/user/novnc.service > /dev/null << 'EOF'
[Unit]
Description=noVNC Service
After=x11vnc.service network.target
Wants=x11vnc.service

[Service]
Type=simple
ExecStart=/snap/bin/novnc --vnc localhost:5900 --listen 5910
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
EOF

# Set up OSWorld server directory
echo "Setting up OSWorld server directory..."
mkdir -p /home/user/server

# Enable systemd services

# 必须重启gdm3，让图形界面loginctl show-session $(loginctl | grep $USER | awk '{print $1}') -p Type显示为x11而非wayland，才能在浏览器上访问。
sudo systemctl restart gdm3

echo "Enabling systemd services..."
systemctl --user daemon-reload
systemctl --user enable x11vnc.service
systemctl --user enable novnc.service
