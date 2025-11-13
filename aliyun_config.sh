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
    git \
    openssh-server \
    tigervnc-common

# Install noVNC
echo "Installing noVNC..."
sudo snap install novnc

# Install dummy video driver (already included above, but ensuring it's there)
echo "Ensuring dummy video driver is installed..."
sudo apt-get install -y xserver-xorg-video-dummy

# Set up VNC password
echo "Setting up VNC password..."
mkdir -p ~/.vnc

# Create VNC password file using interactive method
echo "Creating VNC password file..."
echo "Please enter VNC password when prompted (recommended: osworld-public-evaluation)"
x11vnc -storepasswd ~/.vnc/passwd
chmod 600 ~/.vnc/passwd

# Verify password file was created
if [ -f ~/.vnc/passwd ]; then
    echo "✅ VNC password file created successfully"
else
    echo "❌ Failed to create VNC password file"
    exit 1
fi

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

# Enable and start SSH service
echo "Enabling and starting SSH service..."
sudo systemctl enable ssh
sudo systemctl start ssh

# Check SSH service status
echo "Checking SSH service status..."
sudo systemctl status ssh --no-pager -l

# Ensure SSH allows password authentication
echo "Configuring SSH..."
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
sudo sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config
sudo sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Restart SSH service to apply configuration changes
echo "Restarting SSH service..."
sudo systemctl restart ssh

# Check if SSH is listening on port 22
echo "Verifying SSH is listening on port 22..."
sudo netstat -tlnp | grep :22

# Create systemd user directory if it doesn't exist
mkdir -p ~/.config/systemd/user

# Create x11vnc service with password authentication
echo "Creating x11vnc service..."
tee ~/.config/systemd/user/x11vnc.service > /dev/null << 'EOF'
[Unit]
Description=X11 VNC Server
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=simple
ExecStart=/bin/bash -c 'sleep 5 && sudo chown $USER:$USER /tmp/.X11-unix/X0 2>/dev/null || true && x11vnc -display :0 -rfbport 5900 -forever -shared -noxdamage -noxfixes -noxrandr -rfbauth ~/.vnc/passwd'
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
echo "Enabling systemd services..."
systemctl --user daemon-reload
systemctl --user enable x11vnc.service
systemctl --user enable novnc.service

# Create manual startup script
echo "Creating manual startup script..."
tee ~/start_vnc.sh > /dev/null << 'EOF'
#!/bin/bash

echo "Starting VNC services manually..."

# Stop any existing services
sudo pkill -f x11vnc
sudo pkill -f novnc
sudo pkill -f websockify
sleep 2

# Set DISPLAY environment variable
export DISPLAY=:0

# Check if password file exists
if [ ! -f ~/.vnc/passwd ]; then
    echo "VNC password file not found. Creating one..."
    echo "Please enter VNC password when prompted:"
    x11vnc -storepasswd ~/.vnc/passwd
    chmod 600 ~/.vnc/passwd
fi

# Start x11vnc with password
echo "Starting x11vnc..."
x11vnc -display :0 -rfbport 5900 -forever -shared -noxdamage -noxfixes -noxrandr -rfbauth ~/.vnc/passwd &

# Wait for x11vnc to start
sleep 3

# Start noVNC
echo "Starting noVNC..."
/snap/bin/novnc --vnc localhost:5900 --listen 5910 &

# Wait for services to start
sleep 2

# Check status
echo "=== Service Status ==="
echo "x11vnc process:"
ps aux | grep x11vnc | grep -v grep

echo "noVNC process:"
ps aux | grep novnc | grep -v grep

echo "Port status:"
netstat -tlnp | grep -E "(5900|5910)"

echo ""
echo "VNC services started!"
echo "Connect via: http://<server-ip>:5910"
echo "Use the password you set during setup"
EOF

chmod +x ~/start_vnc.sh

echo "=== Setup Complete ==="
echo ""
echo "✅ VNC password has been set interactively"
echo "✅ SSH service has been enabled and started"
echo "✅ SystemD services have been created and enabled"
echo "✅ Manual startup script created: ~/start_vnc.sh"
echo ""
echo "Connection methods:"
echo "- SSH: ssh user@<server-ip> (use system user password)"
echo "- VNC client: <server-ip>:5900 (use the password you set)"
echo "- noVNC web: http://<server-ip>:5910 (use the password you set)"
echo ""
echo "To start VNC services manually:"
echo "  ./start_vnc.sh"
echo ""
echo "To start VNC services via systemd:"
echo "  systemctl --user start x11vnc.service"
echo "  systemctl --user start novnc.service"
echo ""
echo "SSH service is running on port 22"
echo ""
echo "IMPORTANT: The VNC password was set interactively during setup."
echo "If you forget it, run: x11vnc -storepasswd ~/.vnc/passwd to reset it."
