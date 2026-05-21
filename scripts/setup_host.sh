#!/bin/bash
# =============================================================================
# iNIM Host Setup Script
#
# Sets up Intel GPU drivers and compute runtime on Ubuntu 22.04 LTS.
# Run this on the host machine before launching the iNIM container.
#
# Requirements:
#   - Ubuntu 22.04 LTS
#   - Root/sudo access
#   - Internet connectivity
# =============================================================================

set -euo pipefail

echo "════════════════════════════════════════════════"
echo " iNIM Host Setup — Intel GPU Driver Installation"
echo " Target OS: Ubuntu 22.04 LTS"
echo "════════════════════════════════════════════════"

# Check OS
if [ ! -f /etc/os-release ]; then
    echo "ERROR: Cannot detect OS. This script requires Ubuntu 22.04 LTS."
    exit 1
fi

source /etc/os-release
if [ "$ID" != "ubuntu" ] || [[ "$VERSION_ID" != "22.04"* ]]; then
    echo "WARNING: This script is designed for Ubuntu 22.04 LTS."
    echo "  Detected: $PRETTY_NAME"
    read -r -p "Continue anyway? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        exit 1
    fi
fi

echo ""
echo "Step 1: Installing HWE kernel (6.5+) for best Intel GPU support..."
echo "---------------------------------------------------------------------"
sudo apt-get update
sudo apt-get install -y \
    linux-generic-hwe-22.04 \
    linux-headers-generic-hwe-22.04

echo ""
echo "Step 2: Adding Intel Graphics PPA repository..."
echo "---------------------------------------------------------------------"
# Install prerequisites
sudo apt-get install -y \
    curl \
    gpg \
    software-properties-common

# Add Intel Graphics signing key
curl -fsSL https://repositories.intel.com/graphics/intel-graphics.key | \
    sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg

# Add Intel Graphics repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/graphics/ubuntu jammy flex" | \
    sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list > /dev/null

sudo apt-get update

echo ""
echo "Step 3: Installing Intel GPU compute runtime..."
echo "---------------------------------------------------------------------"
sudo apt-get install -y \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    intel-igc-core \
    intel-igc-opencl \
    ocl-icd-libopencl1 \
    clinfo

echo ""
echo "Step 4: Installing Intel media drivers (optional, for iGPU)..."
echo "---------------------------------------------------------------------"
sudo apt-get install -y \
    libva-drm2 \
    libdrm-intel1 \
    || echo "  Some optional packages not available (non-fatal)"

echo ""
echo "Step 5: Configuring i915 GuC firmware..."
echo "---------------------------------------------------------------------"
# Enable GuC and HuC firmware for better GPU scheduling
if ! grep -q "enable_guc=3" /etc/modprobe.d/i915.conf 2>/dev/null; then
    echo "options i915 enable_guc=3" | sudo tee /etc/modprobe.d/i915.conf
    echo "  i915 GuC firmware enabled"
else
    echo "  i915 GuC firmware already configured"
fi

echo ""
echo "Step 6: Adding current user to 'render' group..."
echo "---------------------------------------------------------------------"
CURRENT_USER="${SUDO_USER:-$USER}"
sudo usermod -a -G render "$CURRENT_USER"
echo "  User '$CURRENT_USER' added to 'render' group"
echo "  NOTE: You may need to log out and back in for this to take effect"

echo ""
echo "Step 7: Installing Docker (if not already installed)..."
echo "---------------------------------------------------------------------"
if command -v docker &> /dev/null; then
    echo "  Docker already installed: $(docker --version)"
else
    echo "  Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$CURRENT_USER"
    echo "  Docker installed"
fi

echo ""
echo "Step 8: Verifying GPU detection..."
echo "---------------------------------------------------------------------"
echo ""
echo "=== clinfo output ==="
clinfo --list 2>/dev/null || echo "  clinfo not working (check drivers)"

echo ""
echo "=== /dev/dri devices ==="
ls -la /dev/dri/ 2>/dev/null || echo "  /dev/dri not found"

echo ""
echo "════════════════════════════════════════════════"
echo " Host setup complete!"
echo ""
echo " IMPORTANT: Reboot if you installed a new kernel:"
echo "   sudo reboot"
echo ""
echo " Then test iNIM:"
echo "   ./scripts/run_local.sh"
echo "════════════════════════════════════════════════"
