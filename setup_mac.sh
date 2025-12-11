#!/bin/bash
set -e  # Exit immediately if a command fails

echo "==========================================="
echo "   MineRL Installer for Mac (M1/M2/Intel)  "
echo "==========================================="

# 1. Check for Java 8
if [ -z "$JAVA_HOME" ]; then
    echo "   JAVA_HOME is not set."
    echo "   Please set JAVA_HOME to your Java 8 installation."
    echo "   Example: export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)"
    exit 1
fi
echo "Using Java: $JAVA_HOME"

# 2. Create a temporary folder
BUILD_DIR="temp_minerl_build"
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

# 3. Clone MineRL
echo "   Cloning MineRL v0.4.4..."
git clone --depth 1 --branch v0.4.4 https://github.com/minerllabs/minerl.git . > /dev/null 2>&1

# 4. Patch build.gradle (The 'mkdir' fix)
cp ../backup_stuff/build.gradle minerl/Malmo/Minecraft/build.gradle


# 5. Install
echo "📦 Installing MineRL (This may take a few minutes)..."
pip install .

# 6. Clean up
cd ..
rm -rf $BUILD_DIR

echo "==========================================="
echo "    SUCCESS! MineRL installed successfully."
echo "==========================================="