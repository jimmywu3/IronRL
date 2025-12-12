# IronRL: MineRL PPO Agent

This repository implements a Proximal Policy Optimization (PPO) agent for the MineRL environment. It is designed to be cross-platform, with specific compatibility fixes for Apple Silicon (M1/M2) architecture.

## ⚠️ Prerequisites

Before installing, ensure you have the following software:

1.  **Python 3.8** (Strictly required).
2.  **Java 8 (JDK 1.8)** (Strictly required for the Minecraft subprocess).
    * *Note: Java 11, 17, or 21 will cause the build to crash with `Pack200` errors.*
    ```bash
    brew install --cask temurin@8
    ```

---

## Installation


### Option A: Apple Silicon (M1/M2/M3) Mac (Recommended)
Running legacy MineRL on Apple Silicon requires specific emulation steps. Please follow these commands exactly.

**1. Create an Intel (x86_64) Emulated Environment:**
   You must force Conda to use Intel architecture, or the environment will crash.\
   Installation for Conda: https://www.anaconda.com/docs/getting-started/miniconda/install
   ```bash
   CONDA_SUBDIR=osx-64 conda create -n minerl_env python=3.8
   conda activate minerl_env
   conda config --env --set subdir osx-64
   ```
   
## Downgrade Build tools
Modern pip/setuptools are incompatible with older gym versions.

```bash
pip install "pip<24.0" "setuptools==65.5.0" "wheel<0.40.0"
```

## Install Pre-Built OpenCV

```bash
pip install "opencv-python<4.6" --only-binary=:all:
```

## Set Java 8 Home
```bash
export JAVA_HOME="/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home"
```

## Install MineRL with Patch
Will take a bit.
```bash
sh setup_mac.sh
```

## Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

## How To Run
Will create a video file in folder videos_phase3 on first episdoe and every 5th episodes after.\
Will give warnings, ignore them. Takes 1-3 minutes to boot up Gradle.
```bash
# (Mac Only) export JAVA_HOME="/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home"
python train_ppo_phase3.py
```

---

### Option B: Windows 10/11
## 1. Install Java 8 (JDK 1.8):

Download Eclipse Temurin JDK 8 (LTS) installer (.msi) from [Adoptium.net.](https://adoptium.net/temurin/releases/?version=8)

Crucial: During installation, select "Set JAVA_HOME variable" in the setup menu.

## 2. Install C++ Build Tools:

If you encounter errors building gym, ensure you have Visual Studio Build Tools installed with the "Desktop development with C++" workload.

## 3. Create Conda Environment:
If you don't have conda, use this to install conda: https://www.anaconda.com/docs/getting-started/miniconda/install
```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe
start /wait "" .\miniconda.exe /S
del .\miniconda.exe
```
Click the Windows key and search "Anaconda Prompt", open it we will be using this command prompt for the project.

```bash
conda create -n minerl_env python=3.8
conda activate minerl_env
```

## 4. Downgrade Build Tools: Required for legacy Gym compatibility.
```bash
pip install "pip<24.0" "setuptools==65.5.0" "wheel<0.40.0"
```

## 5. Install Dependencies:
```bash
# Install MineRL explicitly
pip install minerl==0.4.4

# Install the rest
pip install -r requirements.txt
```

## How To Run
Will create a video file in folder videos_phase3 on the first episode and every 5th episode after. It takes 1-3 minutes to boot up Gradle/Minecraft. Ignore the initial warnings.

```bash
python train_ppo_phase3.py
```