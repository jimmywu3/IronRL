# MineRL PPO Agent

This project implements a PPO agent for the MineRL environment.

## ⚠️ Important Prerequisites
**MineRL v0.4 requires specific system dependencies to run.**

1.  **Python Version:** You must use **Python 3.9.4**
2.  **Java Version:** You must have **Java 8 (JDK 8)** installed.
    * *Note: Newer versions like Java 11 or 17 will cause the build to crash.*

## Installation

1.  **Install System Libraries** (Linux/Mac only):
    * **Ubuntu/Debian:** `sudo apt-get install openjdk-8-jdk ffmpeg`
    * **MacOS:** `brew install --cask adoptopenjdk/openjdk/adoptopenjdk8` and `brew install ffmpeg`

2.  **Create a Virtual Environment:**
    ```bash
    python3.8 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## How to Run

To train the agent:
```bash
python train_ppo_phase3.py