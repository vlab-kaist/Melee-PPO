# Stable-Baseline3-SSBM

This project provides a framework for training reinforcement learning models using Stable Baselines3 for Super Smash Bros. Melee.

## Requirements

- Super Smash Bros. Melee ISO file
- Python 3.8 or higher
- PyTorch 1.13 or higher

## Installation

Follow the steps below to set up the environment and install the necessary dependencies:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/vlab-kaist/Stable-Baseline3-SSBM.git
    cd Stable-Baseline3-SSBM/
    ```

2. **Upgrade `pip`, `setuptools`, and `wheel`**

    ```bash
    pip install --upgrade pip setuptools wheel
    ```

3. **Install the Required Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Training

### Without Continuing Training

To start a new training session:

```bash
cd ~/Stable-Baseline3-SSBM/stable-baseline/
python train_ppo.py --iso "your_iso_file" --log_name "name_of_experiment" --save_dir "where to save model"
```

### Continuing Training

To continue training from a previously saved model:

```bash
cd ~/Stable-Baseline3-SSBM/stable-baseline/
python train_ppo.py --iso "your_iso_file" --continue_train True --save_dir "prev path_to_saved_model" --log_name "prev_log_name"
```

Replace `"your_iso_file"` with the path to your Super Smash Bros. Melee ISO file. Replace `"path_to_saved_model"` with the directory where your saved model is located and `"your_log_name"` with the name for your training logs.

**Note:** The saved model in the `save_dir` should be named `last_saved_model` for the training to continue properly.

### Additional Training Arguments

You can customize the training process using the following arguments:

- `--continue_train`: Whether to continue training with existing models. Default is `"False"`.
- `--save_dir`: Directory where the model is saved. Default is `"./models/"`.
- `--log_dir`: Directory where logs are saved. Default is `"./result_logs/"`.
- `--log_name`: Name of the experiment. Default is `"PPO"`.
- `--iso`: Path to your NTSC 1.02/PAL SSBM Melee ISO. Default is `"/home/tgkang/ssbm.iso"`.
- `--cpu_diff`: Difficulty of CPU agent (1-9). Default is `9`.
- `--total_timesteps`: Number of maximum timesteps to train. Default is `100000000` (1e8).
- `--total_episodes`: Number of maximum episodes to train. Default is `10`.
- `--n_stack`: Number of observations to stack. Default is `10`.
- `--n_envs`: Number of environments for multiprocessing. Default is `1`.
- `--n_steps`: Number of steps stored in the buffer. Default is `8192`.
- `--batch_size`: Size of the batch. Default is `1024`.

Example usage with additional arguments:

```bash
cd ~/Stable-Baseline3-SSBM/stable-baseline/
python train_ppo.py --iso "your_iso_file" --cpu_diff 5 --total_episodes 100 --log_name "MyExperiment"
```

## Accelerating Dolphin

To accelerate the Dolphin emulator, you need to press the TAB key. We provide a GUI tool to automate pressing the TAB key.

**Run the Key Macro GUI**

```bash
cd ~/Stable-Baseline3-SSBM/
python keymacro.py
```

## Additional Information

- Ensure you have the necessary permissions and environment setup to run the scripts.
- For more detailed information on the parameters and additional options, refer to the documentation or use the help command for each script.
