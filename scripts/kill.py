import subprocess
import os
import psutil

def get_nvidia_smi_output():
    # Get the output of nvidia-smi
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def parse_nvidia_smi_output(output):
    # Parse the nvidia-smi output and return a list of (pid, memory) tuples
    processes = []
    for line in output.strip().split('\n'):
        if line:
            pid, memory = line.split(',')
            pid = int(pid.strip())
            memory = int(memory.strip())
            processes.append((pid, memory))
    return processes

def terminate_large_processes(processes, n_keep=0, min_memory=600):
    # Sort processes by PID in descending order and filter by memory usage
    sorted_processes = sorted(processes, key=lambda x: x[0], reverse=True)
    
    current_uid = os.getuid()
    user_processes = [p for p in sorted_processes if psutil.Process(p[0]).uids().real == current_uid]

    print("User's GPU processes:", user_processes)

    # Determine which processes to terminate
    processes_to_terminate = [p for p in user_processes[n_keep:] if p[1] > min_memory]
    
    for pid, memory in processes_to_terminate:
        try:
            subprocess.run(['kill', '-9', str(pid)])
            print(f"Terminated process PID {pid} with memory {memory}MiB")
        except Exception as e:
            print(f"Failed to terminate process PID {pid}: {e}")

# Get the current GPU processes
nvidia_smi_output = get_nvidia_smi_output()
processes = parse_nvidia_smi_output(nvidia_smi_output)
terminate_large_processes(processes)
