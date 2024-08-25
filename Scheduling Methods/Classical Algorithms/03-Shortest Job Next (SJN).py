class Process:
    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.completion_time = 0
        self.turnaround_time = 0
        self.waiting_time = 0

def sjn_scheduling(processes):
    time = 0
    completed = 0
    n = len(processes)
    is_completed = [False] * n
    processes.sort(key=lambda x: x.arrival_time)
    
    while completed != n:
        idx = -1
        min_burst = float('inf')
        for i in range(n):
            if processes[i].arrival_time <= time and not is_completed[i]:
                if processes[i].burst_time < min_burst:
                    min_burst = processes[i].burst_time
                    idx = i
                if processes[i].burst_time == min_burst:
                    if processes[i].arrival_time < processes[idx].arrival_time:
                        idx = i
        if idx != -1:
            processes[idx].completion_time = time + processes[idx].burst_time
            processes[idx].turnaround_time = processes[idx].completion_time - processes[idx].arrival_time
            processes[idx].waiting_time = processes[idx].turnaround_time - processes[idx].burst_time
            time += processes[idx].burst_time
            is_completed[idx] = True
            completed += 1
        else:
            time += 1

    # Print the results
    print("PID\tArrival\tBurst\tCompletion\tTurnaround\tWaiting")
    for process in processes:
        print(f"{process.pid}\t{process.arrival_time}\t{process.burst_time}\t"
              f"{process.completion_time}\t{process.turnaround_time}\t{process.waiting_time}")

# Example usage:
if __name__ == "__main__":
    processes = [
        Process(1, 0, 6),
        Process(2, 2, 8),
        Process(3, 4, 7),
        Process(4, 5, 3)
    ]
    sjn_scheduling(processes)
"""
Shortest Job Next (SJN) is a non-preemptive CPU scheduling algorithm that selects the process with the shortest burst time from the list of available processes for execution next. If two processes have the same burst time, the one that arrived first is selected. This algorithm is optimal in terms of minimizing the average waiting time for a given set of processes, but it requires prior knowledge of the burst time of each process, which may not always be available in real-world scenarios.
The Shortest Job Next (SJN) algorithm implementation does not require any external Python libraries beyond the standard Python library.
Explanation:
    Process Class: Each process is represented by a Process class, which includes attributes such as process ID (pid), arrival time, burst time, completion time, turnaround time, and waiting time.

    Scheduling Function:
        Sorting: The processes are initially sorted by their arrival time.
        Process Selection: The algorithm selects the process with the shortest burst time that has already arrived by the current time.
        Execution: The selected process is executed until completion, and its completion time, turnaround time, and waiting time are calculated.
        Time Advancement: The system time is advanced by the burst time of the executed process, and the process is marked as completed.
        This process repeats until all processes are completed.
    Output: The program prints a table showing the Process ID, arrival time, burst time, completion time, turnaround time, and waiting time for each process.
Shortest Job Next is advantageous for reducing average waiting time but can suffer from the "starvation" problem, where longer processes may be perpetually delayed if shorter processes keep arriving. This makes SJN ideal for batch processing systems where all jobs are known ahead of time.
"""