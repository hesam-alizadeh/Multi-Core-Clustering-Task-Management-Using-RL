class Process:
    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.completion_time = 0
        self.turnaround_time = 0
        self.waiting_time = 0

def round_robin_scheduling(processes, quantum):
    time = 0
    queue = []
    complete = []
    
    while processes or queue:
        # Add arriving processes to the queue
        while processes and processes[0].arrival_time <= time:
            queue.append(processes.pop(0))
        
        if queue:
            current_process = queue.pop(0)
            execution_time = min(quantum, current_process.remaining_time)
            current_process.remaining_time -= execution_time
            time += execution_time

            # If the process is completed
            if current_process.remaining_time == 0:
                current_process.completion_time = time
                current_process.turnaround_time = time - current_process.arrival_time
                current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                complete.append(current_process)
            else:
                # Re-queue the process if not completed
                queue.append(current_process)
        else:
            time += 1  # If no process is ready, increment time

    # Print the results
    print("PID\tArrival\tBurst\tCompletion\tTurnaround\tWaiting")
    for process in sorted(complete, key=lambda x: x.pid):
        print(f"{process.pid}\t{process.arrival_time}\t{process.burst_time}\t"
              f"{process.completion_time}\t{process.turnaround_time}\t{process.waiting_time}")

# Example usage:
if __name__ == "__main__":
    processes = [
        Process(1, 0, 5),
        Process(2, 1, 4),
        Process(3, 2, 2),
        Process(4, 3, 1)
    ]
    quantum = 2
    round_robin_scheduling(processes, quantum)
"""
Round Robin is particularly effective in time-sharing systems and ensures that all processes receive a fair share of CPU time. However, the choice of quantum size is crucial; if the quantum is too large, RR behaves like FCFS, and if it's too small, it results in high context switching overhead.
The Round Robin algorithm implementation does not require any external Python libraries beyond the standard Python library.
Explanation:
Round Robin (RR) is a preemptive CPU scheduling algorithm where each process is assigned a fixed time slice or "quantum." The process executes for this quantum, and if it does not finish, it is moved to the back of the queue. This approach is designed to ensure fairness among processes and to reduce the risk of long waiting times for shorter jobs.
    Process Class: Each process is represented by a Process class, which includes attributes such as process ID (pid), arrival time, burst time, remaining time, completion time, turnaround time, and waiting time.
    Queue and Time Management:
        The queue stores processes that are ready to be executed.
        The time variable tracks the current time in the system.
        Processes are added to the queue when their arrival time is less than or equal to the current time.
        The CPU executes the process for a duration that is either the quantum or the remaining time of the process, whichever is smaller.
    Execution and Completion:
        If a process finishes execution within its quantum, it is marked as complete.
        If it does not finish, it is placed back in the queue for further execution.
        The algorithm continues until all processes have completed execution.
    Output: The program prints the Process ID, arrival time, burst time, completion time, turnaround time, and waiting time for each process in a well-organized table.

"""