class Process:
    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.completion_time = 0
        self.turnaround_time = 0
        self.waiting_time = 0

def fcfs_scheduling(processes):
    processes.sort(key=lambda x: x.arrival_time)  # Sort processes by arrival time
    current_time = 0
    for process in processes:
        if current_time < process.arrival_time:
            current_time = process.arrival_time
        process.completion_time = current_time + process.burst_time
        process.turnaround_time = process.completion_time - process.arrival_time
        process.waiting_time = process.turnaround_time - process.burst_time
        current_time = process.completion_time

    # Print the results
    print("PID\tArrival\tBurst\tCompletion\tTurnaround\tWaiting")
    for process in processes:
        print(f"{process.pid}\t{process.arrival_time}\t{process.burst_time}\t"
              f"{process.completion_time}\t{process.turnaround_time}\t{process.waiting_time}")

# Example usage:
if __name__ == "__main__":
    processes = [
        Process(1, 0, 4),
        Process(2, 1, 3),
        Process(3, 2, 1),
        Process(4, 3, 2)
    ]
    fcfs_scheduling(processes)
"""
First-Come, First-Served (FCFS) is one of the simplest scheduling algorithms used in operating systems. It works on a straightforward principle: processes are attended to in the order they arrive, without considering their burst time or priority.
The implementation of the FCFS algorithm in Python is simple and does not require any external libraries other than the standard Python library.
Explanation:
    Process Class: Each process is represented by a Process class, which stores its ID, arrival time, burst time, completion time, turnaround time, and waiting time.
    Sorting by Arrival Time: The processes are first sorted by their arrival time to ensure they are executed in the order they arrive.
    Scheduling Logic:
        The current_time variable keeps track of the system clock.
        For each process, the completion time is calculated based on the current_time.
        The turnaround time is the difference between the completion time and the arrival time, while the waiting time is the difference between the turnaround time and the burst time.
        The current_time is updated after each process is executed.
    Output: The program prints the Process ID (PID), arrival time, burst time, completion time, turnaround time, and waiting time for each process.
FCFS is simple and easy to implement, but it can lead to poor performance in certain scenarios, such as when processes with long burst times arrive first, causing the shorter jobs to wait, a phenomenon known as the "convoy effect." Despite this, it remains a foundational concept in understanding process scheduling.
"""