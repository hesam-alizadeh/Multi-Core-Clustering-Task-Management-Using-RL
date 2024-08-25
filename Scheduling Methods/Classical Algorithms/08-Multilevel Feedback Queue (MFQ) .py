import queue

class Process:
    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.start_time = -1
        self.completion_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0
        self.queue_level = 0

def multilevel_feedback_queue(process_list, time_quantums):
    queues = [queue.Queue() for _ in time_quantums]
    time = 0
    completed_processes = []

    while process_list or any(not q.empty() for q in queues):
        for process in process_list[:]:
            if process.arrival_time <= time:
                queues[0].put(process)
                process_list.remove(process)
        
        for i, quantum in enumerate(time_quantums):
            if not queues[i].empty():
                current_process = queues[i].get()

                if current_process.start_time == -1:
                    current_process.start_time = time

                if current_process.remaining_time > quantum:
                    time += quantum
                    current_process.remaining_time -= quantum
                    if i + 1 < len(queues):
                        queues[i + 1].put(current_process)
                else:
                    time += current_process.remaining_time
                    current_process.remaining_time = 0
                    current_process.completion_time = time
                    current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                    current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                    completed_processes.append(current_process)
                break
        else:
            time += 1

    return completed_processes

# Example usage
if __name__ == "__main__":
    process_list = [
        Process(pid=1, arrival_time=0, burst_time=10),
        Process(pid=2, arrival_time=2, burst_time=4),
        Process(pid=3, arrival_time=4, burst_time=6),
        Process(pid=4, arrival_time=6, burst_time=8),
    ]

    time_quantums = [4, 8, 12]  # Different time quantums for each queue level

    completed_processes = multilevel_feedback_queue(process_list, time_quantums)

    print("PID\tArrival Time\tBurst Time\tCompletion Time\tWaiting Time\tTurnaround Time")
    for process in completed_processes:
        print(f"{process.pid}\t{process.arrival_time}\t\t{process.burst_time}\t\t{process.completion_time}\t\t{process.waiting_time}\t\t{process.turnaround_time}")
"""
Multilevel Feedback Queue (MFQ) is an advanced CPU scheduling algorithm that allows processes to move between different queues based on their behavior and characteristics. The primary objective of MFQ is to optimize both response time and turnaround time while ensuring fairness among processes.
Required Libraries: The Multilevel Feedback Queue (MFQ) algorithm implementation uses the built-in "queue" library in Python. No external libraries are required, making the code straightforward to run in any Python environment.
Explanation:
Key Features of the Code:
    Process Class:
        Encapsulates the details of each process, including pid, arrival_time, burst_time, remaining_time, start_time, completion_time, waiting_time, and turnaround_time.
        The queue_level attribute determines the current queue level of the process.

    Multiple Queues:
        Processes are organized into multiple queues, each with a different time quantum.
        The process starts in the highest-priority queue and may move to lower-priority queues if it exceeds the time quantum.

    Time Quantum:
        Each queue has a defined time quantum. If a process exceeds its time quantum in one queue, it is moved to the next lower-priority queue.
        Processes are only completed when their remaining_time is fully executed.

    Dynamic Time Management:
        The system clock (time) advances dynamically based on the time quantum and process execution time.
        The code handles idle periods when no processes are ready to execute, ensuring continuous advancement of time.

    Non-Preemptive Execution:
        Each process is executed non-preemptively within its assigned queue for the duration of its time quantum.
"""