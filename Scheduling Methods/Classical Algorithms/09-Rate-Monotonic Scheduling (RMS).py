import math

class Task:
    def __init__(self, task_id, execution_time, period):
        self.task_id = task_id
        self.execution_time = execution_time
        self.period = period
        self.remaining_time = execution_time
        self.deadline = period
        self.completed = False

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

def calculate_hyperperiod(tasks):
    hyperperiod = tasks[0].period
    for task in tasks[1:]:
        hyperperiod = lcm(hyperperiod, task.period)
    return hyperperiod

def rate_monotonic_scheduling(tasks):
    hyperperiod = calculate_hyperperiod(tasks)
    time = 0
    scheduled_tasks = []

    while time < hyperperiod:
        current_task = None

        for task in sorted(tasks, key=lambda x: x.period):
            if time % task.period == 0:
                task.remaining_time = task.execution_time
                task.deadline = time + task.period

            if task.remaining_time > 0 and (current_task is None or task.period < current_task.period):
                current_task = task

        if current_task:
            current_task.remaining_time -= 1
            scheduled_tasks.append(current_task.task_id)
            if current_task.remaining_time == 0:
                current_task.completed = True
        else:
            scheduled_tasks.append(None)

        time += 1

    return scheduled_tasks, hyperperiod

# Example usage
if __name__ == "__main__":
    tasks = [
        Task(task_id=1, execution_time=1, period=4),
        Task(task_id=2, execution_time=2, period=6),
        Task(task_id=3, execution_time=3, period=8)
    ]

    scheduled_tasks, hyperperiod = rate_monotonic_scheduling(tasks)

    print("Scheduled Tasks:")
    for i in range(hyperperiod):
        task_id = scheduled_tasks[i]
        if task_id:
            print(f"Time {i}: Task {task_id}")
        else:
            print(f"Time {i}: Idle")
"""
Rate-Monotonic Scheduling (RMS) is a real-time scheduling algorithm that is based on static priorities. In RMS, tasks are assigned priorities according to the length of their periods: the shorter the period, the higher the priority. RMS is an optimal algorithm for fixed-priority scheduling on a single processor, meaning that if a set of tasks can be scheduled by any fixed-priority algorithm, it can also be scheduled by RMS.
Required Libraries: The Rate-Monotonic Scheduling (RMS) algorithm implementation uses the built-in "math" library in Python, which provides functions like "gcd" for calculating the greatest common divisor, used in the LCM calculation.
Explanation:
Key Features of the Code:
    Task Class:
        The Task class encapsulates the properties of each task, including task_id, execution_time, period, remaining_time, deadline, and completed.
        The remaining_time attribute tracks the amount of execution time left for a task, and deadline is used to monitor when the task should complete within its period.
    Hyperperiod Calculation:
        The hyperperiod is the least common multiple (LCM) of all task periods. It represents the time span over which the task set repeats its schedule.
        The lcm function calculates the least common multiple, and calculate_hyperperiod determines the hyperperiod for the entire task set.

    Task Scheduling:
        Tasks are scheduled based on their priority (i.e., the period). The scheduler checks each task at each time unit to determine which one to run.
        If a task has arrived (its period has started) and it has the shortest period (highest priority), it is selected to execute.
        The task's remaining execution time is decremented each time it runs, and it is marked as completed when remaining_time reaches zero.

    Idle Time:
        If no tasks are ready to execute at a given time, the system is idle. This is handled by appending None to the scheduled_tasks list, indicating idle time.
"""