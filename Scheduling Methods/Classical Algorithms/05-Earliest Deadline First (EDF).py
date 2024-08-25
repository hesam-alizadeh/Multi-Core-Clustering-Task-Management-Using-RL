import heapq

class Task:
    def __init__(self, name, deadline, duration):
        self.name = name
        self.deadline = deadline
        self.duration = duration

    def __lt__(self, other):
        return self.deadline < other.deadline

def earliest_deadline_first(tasks):
    heapq.heapify(tasks)
    current_time = 0
    schedule = []

    while tasks:
        task = heapq.heappop(tasks)
        schedule.append((task.name, current_time, current_time + task.duration))
        current_time += task.duration

    return schedule

# Example usage
if __name__ == "__main__":
    tasks = [
        Task('Task 1', 4, 2),
        Task('Task 2', 2, 1),
        Task('Task 3', 6, 2),
        Task('Task 4', 8, 1)
    ]

    schedule = earliest_deadline_first(tasks)

    print("Task\tStart Time\tEnd Time")
    for task in schedule:
        print(f"{task[0]}\t{task[1]}\t\t{task[2]}")
"""
Earliest Deadline First (EDF) is a dynamic scheduling algorithm widely used in real-time systems where tasks must be completed within specific deadlines. The primary goal of EDF is to prioritize tasks based on their deadlines, ensuring that the task with the earliest deadline is executed first.
Required Libraries: The Earliest Deadline First (EDF) algorithm implementation does not require any external Python libraries beyond the standard Python library.
    heapq: This is a standard Python library used to implement the priority queue. No additional installation is required as it is part of Python's standard library.
Explanation:
Key Features of the Code:
    Heap Queue: The heapq library is used to efficiently manage the priority queue of tasks, enabling the task with the earliest deadline to be selected for execution.
    Task Representation: Each task is represented as an object with attributes such as name, deadline, and duration. The __lt__ method allows the tasks to be compared based on their deadlines, which is crucial for maintaining the correct order in the heap.
    Scheduling Mechanism: The algorithm repeatedly pops tasks from the heap, schedules them by recording their start and end times, and increments the current time accordingly. This process continues until all tasks have been scheduled.
    Time Management: The code carefully tracks the current time, ensuring that tasks are scheduled in a way that respects their deadlines. This is essential for real-time systems where timing constraints are critical.
    Efficiency: The use of a heap ensures that task selection is done in logarithmic time, making the algorithm efficient even when dealing with a large number of tasks.
"""