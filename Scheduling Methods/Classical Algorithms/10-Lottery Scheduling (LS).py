import random

class Task:
    def __init__(self, task_id, tickets):
        self.task_id = task_id
        self.tickets = tickets
        self.execution_time = 0

def lottery_scheduling(tasks, total_time):
    total_tickets = sum(task.tickets for task in tasks)
    scheduled_tasks = []

    for _ in range(total_time):
        winning_ticket = random.randint(1, total_tickets)
        current_ticket_sum = 0
        selected_task = None

        for task in tasks:
            current_ticket_sum += task.tickets
            if winning_ticket <= current_ticket_sum:
                selected_task = task
                break

        selected_task.execution_time += 1
        scheduled_tasks.append(selected_task.task_id)

    return scheduled_tasks

# Example usage
if __name__ == "__main__":
    tasks = [
        Task(task_id=1, tickets=10),
        Task(task_id=2, tickets=20),
        Task(task_id=3, tickets=30)
    ]
    
    total_time = 50
    scheduled_tasks = lottery_scheduling(tasks, total_time)

    print("Scheduled Tasks:")
    for i in range(total_time):
        print(f"Time {i}: Task {scheduled_tasks[i]}")
"""
Lottery Scheduling (LS) is a randomized algorithm for allocating resources among tasks. Each task is assigned a certain number of lottery tickets, and the scheduler selects a task by randomly drawing a ticket. The task holding the winning ticket is then allowed to execute. This method provides a probabilistic guarantee of fairness, where the number of tickets a task has determines its likelihood of being chosen.
equired Libraries: The Lottery Scheduling (LS) algorithm implementation uses Python's built-in "random" library for generating random numbers, which is essential for simulating the lottery draw.
Explanation:
Key Features of the Code:
    Task Class:
        The Task class represents each task in the system. Each task has a unique task_id, a number of tickets, and an execution_time to keep track of the total time the task has been scheduled.
    Ticket Distribution:
        The total number of tickets is the sum of tickets from all tasks. The scheduler simulates the lottery by generating a random number within the range of total tickets and assigns the execution to the task corresponding to the winning ticket.
    Task Scheduling:
        The algorithm iterates over the total time specified (total_time), selecting a task at each time unit based on the lottery system.
        Each time a task is selected, its execution_time is incremented, and the task is recorded in the scheduled_tasks list.
    Random Selection:
        The core idea of the algorithm is random selection, making it distinct from deterministic scheduling algorithms like Round Robin or Priority Scheduling.
        The probability of a task being selected is proportional to the number of tickets it holds. Hence, a task with more tickets is more likely to be chosen, but the randomness ensures that all tasks have a chance to run.
    Fairness and Flexibility:
        Lottery Scheduling is particularly useful in systems where tasks have varying priorities, but strict deterministic scheduling is not desirable.
        By adjusting the number of tickets, the system can dynamically control the priority of tasks while maintaining a level of unpredictability.
"""