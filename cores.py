from cores_and_tasks_batt import Core

cores = [
        Core(False, False, False, False, avg_power=4.445),  # #1
        Core(True, False, False, False, avg_power=5.07),  # #2
        Core(False, True, False, False, avg_power=4.76),  # #3
        Core(False, False, True, False, avg_power=8.586),  # #4
        Core(False, False, False, True, avg_power=7.58),  # #5
        Core(False, True, True, False, avg_power=8.901),  # #6
        Core(True, False, False, True, avg_power=8.205),  # #7
        Core(True, False, True, False, avg_power=9.211),  # #8
        Core(False, True, False, True, avg_power=7.895),  # #9
    ]