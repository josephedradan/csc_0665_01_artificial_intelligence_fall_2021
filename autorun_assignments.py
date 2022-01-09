"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 1/9/2022

Purpose:
    Run all the assignments at the same time

Details:

Description:

Notes:
    The execution of the command will be shown in the title of the cmd shell title

IMPORTANT NOTES:
    ONLY WORKS ON WINDOWS

Explanation:

Tags:

Reference:
    call cmd from shortcut with custom title
        Notes:
            start "Title of Window" cmd.exe
            OR
            start cmd.exe /k "Title of Window"

        Reference:
            https://stackoverflow.com/questions/8574497/call-cmd-from-shortcut-with-custom-title
"""
import os
import subprocess

LIST_PATH_RELATIVE_ASSIGNMENT = [
    r"assignment_0/assignment_0",
    r"assignment_1/proj1-search-python3",
    r"assignment_2/multiagent",
    r"assignment_3/reinforcement",
    r"assignment_5/machinelearning",
]

"""
This command will open a new command line prompt and run the assignment, you must close the window afters

Notes:
    cmd 
        /k 
            Run Command and then return to the CMD prompt.
        /c 
            Run Command and then terminate (You won't see the score)
"""

# Version 1
# COMMAND_RUN = "start cmd.exe /k \"title {} && py -3.6 autograder.py\""  # ONLY WORKS ON WINDOWS

# Version 2
COMMAND_RUN = "start \"{}\" cmd.exe /k py -3.6 autograder.py"  # ONLY WORKS ON WINDOWS


def main():
    cwd = os.getcwd()

    # Loop and run all the assignments as the same time
    for path_relative in LIST_PATH_RELATIVE_ASSIGNMENT:
        path_full_assignment = os.path.join(cwd, path_relative)
        # print(path_full_assignment)

        command = COMMAND_RUN.format("{}".format(path_relative))
        print(command)
        # Execute command in corresponding directory
        subprocess.Popen(command,
                         cwd=path_full_assignment,
                         shell=True  # This must be true for the commands to work
                         )


if __name__ == '__main__':
    main()
