import subprocess

# The command you want to run
command = ["python", "script_to_run.py"]

# Number of times you want to run the script
times_to_run = 3

# Run the command a specified number of times
for i in range(times_to_run):
    # Write the iteration number to a file
    with open("iteration.txt", "w") as f:
        f.write(str(i + 1))

    # Print a message to the console
    print(f"Starting iteration {i + 1}...")

    # Run the script
    subprocess.run(command)

    # Print a message to the console
    print(f"Finished iteration {i + 1}.")
