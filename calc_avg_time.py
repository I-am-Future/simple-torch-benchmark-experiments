import re

def extract_duration(filename):
    ''' Extract `duration` from log file. '''
    with open(filename, "r") as f:
        logstr = f.read()
    durations = re.findall(r"duration=(\d+\.\d+)s", logstr)
    return [float(x) for x in durations]

def average_time(filename):
    ''' Calculate average time. '''
    durations = extract_duration(filename)
    return sum(durations) / len(durations)


if __name__ == '__main__':
    print(average_time("results/Win11.log"))
    print(average_time("results/WSL2.log"))
    print(average_time("results/Docker.log"))
