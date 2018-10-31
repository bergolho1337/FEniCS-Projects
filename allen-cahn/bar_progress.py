import sys


def print_progress (iter,max_iter):
    progress = float(iter / max_iter)
    barWidth = 100

    sys.stdout.write("\r[")
    pos = int(barWidth * progress)
    for i in range(barWidth):
        if (i < pos):
            sys.stdout.write("=")
        elif (i == pos):
            sys.stdout.write(">")
        else: 
            sys.stdout.write(" ")
    sys.stdout.write("] %d %% \r" % (int(progress * 100.0)))
    sys.stdout.flush()

max_iter = 100000.0
iter = 0.0

while (iter <= max_iter):
    print_progress(iter,max_iter)
    iter = iter + 1
print()