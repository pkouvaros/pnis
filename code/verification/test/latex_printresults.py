import os

file_list = []
output = {}
b1 = {}
b2 = {}
box = [b1,b2]
TO_HOURS = 1
TIMEOUT = 3600 * TO_HOURS 


METHODS = {0: "Comp-Par", 1: "Comp-Seq", 2: "Mono"}
METHODSLC = {0: "comppar", 1: "compseq", 2: "mono"}

agents = [2, 3]
TIMESTEPS = 4

for formula in [0, 1]:
    print('phi {}'.format(formula+1))
    for agent_count in agents:
        for k in range(1, TIMESTEPS + 1):
            f = "results_{}_{}_{}.txt".format(str(formula), str(agent_count), str(k))
            txtfile = open(f, 'r')

            key = (agent_count, k)
            output[f] = []
            result = 'False'
            time_taken = 0.0
            got_vars, got_constrs = False, False
            maxvars, maxconstrs = (0, 0)
            for line in txtfile:
                line = line.strip()
                if "Overall result" in line:
                    output[f].append(line)
                    time_taken = float(line.split(':')[1].split(' ')[2])
                    result = "Timeout" if time_taken >= 3600 else line.split(':')[1].split(' ')[1]
                if "Max number of variables" in line and not got_vars:
                    maxvars = int(line.split()[-1])
                    got_vars = True
                if "Max number of constraints" in line and not got_constrs:
                    maxconstrs = int(line.split()[-1])
                    got_constrs = True
            box[formula][key] = (time_taken, (result, maxconstrs, maxvars))

    print("  &          h = {}   &            h = {}   ".format(*map(str, agents)))
    for k in range(1, TIMESTEPS+1):
        line = []
        for agent_count in agents:
            c = '--'
            v = '--'
            if (agent_count, k) in box[formula]:
                time, (result, c, v) = box[formula][(agent_count, k)]
                line.append('{:10.4f}s & {:7}'.format(time, result))
            else:
                line.append('--')
        # line.extend([str(c), str(v)])

        print(str(k) + ' & ' + ' & '.join(line) + "\\\\")


