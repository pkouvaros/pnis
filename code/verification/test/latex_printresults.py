import os

file_list = []
output = {}
b1 = {}
b2 = {}
box = [b1,b2]
TO_HOURS = 1
TIMEOUT = 3600 * TO_HOURS 


agents = [2, 3]
TEMP_DEPTH = 6

def experiments_universal():

    for method in [0]:
        print(f"method {method}")
        for formula in [1]:
            print('phi {}'.format(formula+1))
            for agent_count in agents:
                for k in range(1, TEMP_DEPTH + 1):
                    f = "results_m{}_f{}_a{}_k{}.txt".format(str(method), str(formula), str(agent_count), str(k))
                    if os.path.exists(f):
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

            print(" k & " + " & ".join(["\multicolumn{1}{c}{$k = "+f"{k}"+"$}" for k in range(1, TEMP_DEPTH + 1)]) + "\\\\")
            for agent_count in agents:
                line = []
                for k in range(1, TEMP_DEPTH + 1):
                    c = '--'
                    v = '--'
                    if (agent_count, k) in box[formula]:
                        time, (result, c, v) = box[formula][(agent_count, k)]
                        cellcolor = "         "
                        if result == 'False':
                            cellcolor = "\graycell"
                        line.append('{} {:7.2f}s'.format(cellcolor, time))
                    else:
                        line.append('--')
                # line.extend([str(c), str(v)])

                print(str(agent_count) + ' & ' + ' & '.join(line) + "\\\\")


THRESHOLD = 6

TIME_STEPS = [2,3,4,5]

def experiments_emergence_existential_property():
    for formula in [0]:
        # print('phi {}'.format(formula + 1))
        for agent_count in [2]:
            for k in TIME_STEPS:
                for n in range(agent_count, THRESHOLD + 1):
                    f = f"results_f{formula}_k{k}_{agent_count}_{n}.txt"
                    if os.path.exists(f):
                        txtfile = open(f, 'r')

                        key = (k, agent_count, n)
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

            print(" k &" + " & ".join(["\multicolumn{2}{c}{$t = "+f"{n}"+"$}" for n in range(agent_count, THRESHOLD + 1)]) + "\\\\")
            for k in TIME_STEPS:
                line = []
                for n in range(agent_count, THRESHOLD + 1):
                    c = '--'
                    v = '--'
                    if (k, agent_count, n) in box[formula]:
                        time, (result, c, v) = box[formula][(k, agent_count, n)]
                        cellcolor = "         "
                        if result == 'False':
                            cellcolor = "\graycell"
                        line.append('{} {:7.2f}s'.format(cellcolor, time))
                    else:
                        line.append('--')
                # line.extend([str(c), str(v)])

                print(" & " + str(k) + ' & ' + ' & '.join(line) + "\\\\")

# experiments_emergence_existential_property()

experiments_universal()