import os

file_list = []
output = {}
b1 = {}
b2 = {}
box = [b1,b2]
TO_HOURS = 1
TIMEOUT = 3600 * TO_HOURS 


agents = [1, 2, 3]
THRESHOLD = 6

def experiments1():

    for method in [0]:
        print(f"method {method}")
        for formula in [0, 1]:
            print('phi {}'.format(formula+1))
            for agent_count in agents:
                for k in range(1, THRESHOLD + 1):
                    f = "results_{}_{}_{}_{}.txt".format(str(method), str(formula), str(agent_count), str(k))
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

            print("  &          m = {}   &            m = {}      &           m = {} ".format(*map(str, agents)))
            for k in range(1, THRESHOLD + 1):
                line = []
                for agent_count in agents:
                    c = '--'
                    v = '--'
                    if (agent_count, k) in box[formula]:
                        time, (result, c, v) = box[formula][(agent_count, k)]
                        line.append('{:10.2f}s & {:7}'.format(time, result))
                    else:
                        line.append('--')
                # line.extend([str(c), str(v)])

                print(" & " + str(k) + ' & ' + ' & '.join(line) + "\\\\")


THRESHOLD = 8


def experiments_emergence_existential_property():
    for k in [5,6]:
        print(f"temporal depth {k}")
        for formula in [0]:
            print('phi {}'.format(formula + 1))
            for agent_count in [2]:
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

            print("  &" + " & ".join([f"t = {n}" for n in range(agent_count, THRESHOLD + 1)]) + "\\\\")
            for agent_count in [2]:
                line = []
                for n in range(agent_count, THRESHOLD + 1):
                    c = '--'
                    v = '--'
                    if (k, agent_count, n) in box[formula]:
                        time, (result, c, v) = box[formula][(k, agent_count, n)]
                        line.append('{:10.2f}s & {:7}'.format(time, result))
                    else:
                        line.append('--')
                # line.extend([str(c), str(v)])

                print(" & " + str(k) + ' & ' + ' & '.join(line) + "\\\\")

experiments_emergence_existential_property()