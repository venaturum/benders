# usage: python benders_basic_bounded.py --iters=100 MIPFocus=3 /path/to/model.mps.bz2

import scipy.sparse as ss
import gurobipy as gp
import numpy as np
import argparse

def run(filename, max_iters, **master_params):
    m = gp.read(filename)
    vars = m.getVars()
    for v in vars:
        if v.vtype == "B":
            raise ValueError("Not built for binaries")
        if v.lb < 0:
            raise ValueError("Not built for variables with negative lower bounds")
        if v.lb > 0:
            m.addConstr(v >= v.lb)
            v.lb = 0
        if v.ub != float("inf"):
            m.addConstr(v <= v.ub)
            v.ub = float("inf")
    m.update()
    i_vars  = [v for v in vars if v.vtype!="C"]
    c_vars = [v for v in vars if v.vtype=="C"]

    c_1 = np.array([v.obj for v in i_vars])
    c_2 = np.array([v.obj for v in c_vars])
    b = np.array(m.getAttr("RHS", m.getConstrs()))
    sense = np.array(m.getAttr("Sense", m.getConstrs()))

    A = m.getA()

    eqinds = []
    for i, s in enumerate(sense):
        if s == "<":
            b[i] = -b[i]
            A[i,:] = -A[i,:]
        elif s == "=":
            eqinds.append(i)


    A = ss.vstack((A, -A[eqinds,:]))
    b = ss.hstack((b, -b[eqinds]))

    A_1 = A[:, [v.index for v in i_vars]]
    A_2 = A[:, [v.index for v in c_vars]]

    m.dispose()

    master = gp.Model()
    for k,v in master_params.items():
        master.setParam(k,v)
    master.params.PreCrush=1
    master.params.OutputFlag=0
    x = master.addMVar(len(i_vars), vtype="I")
    theta = master.addVar(lb = -100000)
    master.setObjective(c_1@x + theta)


    def solve_subproblem(xval):
        print("solving subproblem")
        subproblem = gp.Model()
        subproblem.params.OutputFlag=0
        y = subproblem.addMVar(len(c_vars))
        con=subproblem.addConstr(A_2@y >= b - A_1@xval)
        subproblem.setObjective(c_2@y)
        subproblem.params.infUnbdInfo=1
        subproblem.optimize()
        print(subproblem.status)
        if 3 <= subproblem.status <= 4:
            objval = None
            ray = np.array(subproblem.farkasdual)
            if any(ray < 0):
                ray = -ray
        else:
            objval = subproblem.ObjVal
            ray = None
        duals = np.array(con.Pi).flatten()
        subproblem.dispose()
        return objval, ray, duals

    def print_iteration(k, *args):
        print(k, args)

    MAXIMUM_ITERATIONS = max_iters
    ABSOLUTE_OPTIMALITY_GAP = 1e-9

    print("Iteration  Lower Bound  Upper Bound          Gap")
    for k in range(MAXIMUM_ITERATIONS):
        master.optimize()
        lower_bound = master.ObjVal
        x_k = x.X
        sub_objval, ray, sub_duals = solve_subproblem(x_k)
        if sub_objval is not None:
            upper_bound = c_1@x_k + sub_objval
            gap = (upper_bound - lower_bound) / upper_bound
            print_iteration(k, lower_bound, upper_bound, gap)
            if gap < ABSOLUTE_OPTIMALITY_GAP:
                print("Terminating with the optimal solution")
                break
            master.addConstr(theta >= (b-A_1@x)@sub_duals)
        else:
            print_iteration(k, "adding feasiblity cut")
            cut = master.addConstr((b-A_1@x)@ray <= 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benders.py",
        description="A simple benders implementation",

    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="The maximum number of iterations",
    )
    args, remaining_args = parser.parse_known_args()
    model = remaining_args.pop(-1)

    def process_arg(s):
        key, val = s.split("=")
        try:
            return (key, int(val))
        except ValueError:
            pass

        try:
            return (key, float(val))
        except ValueError:
            return (key, val)
    
    kwargs = dict(process_arg(s) for s in remaining_args)
    run(model, args.iters, **kwargs)

