# Francesco Bellezza

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from matplotlib import animation
from numpy import linalg as la
from decimal import Decimal


def compare_two_algorithms_time():
    points = 200_000
    arr_t = np.linspace(0, 1, points)
    ctrl = [np.array([0, 1]), np.array([0.5, 0.25]), np.array([0, 0.5])]
    start = time.time()
    for i in range(num):
        de_casteljau(arr_t[i], ctrl)
    de_casteljau_time = time.time() - start
    start = time.time()
    for w in range(num):
        bernstein_evaluation(ctrl, arr_t[w])
    bern_time = time.time() - start
    print("".join(["De casteljau time:", str(de_casteljau_time)]))
    print("".join(["Evaluation with bernstein time:", str(bern_time)]))


def de_casteljau(t, ctrl_points):
    n = len(ctrl_points)
    points = np.copy(ctrl_points)
    running_error = [0.5 * la.norm(np.copy(x)) for x in points]
    for i in range(1, n):
        for j in range(n - i):
            points[j] = points[j] * (1 - t) + points[j + 1] * t
            running_error[j] = (1 - t) * running_error[j] + t * running_error[j + 1] + la.norm(points[j])
    return points[0], running_error[0]


def bernstein_base(n, i, t):
    return binomial_coef(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bernstein_evaluation(points, t):
    point_evaluation = np.array([0, 0])
    n = len(points)
    for i in range(n):
        point_evaluation = point_evaluation + (points[i] * bernstein_base(n - 1, i, t))
    return point_evaluation


def bernstein_evaluation_norm(points, t):
    point_evaluation = 0
    n = len(points)
    for i in range(n):
        point_evaluation = point_evaluation + (la.norm(points[i]) * bernstein_base(n - 1, i, t))
    return point_evaluation


def condition_number(points, t):
    point_evaluation = 0
    n = len(points)
    for i in range(n):
        point_evaluation = point_evaluation + la.norm(points[i] * bernstein_base(n - 1, i, t))
    return point_evaluation


def binomial_coef(n, i):
    return math.factorial(n) / (math.factorial(i) * math.factorial(n - i))


def update_casteljau(frame):
    x = [p[0] for p in final_points][:frame]
    y = [p[1] for p in final_points][:frame]
    data = np.stack([x, y]).T
    scatter.set_offsets(data)
    return scatter


# for using in pycharm
matplotlib.use("TkAgg")

epsilon = 1e-4
# definition
# bernstein basis to calculate... and then
save = input("Do you want to save the curve animation? 1. Yes, Otherwise no:")
fig, ax = plt.subplots()
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
num = 100
arr = np.linspace(0, 1, num)
n_points = int(input("Insert number of control points:"))
gamma = (2 * n_points * epsilon) / (1 - 2 * n_points * epsilon)
control_points = fig.ginput(n_points)
control_points = [np.array(x) for x in control_points]
initial_ctrl = ax.scatter([p[0] for p in control_points], [p[1] for p in control_points], color="black", marker="o")
initial_lines = [
    ax.plot([control_points[k][0], control_points[k + 1][0]], [control_points[k][1], control_points[k + 1][1]], 'ko-')
    for k in range(len(control_points) - 1)]
final_points = []
theorem1b = list()
theorem1c = list()
theorem2a = list()
theorem2b = list()

norm_bernstein_new_point_l = list()
gamma_bern_ev_norm_l = list()
run_er_norm_l = list()
for z in range(len(arr)):
    # calculate new point
    new_point, running_er = de_casteljau(arr[z], ctrl_points=control_points)
    # calculate the same point thanks to bernstein
    bernstein_point = bernstein_evaluation(points=control_points, t=arr[z])
    final_points.append(new_point)
    # save all values for later grid...
    norm_bernstein_new_point = la.norm(bernstein_point - new_point)
    norm_bernstein_new_point_l.append(norm_bernstein_new_point)
    gamma_bern_ev_norm = gamma * bernstein_evaluation_norm(control_points, arr[z])
    gamma_bern_ev_norm_l.append(gamma_bern_ev_norm)
    run_er_norm = epsilon * (2 * running_er - la.norm(new_point))
    run_er_norm_l.append(run_er_norm)

    # check the running error Theorem 1. b
    theorem1b.append(norm_bernstein_new_point <= gamma_bern_ev_norm)
    # check the forward error... Theorem 1. c
    theorem1c.append(norm_bernstein_new_point <= run_er_norm)

    # save the interest values for avoiding repeated computations...
    norm_bernstein = la.norm(bernstein_point)
    norm_new_point = la.norm(new_point)

    if norm_new_point > gamma * condition_number(control_points, arr[z]):
        theorem2a.append(
            norm_bernstein_new_point / norm_bernstein <= gamma_bern_ev_norm / (
                norm_new_point))
    # check the running relative error Theorem 2. b
    if norm_new_point > run_er_norm:
        theorem2b.append(
            norm_bernstein_new_point / norm_bernstein <= run_er_norm / (
                norm_new_point))

scatter = ax.scatter(final_points[0][0], final_points[0][1], color="red")
ani = animation.FuncAnimation(fig=fig, func=update_casteljau, interval=1, frames=num, repeat=True)
if save == "1":
    ani.save('curve.gif', writer="pillow")
plt.show()
dict_word = {np.bool_(True): "satisfied", np.bool_(False): "not satisfied"}
print("".join(["Theorem 1.b is ", dict_word[np.array(theorem1b).all()]]))
print("".join(["Theorem 1.c is ", dict_word[np.array(theorem1b).all()]]))
print("".join(["Theorem 2.a is ", dict_word[np.array(theorem1b).all()]]))
print("".join(["Theorem 2.b is ", dict_word[np.array(theorem1b).all()]]))

print("Table of evaluation...")
print("|p(t)-p^(t)|", end="\t")
print("Gamma_{2n}*...", end="\t")
print("Runtime*...", end="\t")
print("")
for s in range(len(arr)):
    print('%.2E' % Decimal(norm_bernstein_new_point_l[s]), end="\t \t")
    print('%.2E' % Decimal(gamma_bern_ev_norm_l[s]), end="\t \t")
    print('%.2E' % Decimal(run_er_norm_l[s]), end="\t \t")
    print("")

compare_two_algorithms_time()