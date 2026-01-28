import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
from structured_test import extract_6x6_binary_arrays

state = "zero"

pat = re.compile(r"number of order-three faults leading to weight-four residual error: (\d+); weight-eight residual error: (\d+)")
pat_order_4 = re.compile(r"number of order-four faults leading to weight > four residual error: (\d+)")

first_test_total_match = 0
first_test_cnt = []
first_test_order_four_cnt = []
with open(f"{state}_first_test.log", 'r') as file:
    for line in file:
        if line.startswith("number of order-three faults leading to"): 
            match = pat.search(line)
            if match:
                tup = tuple(map(int, match.groups()))
                first_test_total_match += 1
                first_test_cnt.append((int(match.group(1)), int(match.group(2))))

        if line.startswith("number of order-four faults leading to"):
            match = pat_order_4.search(line)
            if match:
                first_test_order_four_cnt.append(int(match.group(1)))


second_test_total_match = 0
second_test_cnt = []
second_test_order_four_cnt = []
with open(f"{state}_second_test.log", 'r') as file:
    for line in file:
        if line.startswith("number of order-three faults leading to"): 
            match = pat.search(line)
            if match:
                tup = tuple(map(int, match.groups()))
                second_test_total_match += 1
                second_test_cnt.append((int(match.group(1)), int(match.group(2))))

        if line.startswith("number of order-four faults leading to"):
            match = pat_order_4.search(line)
            if match:
                second_test_order_four_cnt.append(int(match.group(1)))


print(f"#match in first test: {first_test_total_match}, #match in second test: {second_test_total_match}")
# assert first_test_total_match == second_test_total_match

first_test_num_wt8 = []
second_test_num_wt8 = []

for i in range(3000):
    tup1 = first_test_cnt[i]
    tup2 = first_test_cnt[i]
    if tup1[1] == 0 and tup2[1] == 0:
        print(f"index {i}", tup1, tup2)
    first_test_num_wt8.append(tup1[1])
    second_test_num_wt8.append(tup2[1])

print(min(first_test_num_wt8))
print(min(second_test_num_wt8))
print(f"order four cnt for index 58: {first_test_order_four_cnt[58]}, {second_test_order_four_cnt[58]}")
print(f"order four cnt for index 1052: {first_test_order_four_cnt[1052]}, {second_test_order_four_cnt[1052]}")


idx = 1052
with open(f"{state}_triplet_perm.log") as f:
    text = f.read()
    all_arrays = extract_6x6_binary_arrays(text)

A2, A3, A4 = all_arrays[idx*3:idx*3+3]

def print_arr(arr):
    for row in arr:
        print([int(a) for a in row], end=",\n")

print("A2")
print_arr(A2)
print("A3")
print_arr(A3)
print("A4")
print_arr(A4)