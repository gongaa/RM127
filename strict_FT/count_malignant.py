import re
import math

def get_factor(input):
    # find factor for over-counting
    tuple_pattern = r'\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)'
    tuples_as_strings = re.findall(tuple_pattern, input)
    parsed_tuples = [tuple(map(int, t.strip('()').split(','))) for t in tuples_as_strings]
    factor = 1
    for x,y,z in zip(*parsed_tuples):
        assert x == y+z
        factor *= math.comb(x, y)
    return factor

# since there is one order-four malignant fault, distributed as (2,1,0,1)
# order-six faults distributed as (2,3,0,1) [split into (2,1,0,0) and (0,2,0,1)]
# and (4,1,0,1) [split into (2,1,0,0) and (2,0,0,1)] will count this order-four faults multiple time
# making the result not divisible by the factor
# they can be get rid of by making use of the fact that DEM contains 402 columns for each patch
for filename in ["state0_Z.log", "state0_X.log", "state+_X.log", "state+_Z.log"]:
    pattern = re.compile(r"^found (\d+) sets violating strict FT")
    order_five_cnt_list = []
    order_six_cnt_list = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("test 5 faults"):
                nextline = next(file, None)
                match = pattern.match(nextline)
                if match:
                    temp_cnt = int(match.group(1))
                    factor = get_factor(line)
                    if temp_cnt % factor != 0:
                        print(filename, line, nextline)
                        break
                    assert temp_cnt % factor == 0
                    order_five_cnt_list.append(temp_cnt // factor)
            if line.startswith("test 6 faults"):
                nextline = next(file, None)
                match = pattern.match(nextline)
                if match:
                    temp_cnt = int(match.group(1))
                    factor = get_factor(line)
                    if temp_cnt % factor != 0:
                        print(filename, line, nextline)
                        if temp_cnt == 1235: # (2,3,0,1)
                            order_six_cnt_list.append((temp_cnt-401)//factor)
                        elif temp_cnt == 2618: # (4,1,0,1)
                            order_six_cnt_list.append((temp_cnt-800)//factor)
                    # assert temp_cnt % factor == 0
                    else:
                        order_six_cnt_list.append(temp_cnt // factor)

    print(filename)
    print("order five", sum(order_five_cnt_list))
    print("order six", sum(order_six_cnt_list))