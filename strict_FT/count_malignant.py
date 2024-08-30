import re

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
                    order_five_cnt_list.append(int(match.group(1)))
            if line.startswith("test 6 faults"):
                nextline = next(file, None)
                match = pattern.match(nextline)
                if match:
                    order_six_cnt_list.append(int(match.group(1)))

    print(filename)
    print("order five", sum(order_five_cnt_list))
    print("order six", sum(order_six_cnt_list))