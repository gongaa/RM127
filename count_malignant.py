import re

pattern = re.compile(r"^found (.*?) sets violating strict FT")
cnt_list = []
with open("order_six.log", 'r') as file:
    for line in file:
        match = pattern.match(line)
        if match:
            cnt_list.append(int(match.group(1)))

print(len(cnt_list))
print(sum(cnt_list))