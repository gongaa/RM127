from PyDecoder_polar import PyDecoder_polar_SCL
import os
import glob
import re

pattern = re.compile(r"^final wt on output .* X: \[(.*?)\], Y: \[(.*?)\], Z: \[(.*?)\]")
# pattern = re.compile(r"^final wt after copying: .* X: \[(.*?)\], Y: \[(.*?)\], Z: \[(.*?)\]")
d = 7
rz = 4 if d==7 else 3
rx = 2 if d==7 else 3
decoder_X = PyDecoder_polar_SCL(rx)
decoder_Z = PyDecoder_polar_SCL(rz)
state = "plus"
    
log_files = glob.glob(os.path.join(f'logs_prep_d{d}_{state}_p0008', '*.log'))

for log_file in log_files:
    print(log_file)
    with open(log_file, 'r') as file:
        for line in file:
            if line.startswith("2 faults occurred"):
                print("ALERT! two faults occurred causing high-weight error")
            match = pattern.match(line)
            if match:
                x_list = list(filter(lambda s: s != '', match.group(1).split(',')))
                y_list = list(filter(lambda s: s != '', match.group(2).split(',')))
                z_list = list(filter(lambda s: s != '', match.group(3).split(',')))
                x_list = [int(x.strip()) for x in x_list]
                y_list = [int(y.strip()) for y in y_list]
                z_list = [int(z.strip()) for z in z_list]
                print(x_list, y_list, z_list)
                if len(x_list) > (d-1)//2: # check if wt smaller up to stabilizer
                    num_flip = decoder_X.decode(x_list + y_list)
                    class_bit = decoder_X.last_info_bit
                    # print(f"degeneracy #X flip: {num_flip}, class bit {class_bit} (can be 0/1 if in plus state)")
                    if num_flip > 2:
                        print("ALERT! X-flip weight > 2")
                        print(x_list, y_list, z_list)
                    if state == 'zero' and class_bit == 1:
                        print("ALERT! X-flip causing logical X errors")
                if len(z_list) > (d-1)//2:
                    num_flip = decoder_Z.decode(z_list + y_list)
                    class_bit = decoder_Z.last_info_bit
                    # print(f"degeneracy #Z flip: {num_flip}, class bit {class_bit} (can be 0/1 if in zero state)")
                    if num_flip > 2:
                        print("ALERT! Z-flip weight > 2")
                        print(x_list, y_list, z_list)
                    if state == 'plus' and class_bit == 1:
                        print("ALERT! Z-flip causing logical Z errors")