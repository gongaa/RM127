from PyDecoder_polar import PyDecoder_polar_SCL
import os
import glob
import re

num_fault_pattern = re.compile(r"^(\d+) faults occurred")
pattern = re.compile(r"^final wt on output .* X: \[(.*?)\], Y: \[(.*?)\], Z: \[(.*?)\]")
# pattern = re.compile(r"^final wt after copying: .* X: \[(.*?)\], Y: \[(.*?)\], Z: \[(.*?)\]")
d = 15
rz = 4 if d==7 else 3
rx = 2 if d==7 else 3
decoder_X = PyDecoder_polar_SCL(rx)
decoder_Z = PyDecoder_polar_SCL(rz)
state = "zero"
    
log_files = glob.glob(os.path.join('logs_prep_SPAM_equal_CNOT', f"d{d}_{state}" , "*", '*.log'))

for log_file in log_files:
    # print(log_file)
    with open(log_file, 'r') as file:
        num_fault = None
        for line in file:
            if line.startswith("2 faults occurred"):
                print(f"ALERT! two faults occurred causing high-weight error, log file {log_file}")
            num_fault_match = num_fault_pattern.match(line)
            if num_fault_match:
                num_fault = int(num_fault_match.group(1))
            match = pattern.match(line)
            if match:
                x_list = list(filter(lambda s: s != '', match.group(1).split(',')))
                y_list = list(filter(lambda s: s != '', match.group(2).split(',')))
                z_list = list(filter(lambda s: s != '', match.group(3).split(',')))
                x_list = [int(x.strip()) for x in x_list]
                y_list = [int(y.strip()) for y in y_list]
                z_list = [int(z.strip()) for z in z_list]
                # print(x_list, y_list, z_list)
                new_x_list = x_list + y_list
                if len(new_x_list) > (d-1)//2: # check if wt smaller up to stabilizer
                    num_flip = decoder_X.decode(new_x_list)
                    class_bit = decoder_X.last_info_bit
                    new_x_list = decoder_X.correction
                    # print(f"degeneracy #X flip: {num_flip}, class bit {class_bit} (can be 0/1 if in plus state)")
                    if num_flip > 2:
                        print(f"ALERT! X-flip weight > 2, #flip={num_flip}, #fault={num_fault}, log file {log_file}")
                        print(x_list, y_list, z_list)
                    if state == 'zero' and class_bit == 1:
                        print(f"ALERT! X-flip causing logical X errors, log file {log_file}")
                new_z_list = z_list + y_list
                if len(new_z_list) > (d-1)//2:
                    num_flip = decoder_Z.decode(new_z_list)
                    class_bit = decoder_Z.last_info_bit
                    new_z_list = decoder_Z.correction
                    # print(f"degeneracy #Z flip: {num_flip}, class bit {class_bit} (can be 0/1 if in zero state)")
                    if num_flip > 2:
                        print(f"ALERT! Z-flip weight > 2, #flip={num_flip}, #fault={num_fault}, log file {log_file}")
                        print(x_list, y_list, z_list)
                    if state == 'plus' and class_bit == 1:
                        print(f"ALERT! Z-flip causing logical Z errors, log file {log_file}")
                total_wt = len(set(new_x_list) | set(new_z_list))
                if total_wt > 2:
                    print(f"ALERT: total weight {total_wt}, num fault {num_fault}, log file {log_file}")