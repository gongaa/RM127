import numpy as np
import operator
from functools import reduce
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

n = 7 
N = 2 ** n
bin_wt = lambda i: bin(i)[2:].count('1')
bit_rev = lambda t: int(bin(t)[2:].rjust(n, '0')[::-1], 2)

int2bin = lambda i: [int(c) for c in bin(i)[2:].rjust(n, '0')]
bin2int = lambda l: int(''.join(map(str, l)), 2)

def Eij(i,j):
    A = np.eye(n, dtype=int)
    A[i,j] = 1
    return A
PA = [(1,2),(6,0),(4,3),(3,6),(0,1),(2,3),(1,6)]
PB = [(2,6),(5,1),(6,0),(0,5),(4,2),(0,3),(1,4)] 
PC = [(3,1),(0,2),(2,6),(6,4),(5,0),(6,5),(3,6)] 
PD = [(5,3),(6,1),(1,2),(2,5),(4,0),(3,4),(4,5)] 
list_prod = lambda A : reduce(operator.matmul, [Eij(a[0],a[1]) for a in A], np.eye(n, dtype=int)) % 2

A1 = list_prod(PA[::-1]) % 2
A2 = list_prod(PB[::-1]) % 2
A3 = list_prod(PC[::-1]) % 2
A4 = list_prod(PD[::-1]) % 2

Ax = lambda A, i: bin2int(A @ np.array(int2bin(i)) % 2)
a1_permute = [Ax(A1, i) for i in range(N)]
a2_permute = [Ax(A2, i) for i in range(N)]
a3_permute = [Ax(A3, i) for i in range(N)]
a4_permute = [Ax(A4, i) for i in range(N)]


layout = np.zeros((2**(n//2),2**((n+1)//2)), dtype=int)
for i in range(n):
    if i % 2 == 1: # odd down
        a, b = 2**(i//2), 2**(i//2 + 1)
        layout[a:2*a,:b] = layout[:a,:b] + 2**i
    else: # even go to right
        a = 2**(i//2)
        layout[:a,a:2*a] = layout[:a,:a] + 2**i
        
layout = 127 - layout
print(layout)
     
new_layout = np.array([[a1_permute[i] for i in row] for row in layout])
print(new_layout)

# Function to draw the initial plot with dots and circles
def draw_plot(bit_i, bit_j):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(0, 8.5)
    ax.grid(True)

    for i in range(16):
        for j in range(8):
            index = layout[7-j, i]
            bin_index = int2bin(index)
            # print(f"i={i}, j={j}, index={index}")
            if bin_index[bit_i] == 1 and bin_index[bit_j] == 1:
                ax.plot(i + 0.5, j + 0.5, 'o', color='blue', zorder=10)
                # circle = plt.Circle((i + 0.5, j + 0.5), 0.25, color='blue', fill=False)
                # ax.add_patch(circle)
            elif bin_index[bit_i] == 0 and bin_index[bit_j] == 1:
                ax.plot(i + 0.5, j + 0.5, 'o', color='gold', zorder=10)
                # circle = plt.Circle((i + 0.5, j + 0.5), 0.25, color='yellow', fill=False)
                # ax.add_patch(circle)
            elif index != 0:
                ax.plot(i + 0.5, j + 0.5, 'ko')  # Black dots
    
    # blue arrows
    # if bin_i == 0, right shift by 8
    # if bin_i == 1, down shift by 4
    # if bin_i == 2, right shift by 4
    # if bin_i == 3, down shift by 2
    # if bin_i == 4, right shift by 2
    # if bin_i == 5, down shift by 1
    # if bin_i == 6, right shift by 1
    
    # Coordinates for the cubic spline
    x_start = 0.5; y_start = 7.5
    x_end = x_start; y_end = y_start
    if bit_i % 2 == 0:
        x_end = 0.5 + 2**(3 - bit_i//2)
        x_middle = (x_start + x_end) / 2
        y_middle = y_start + 0.2 * (4 - bit_i//2)
        x = [x_start, x_middle, x_end]
        for direction in [1,-1]:
            y_middle = y_start + 0.15 * (4 - bit_i//2) * direction
            color = 'blue' if direction == 1 else 'gold'
            y = [y_start, y_middle, y_end]
            cs = CubicSpline(x, y) # Interpolating cubic spline
            x_new = np.linspace(x_start, x_end, 100) # Plot cubic spline
            y_new = cs(x_new)
            ax.plot(x_new, y_new, color=color, linewidth=2, zorder=5)
            ax.arrow(x_middle, y_middle, 0.001 * direction, 0, head_width=0.3, head_length=0.2, fc=color, ec=color) # Adding double arrow heads
    else:
        y_end = y_start - 2**(2 - bit_i//2)
        y_middle = (y_start + y_end) / 2
        for direction in [1,-1]:
            x_middle = x_start - 0.4 * direction
            color = 'blue' if direction == 1 else 'gold'
            x = [x_start, x_middle, x_end] # since x is not strictly increasing, need to use parametric plines
            y = [y_start, y_middle, y_end]
            t = np.arange(len(x))
            cs_x = CubicSpline(t, x)
            cs_y = CubicSpline(t, y)
            t_fine = np.linspace(0, len(x)-1, 100)
            x_new = cs_x(t_fine)
            y_new = cs_y(t_fine)
            ax.plot(x_new, y_new, color, linewidth=2, zorder=5)
            ax.arrow(x_middle, y_middle, 0, -0.001 * direction, head_width=0.3, head_length=0.2, fc=color, ec=color) # Adding double arrow heads

    plt.show()
#     plt.savefig(f"Eij/E{bit_i}{bit_j}.png", bbox_inches='tight')


# Draw initial plot
# for pair in PA+PB+PC+PD:
#     draw_plot(*pair)
draw_plot(0,6)