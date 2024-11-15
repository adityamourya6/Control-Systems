import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np
import random

trials = 2
incl_angle = np.pi/6*1
g=10
mass_cart=100

# gain values
K_p=300
K_d=300
K_i=10

trials_global = trials


# Generate random x-positions for falling cube
def set_x_ref(incl_angle):
    rand_h=random.uniform(0,120)#horizonal
    rand_v=random.uniform(20+120*np.tan(incl_angle)+6.5,40+120*np.tan(incl_angle)+6.5)#vertical
    return rand_h,rand_v

dt=0.02#time interval
t0=0
t_end=5
t=np.arange(t0,t_end+dt,dt) #creates an array of list with even spacing between numbers
# print(len(t))
F_g=mass_cart*g

displ_rail=np.zeros((trials,len(t)))
v_rail=np.zeros((trials,len(t)))
a_rail=np.zeros((trials,len(t)))
pos_x_train = np.zeros((trials,len(t)))
pos_y_train = np.zeros((trials,len(t)))
e = np.zeros((trials,len(t)))
e_dot = np.zeros((trials,len(t)))
e_int = np.zeros((trials,len(t)))

pos_x_cube=np.zeros((trials,len(t)))
pos_y_cube=np.zeros((trials,len(t)))
# print(pos_x_cube)
F_ga_t=F_g*np.sin(incl_angle) # Tangential conponent of gravity force
init_pos_x = 120
init_pos_y = 120*np.tan(incl_angle)+6.5
init_displ_rail = (init_pos_x**2+init_pos_y**2)**(0.5)
init_vel_rail = 0
init_a_rail = 0

init_pos_x_global = init_pos_x

trials_magn = trials
history=np.ones(trials)
print(history)
while (trials) > 0:
    pos_x_cube_ref = set_x_ref(incl_angle)[0]
    pos_y_cube_ref = set_x_ref(incl_angle)[1]
    times = trials_magn - trials
    pos_x_cube[times] = pos_x_cube_ref
    pos_y_cube[times] = pos_y_cube_ref
    win = False
    delta = 1

    #Implement PID for train position
    for i in range(1,len(t)):
        if i==1:
            displ_rail[times][0] = init_displ_rail
            v_rail[times][0] = init_vel_rail
            a_rail[times][0] = init_a_rail
            pos_x_train[times][0] = init_pos_x
            pos_y_train[times][0] = init_pos_y

        # compute horizontal error
        e[times][i-1] = pos_x_cube_ref - pos_x_train[times][i-1]


        if i>1:
            e_dot[times][i-1] = (e[times][i-1]-e[times][i-2])/dt
            e_int[times][i-1] = e_int[times][i-2]+(e[times][i-2]+e[times][i-1])/2*dt

        if i == len(t)-1:
            e[times][-1] = e[times][-2]
            e_dot[times][-1] = e_dot[times][-2]
            e_int[times][-1] = e_int[times][-2]

        F_a = K_p*e[times][i-1]+K_d*e_dot[times][i-1]+K_i*e_int[times][i-1]
        F_net = F_a+F_ga_t
        a_rail[times][i] = F_net/mass_cart
        v_rail[times][i] = v_rail[times][i-1] + (a_rail[times][i-1] + a_rail[times][i])/2*dt
        displ_rail[times][i]=displ_rail[times][i-1]+(v_rail[times][i-1]+v_rail[times][i])/2*dt

        pos_x_train[times][1] = displ_rail[times][i]*np.cos(incl_angle)

        pos_y_train[times][i] = displ_rail[times][1]*np.sin(incl_angle)+6.5

        #Try to catch
        if (pos_x_train[times][i]-5<pos_x_cube[times][0]+3 and pos_x_train[times][i]+5>pos_x_cube[times][1]-3) or win == True:
            if(pos_y_train[times][i]+3<pos_y_cube[times][i]-2 and pos_y_train[times][i]+8>pos_y_cube[times][i]+2) or win == True:
                win=True
                if delta==1:
                    change = pos_x_train[times][i]-pos_x_cube[times][i]
                    delta=0
                pos_x_cube[times][i] = pos_x_train[times][i]-change
                pos_y_cube[times][i] = pos_y_train[times][i]+5

    init_displ_rail=displ_rail[times][-1]
    init_pos_x = pos_x_train[times][-1] + v_rail[times][-1] * np.cos(incl_angle) * dt
    init_pos_y = pos_y_train[times][-1] + v_rail[times][-1] * np.sin(incl_angle) * dt
    init_vel_rail = v_rail[times][-1]
    init_a_rail = a_rail[times][-1]
    history[times]=delta
    trials = trials - 1