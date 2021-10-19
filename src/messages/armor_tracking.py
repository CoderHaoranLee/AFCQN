#! /usr/bin/env python
import rospy

from messages.msg import EnemyPosMap
from messages.msg import ArmorsPos

import numpy as np
from scipy.spatial.distance import pdist
import cv2
import matplotlib.pyplot as plt


def init_path(num):
    D1 = []
    [D1.append([]) for x in range(num)]
    #Dist_Matrix = []
    path_Matrix = []
    # for n in range(10):
    D = D1[:]
    #Dist_Matrix.append(D)
    path_Matrix_info = []
    path_Matrix_info.append(-1)
    path_Matrix_info.append(-1)
    # [Dist_Matrix_info.append([]) for xx in range(10)]
    path_Matrix.append(D)
    D1 = []
    [D1.append([]) for x in range(num)]
    D = D1[:]
    path_Matrix.append(D)
    return path_Matrix,path_Matrix_info
def if_path_none(Armor_info,path,index):
    flag = 0
    if (Armor_info[path] != -1) and (index - Armor_info[path]) < 20:
        flag = 1
    return flag
def if_armor_is_none(one_armor):
    flag = 1
    if (one_armor[0] == 0) & (one_armor[1] == 0):
        flag = 0
    # if (len(one_armor) == 0):
    #     flag = 0
    return flag
def distance(a,b):
    # print a, b
    if len(b) == 0:
        b = 0
    dist = pdist(np.vstack([( a ), (b)]))
    Dist = [dist]
    return Dist

def tracking(dist00=None, dist01=None,dist10=None,dist11=None,case=1,dist_threthold=0):#dist_t_d
    ID0 = [-1,-1] # the first item: path_index, the second item : if need amend, 1 means need; -1 didn't.
    ID1 = [-1,-1]
    if case == 1:  ## D1 , D2 T1 T2 all have target
        if (dist00[0]<=dist01[0]) & (dist00[0]<=dist10[0]) & (dist00[0] < dist_threthold): # D0 belong to T0
            ID0[0] = 0
            ID1[0] = 1
            if dist11[0]>dist_threthold:
                ID1[1] = 1
        if (dist10[0]<=dist11[0]) & (dist10[0]<=dist00[0]) & (dist10[0] < dist_threthold): # D0 belong to T1
            ID0[0] = 1
            ID1[0] = 0
            if dist01[0]>dist_threthold:
                ID1[1] = 1
        if (dist00[0]<=dist01[0]) &(dist00[0]<=dist10[0]) & (dist00[0] > dist_threthold): # D0 belong to T0 but dist too large
            ID0[1] = 1
            if dist11[0] < dist_threthold:
                ID1[0] = 1
                ID0[0] = 0
            else:
                ID1[1] = 1

        if (dist10[0]<=dist11[0]) & (dist10[0]<=dist00[0]) & (dist10[0] < dist_threthold): # D0 belong to T1 but dist too large
            ID0[1] = 1

            if dist01[0] < dist_threthold:
                ID1[0] = 0
                ID0[0] = 1
            else:
                ID1[1] = 1
        if (dist01[0]<=dist00[0]) & (dist01[0]<=dist11[0]) & (dist01[0] < dist_threthold): # D0 belong to T0
            ID0[0] = 1
            ID1[0] = 0
            if dist10[0]>dist_threthold:
                ID0[1] = 1
        if (dist11[0]<=dist10[0]) & (dist11[0]<=dist01[0]) & (dist11[0] < dist_threthold): # D0 belong to T1
            ID0[0] = 0
            ID1[0] = 1
            if dist00[0]>dist_threthold:
                ID0[1] = 1
    if case == 2: # T0 !=0 & T1 !=0 & D0 != 0 & D1 = 0:
        if (dist00[0]<=dist10[0]) & (dist00[0] <= dist_threthold): # D0 belong to T0
            ID0[0] = 0
            ID1[0] = 1
            ID1[1] = 1
        elif (dist10[0]<=dist00[0]) & (dist10[0] <= dist_threthold): # D0 belong to T1
            ID0[0] = 1
            ID1[0] = 0
            ID1[1] = 1
        else :
            ID0[0] = 1
            ID0[1] = 1
            ID1[0] = 0
            ID1[1] = 1
    if case ==3 :# T0 !=0 & T1 !=0 & D0 == 0 & D1 != 0
        if (dist01[0]<=dist11[0] ) & (dist01[0] <= dist_threthold): # D0 belong to T0
            ID1[0] = 0
            ID0[0] = 1
            ID0[1] = 1
        elif (dist11[0]<=dist01[0]) & (dist11[0] <= dist_threthold): # D0 belong to T1
            ID1[0] = 1
            ID0[0] = 0
            ID0[1] = 1
        else:
            ID0[0] = 1
            ID0[1] = 1
            ID1[0] = 0
            ID1[1] = 1
    if case ==4:# T0 !=0 & T1 !=0 & D0 == 0 & D1 == 0
        ID0[0] = 1
        ID0[1] = 1
        ID1[0] = 0
        ID1[1] = 1
    if case == 5:  ## T0 !=0 & T1 ==0 & D0 != 0 & D1 != 0
        if dist00[0]<=dist01[0]:
            ID0[0] = 0
            ID1[0] = 1
            if dist00[0] > dist_threthold: # D0 belong to T0
                ID0[1] = 1
        if dist01[0]<=dist00[0]:
            ID0[0] = 1
            ID1[0] = 0
            if dist01[0] < dist_threthold: # D0 belong to T0
                ID1[1] = 1
    if case ==6 : ## T0 !=0 & T1 ==0 & D0 != 0 & D1 == 0:
            ID0[0] = 0
            ID1[0] = 1
            if dist00[0] > dist_threthold: # D0 belong to T0
                ID0[1] = 1
    if case == 7 : ## T0 !=0 & T1 ==0 & D0 = 0 & D1 != 0:
            ID0[0] = 1
            ID1[0] = 0
            if dist00[0] > dist_threthold: # D0 belong to T0
                ID1[1] = 1
    if case == 8:  ## T0 ==0 & T1 !=0 & D0 != 0 & D1 != 0
        if dist10[0]<=dist11[0]:
            ID0[0] = 1
            ID1[0] = 0
            if dist10[0] > dist_threthold: # D0 belong to T0
                ID0[1] = 1
        if dist11[0]<=dist10[0]:
            ID0[0] = 0
            ID1[0] = 1
            if dist11[0] < dist_threthold: # D0 belong to T0
                ID1[1] = 1
    if case == 9 : ## T0 ==0 & T1 !=0 & D0 != 0 & D1 == 0:
            ID0[0] = 1
            ID1[0] = 0
            if dist00[0] > dist_threthold:  # D0 belong to T0
                ID0[1] = 1
    if case == 10 : ## T0 ==0 & T1 !=0 & D0 == 0 & D1 != 0:
            ID1[0] = 1
            ID0[0] = 0
            if dist00[0] > dist_threthold:  # D0 belong to T0
                ID1[1] = 1
    if case == 11: ## T0 == 0&
        ID1[0] = 0
        ID0[0] = 1

    return ID0,ID1

def mend_armor(Armor_path,Armor_path_info):
    armor = []
    armor = Armor_path[Armor_path_info]
    return armor


def callback(data):
    global Armor_path_Matrix
    global Armor_path_Matrix_info
    global ith
    dist_threthold = 2.
    armor = data ##replace  get topic about the position in map: armor=[[x,y],[x,y]];
    # if len(armor.armor_0) == 0:
    #     armor.armor_0 = [0, 0]
    # if len(armor.armor_1) == 0:
    #     armor.armor_1 = [0, 0]

    D0 = if_armor_is_none(armor.armor_0)
    D1 = if_armor_is_none(armor.armor_1)
    T0 = if_path_none(Armor_path_Matrix_info,0,ith)
    T1 = if_path_none(Armor_path_Matrix_info, 1, ith)

    dist00 = [-1, -1, -1]
    dist01 = [-1, -1, -1]
    dist10 = [-1, -1, -1] # T1 with D0
    dist11 = [-1, -1, -1] # T1 with D1

    if (T0 !=0) & (T1 !=0):
        if (D0 != 0) & (D1 != 0):
            dist00 = distance(armor.armor_0,Armor_path_Matrix[0][Armor_path_Matrix_info[0]]) # T0 with D0
            dist01 = distance(armor.armor_1,Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D1
            dist10 = distance(armor.armor_0,Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            dist11 = distance(armor.armor_1,Armor_path_Matrix[1][Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist00=dist00, dist01=dist01, dist10=dist10, dist11=dist11,case=1,dist_threthold=dist_threthold);
            if ID0[1]!= 1:
                Armor_path_Matrix[ID0[0]][ith] = armor.armor_0
                Armor_path_Matrix_info[ID0[0]] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[ID0[0]][:],Armor_path_Matrix_info[ID0[0]])
                Armor_path_Matrix[ID0[0]][ith] = armor_mend
            if ID1[1]!= 1:
                Armor_path_Matrix[ID1[0]][ith] = armor.armor_1
                Armor_path_Matrix_info[ID1[0]] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[ID1[0]][:],Armor_path_Matrix_info[ID1[0]])
                Armor_path_Matrix[ID1[0]][ith] = armor_mend
        if (D0 != 0) & (D1 == 0): #detection one target
            dist00 = distance(armor.armor_0, Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D0
            dist10 = distance(armor.armor_0, Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            [ID0, ID1] = tracking(dist00=dist00,  dist10=dist10, case=2,dist_threthold=dist_threthold)
            if ID0[1]!= 1:
                Armor_path_Matrix[ID0[0]][ith] = armor.armor_0
                Armor_path_Matrix_info[ID0[0]] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[ID0[0]][:],Armor_path_Matrix_info[ID0[0]])
                Armor_path_Matrix[ID0[0]][ith] = armor_mend
            armor_mend = mend_armor(Armor_path_Matrix[ID1[0]][:],Armor_path_Matrix_info[ID1[0]])
            Armor_path_Matrix[ID1[0]][ith] = armor_mend
        if (D0 == 0) & (D1 != 0):  # detection one target
            dist01 = distance(armor.armor_0, Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D0
            dist11 = distance(armor.armor_0, Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            [ID0, ID1] = tracking(dist01=dist01, dist11=dist11, case=3, dist_threthold=dist_threthold)
            if ID1[1] != 1:
                Armor_path_Matrix[ID1[0]][ith] = armor.armor_1
                Armor_path_Matrix_info[ID1[0]] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[ID1[0]][:], Armor_path_Matrix_info[ID1[0]])
                Armor_path_Matrix[ID1[0]][ith] = armor_mend
            armor_mend = mend_armor(Armor_path_Matrix[ID0[0]][:], Armor_path_Matrix_info[ID0[0]])
            Armor_path_Matrix[ID0[0]][ith] = armor_mend
        if (D0 == 0) & (D1 == 0):
            [ID0, ID1] = tracking(case=4, dist_threthold=dist_threthold)
            armor_mend = mend_armor(Armor_path_Matrix[ID0[0]][:],Armor_path_Matrix_info[ID0[0]])
            Armor_path_Matrix[ID0[0]][ith] = armor_mend
            armor_mend = mend_armor(Armor_path_Matrix[ID1[0]][:], Armor_path_Matrix_info[ID1[0]])
            Armor_path_Matrix[ID1[0]][ith] = armor_mend

    if (T0 != 0) & (T1 == 0):
        if (D0 != 0) & (D1 != 0):
            dist00 = distance(armor.armor_0,Armor_path_Matrix[0][Armor_path_Matrix_info[0]]) # T0 with D0
            dist01 = distance(armor.armor_1,Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D1
            #dist10 = distance(armor[0],Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            #dist11 = distance(armor[1],Armor_path_Matrix[1][Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist00=dist00, dist01=dist01,case=5,dist_threthold=dist_threthold);
            Armor_path_Matrix_info[1] = ith
            if ID0[0] == 0:
                Armor_path_Matrix[1][ith] = armor.armor_1

                if ID0[1] == -1:
                    Armor_path_Matrix[0][ith] = armor.armor_1
                    Armor_path_Matrix_info[0] = ith
                else:
                    armor_mend = mend_armor(Armor_path_Matrix[ID0[0]][:],Armor_path_Matrix_info[ID0[0]])
                    Armor_path_Matrix[ID0[0]][ith] = armor_mend
            if ID1[0] == 0:
                Armor_path_Matrix[1][ith] = armor.armor_0
                if ID1[1] == -1:
                    Armor_path_Matrix[0][ith] = armor.armor_1
                    Armor_path_Matrix_info[0] = ith
                else:
                    armor_mend = mend_armor(Armor_path_Matrix[0][:], Armor_path_Matrix_info[0])
                    Armor_path_Matrix[0][ith] = armor_mend

        if (D0 != 0) & (D1 == 0):
            dist00 = distance(armor.armor_0,Armor_path_Matrix[0][Armor_path_Matrix_info[0]]) # T0 with D0
            #dist01 = distance(armor[1],Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D1
            #dist10 = distance(armor[0],Armor_path_Matrix[Armor_path_Matrix_info[1]])  # T1 with D0
            #dist11 = distance(armor[1],Armor_path_Matrix[Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist00=dist00, case=6,dist_threthold=dist_threthold);
            Armor_path_Matrix[1][ith] = armor.armor_1
            if ID0[1] == -1:
                Armor_path_Matrix[0][ith] = armor.armor_0
                Armor_path_Matrix_info[0] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[0], Armor_path_Matrix_info[0])
                Armor_path_Matrix[0][ith] = armor_mend
        if (D0 == 0) & (D1 != 0):
            #dist00 = distance(armor[0],Armor_path_Matrix[0][Armor_path_Matrix_info[0]]) # T0 with D0
            dist01 = distance(armor[1],Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D1
            #dist10 = distance(armor[0],Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            #dist11 = distance(armor[1],Armor_path_Matrix[1][Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist01=dist01, case=7,dist_threthold=dist_threthold);
            Armor_path_Matrix[1][ith] = armor.armor_0
            if ID1[1] == -1:
                Armor_path_Matrix[0][ith] = armor.armor_1
                Armor_path_Matrix_info[0] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[0], Armor_path_Matrix_info[0])
                Armor_path_Matrix[0][ith] = armor_mend

    if (T0 == 0) & (T1 != 0):
        if (D0 != 0) & (D1 != 0):
            #dist00 = distance(armor[0],Armor_path_Matrix[0][Armor_path_Matrix_info[0]]) # T0 with D0
            #dist01 = distance(armor[1],Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D1
            dist10 = distance(armor.armor_0,Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            dist11 = distance(armor.armor_1,Armor_path_Matrix[1][Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist10=dist10, dist11=dist11,case=8,dist_threthold=dist_threthold);
            Armor_path_Matrix_info[0] = ith
            if ID0[0] == 1:
                Armor_path_Matrix[0][ith] = armor.armor_1
                if ID0[1] == -1:
                    Armor_path_Matrix[1][ith] = armor.armor_0
                    Armor_path_Matrix_info[1] = ith
                else:
                    armor_mend = mend_armor(Armor_path_Matrix[ID0[0]][:], Armor_path_Matrix_info[ID0[0]])
                    Armor_path_Matrix[ID0[0]][ith] = armor_mend
            if ID1[0] == 1:
                Armor_path_Matrix[0][ith] = armor.armor_0
                if ID1[1] == -1:
                    Armor_path_Matrix[1][ith] = armor.armor_1
                    Armor_path_Matrix_info[1] = ith
                else:
                    armor_mend = mend_armor(Armor_path_Matrix[1][:], Armor_path_Matrix_info[1])
                    Armor_path_Matrix[1][ith] = armor_mend

        if (D0 != 0) & (D1 == 0):
            #dist00 = distance(armor[0],Armor_path_Matrix[0][Armor_path_Matrix_info[0]]) # T0 with D0
            #dist01 = distance(armor[1],Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D1
            dist10 = distance(armor.armor_0,Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            #dist11 = distance(armor[1],Armor_path_Matrix[Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist10=dist10, case=9,dist_threthold=dist_threthold);
            Armor_path_Matrix[0][ith] = armor.armor_1
            if ID0[1] == -1:
                Armor_path_Matrix[1][ith] = armor.armor_0
                Armor_path_Matrix_info[1] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[1], Armor_path_Matrix_info[1])
                Armor_path_Matrix[1][ith] = armor_mend
        if (D0 == 0) & (D1 != 0):
            #dist00 = distance(armor[0],Armor_path_Matrix[Armor_path_Matrix_info[0]]) # T0 with D0
            #dist01 = distance(armor[1],Armor_path_Matrix[Armor_path_Matrix_info[0]])  # T0 with D1
            #dist10 = distance(armor[0],Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            dist11 = distance(armor[1],Armor_path_Matrix[1][Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist11=dist11, case=10,dist_threthold=dist_threthold);
            Armor_path_Matrix[0][ith] = armor.armor_0
            if ID1[1] == -1:
                Armor_path_Matrix[1][ith] = armor.armor_1
                Armor_path_Matrix_info[1] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[1], Armor_path_Matrix_info[1])
                Armor_path_Matrix[1][ith] = armor_mend
    if (T0 == 0) and (T1 == 0):
        print armor.armor_0, armor.armor_1
        Armor_path_Matrix[0][ith] = armor.armor_0
        Armor_path_Matrix[1][ith] = armor.armor_1

        if D0 != 0:
            Armor_path_Matrix_info[0] = ith
        if D1 != 0:
            Armor_path_Matrix_info[1] = ith
    ith += 1

def armor_tracking():
    global Armor_path_Matrix
    global Armor_path_Matrix_info
    global ith

    rospy.init_node("armor_tracking", anonymous=True)
    topic_name = "armor_detection"
    rospy.loginfo("Visualizing published on '%s'.", topic_name)

    rospy.Subscriber("armors_pos", ArmorsPos, callback, queue_size = 1) # define feedback topic here!
    r = rospy.Rate(5)

    map_file = "/home/drl/ros_codes/RoboRTS/tools/map/icra.pgm"
    im = cv2.imread(map_file)
    while not rospy.is_shutdown():
        # ith += 1
        print(ith)
        armor_path1 = Armor_path_Matrix[0][ith-1]
        armor_path2 = Armor_path_Matrix[1][ith-1]
        print armor_path1

        path = []
        path_2 = []
        if ith > 10:
            for i in range(10):
                if len(Armor_path_Matrix[0][ith-i])>0:
                    x, y = Armor_path_Matrix[0][ith-i]
                    path.append([x, y])
                if len(Armor_path_Matrix[1][ith-i])>0:
                    x, y = Armor_path_Matrix[1][ith-i]
                    path_2.append([x, y])
        else:
            for i in range(ith):
                if len(Armor_path_Matrix[0][ith-i])>0:
                    x, y = Armor_path_Matrix[0][ith-i]
                    path.append([x, y])
                if len(Armor_path_Matrix[1][ith-i])>0:
                    x, y = Armor_path_Matrix[1][ith-i]
                    path_2.append([x, y])

        # plot path, path_2
        print path
        print path_2
        plt.imshow(im, extent=[-1.0, 7.0, -0.8, 4.2])
        if len(path) > 0:
            path = np.asarray(path)
            plt.plot(path[:, 0], path[:, 1], 'r')
        if len(path_2) > 0:
            path_2 = np.asarray(path_2)
            plt.plot(path_2[:, 0], path_2[:, 1], 'g')
        plt.draw()
        plt.pause(0.01)
        plt.clf()
        r.sleep()


if __name__ == '__main__':
    try:
        Armor_path_Matrix, Armor_path_Matrix_info = init_path(5000)
        ith = 0
        armor_tracking()
    except rospy.ROSInterruptException:
        pass