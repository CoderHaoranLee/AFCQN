import numpy as np
from scipy.spatial.distance import pdist


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
    if (Armor_info[path] != -1) and (index - Armor_info[path]) < 10:
        flag = 1
    return flag
def if_armor_is_none(one_armor):
    flag = 1
    if (one_armor[0] == 0) & (one_armor[1] == 0):
        flag = 0
    return flag
def distance(a,b):
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




Armor_path_Matrix, Armor_path_Matrix_info= init_path(5000)


Armor_pos = [[[20,30],[0,1]],[[20,30],[10,10]],[[18,30],[19,10]],[[18,35],[11,13]],[[0,1],[12,11]],[[16,29],[0,0]],[[16,29],[0,0]]]
dist_threthold = 10;
the_first_frame_is_none = 1

for ith in range(len(Armor_pos)):
    armor = []
    armor = Armor_pos[ith]  ##replace  get topic about the position in map: armor=[[x,y],[x,y]];


    D0 = if_armor_is_none(armor[0])
    D1 = if_armor_is_none(armor[1])
    T0 = if_path_none(Armor_path_Matrix_info,0,ith)
    T1 = if_path_none(Armor_path_Matrix_info, 1, ith)



    dist00 =  [-1,-1,-1]
    dist01 =  [-1,-1,-1]
    dist10 =  [-1,-1,-1] # T1 with D0
    dist11 =  [-1,-1,-1] # T1 with D1

    if (T0 !=0) & (T1 !=0):
        if (D0 != 0) & (D1 != 0):
            dist00 = distance(armor[0],Armor_path_Matrix[0][Armor_path_Matrix_info[0]]) # T0 with D0
            dist01 = distance(armor[1],Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D1
            dist10 = distance(armor[0],Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            dist11 = distance(armor[1],Armor_path_Matrix[1][Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist00=dist00, dist01=dist01, dist10=dist10, dist11=dist11,case=1,dist_threthold=dist_threthold);
            if ID0[1]!= 1:
                Armor_path_Matrix[ID0[0]][ith] = armor[0]
                Armor_path_Matrix_info[ID0[0]] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[ID0[0]][:],Armor_path_Matrix_info[ID0[0]])
                Armor_path_Matrix[ID0[0]][ith] = armor_mend
            if ID1[1]!= 1:
                Armor_path_Matrix[ID1[0]][ith] = armor[1]
                Armor_path_Matrix_info[ID1[0]] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[ID1[0]][:],Armor_path_Matrix_info[ID1[0]])
                Armor_path_Matrix[ID1[0]][ith] = armor_mend
        if (D0 != 0) & (D1 == 0): #detection one target
            dist00 = distance(armor[0], Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D0
            dist10 = distance(armor[0], Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            [ID0, ID1] = tracking(dist00=dist00,  dist10=dist10, case=2,dist_threthold=dist_threthold)
            if ID0[1]!= 1:
                Armor_path_Matrix[ID0[0]][ith] = armor[0]
                Armor_path_Matrix_info[ID0[0]] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[ID0[0]][:],Armor_path_Matrix_info[ID0[0]])
                Armor_path_Matrix[ID0[0]][ith] = armor_mend
            armor_mend = mend_armor(Armor_path_Matrix[ID1[0]][:],Armor_path_Matrix_info[ID1[0]])
            Armor_path_Matrix[ID1[0]][ith] = armor_mend
        if (D0 == 0) & (D1 != 0):  # detection one target
            dist01 = distance(armor[0], Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D0
            dist11 = distance(armor[0], Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            [ID0, ID1] = tracking(dist01=dist01, dist11=dist11, case=3, dist_threthold=dist_threthold)
            if ID1[1] != 1:
                Armor_path_Matrix[ID1[0]][ith] = armor[1]
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
            dist00 = distance(armor[0],Armor_path_Matrix[0][Armor_path_Matrix_info[0]]) # T0 with D0
            dist01 = distance(armor[1],Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D1
            #dist10 = distance(armor[0],Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            #dist11 = distance(armor[1],Armor_path_Matrix[1][Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist00=dist00, dist01=dist01,case=5,dist_threthold=dist_threthold);
            Armor_path_Matrix_info[1] = ith
            if ID0[0] == 0:
                Armor_path_Matrix[1][ith] = armor[1]


                if ID0[1] == -1:
                    Armor_path_Matrix[0][ith] = armor[0]
                    Armor_path_Matrix_info[0] = ith
                else:
                    armor_mend = mend_armor(Armor_path_Matrix[ID0[0]][:],Armor_path_Matrix_info[ID0[0]])
                    Armor_path_Matrix[ID0[0]][ith] = armor_mend
            if ID1[0] == 0:
                Armor_path_Matrix[1][ith] = armor[0]
                if ID1[1] == -1:
                    Armor_path_Matrix[0][ith] = armor[1]
                    Armor_path_Matrix_info[0] = ith
                else:
                    armor_mend = mend_armor(Armor_path_Matrix[0][:], Armor_path_Matrix_info[0])
                    Armor_path_Matrix[0][ith] = armor_mend

        if (D0 != 0) & (D1 == 0):
            dist00 = distance(armor[0],Armor_path_Matrix[0][Armor_path_Matrix_info[0]]) # T0 with D0
            #dist01 = distance(armor[1],Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D1
            #dist10 = distance(armor[0],Armor_path_Matrix[Armor_path_Matrix_info[1]])  # T1 with D0
            #dist11 = distance(armor[1],Armor_path_Matrix[Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist00=dist00, case=6,dist_threthold=dist_threthold);
            Armor_path_Matrix[1][ith] = armor[1]
            if ID0[1] == -1:
                Armor_path_Matrix[0][ith] = armor[0]
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
            Armor_path_Matrix[1][ith] = armor[0]
            if ID1[1] == -1:
                Armor_path_Matrix[0][ith] = armor[1]
                Armor_path_Matrix_info[0] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[0], Armor_path_Matrix_info[0])
                Armor_path_Matrix[0][ith] = armor_mend

    if (T0 == 0) & (T1 != 0):
        if (D0 != 0) & (D1 != 0):
            #dist00 = distance(armor[0],Armor_path_Matrix[0][Armor_path_Matrix_info[0]]) # T0 with D0
            #dist01 = distance(armor[1],Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D1
            dist10 = distance(armor[0],Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            dist11 = distance(armor[1],Armor_path_Matrix[1][Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist10=dist10, dist11=dist11,case=8,dist_threthold=dist_threthold);
            Armor_path_Matrix_info[0] = ith
            if ID0[0] == 1:
                Armor_path_Matrix[0][ith] = armor[1]
                if ID0[1] == -1:
                    Armor_path_Matrix[1][ith] = armor[0]
                    Armor_path_Matrix_info[1] = ith
                else:
                    armor_mend = mend_armor(Armor_path_Matrix[ID0[0]][:], Armor_path_Matrix_info[ID0[0]])
                    Armor_path_Matrix[ID0[0]][ith] = armor_mend
            if ID1[0] == 1:
                Armor_path_Matrix[0][ith] = armor[0]
                if ID1[1] == -1:
                    Armor_path_Matrix[1][ith] = armor[1]
                    Armor_path_Matrix_info[1] = ith
                else:
                    armor_mend = mend_armor(Armor_path_Matrix[1][:], Armor_path_Matrix_info[1])
                    Armor_path_Matrix[1][ith] = armor_mend

        if (D0 != 0) & (D1 == 0):
            #dist00 = distance(armor[0],Armor_path_Matrix[0][Armor_path_Matrix_info[0]]) # T0 with D0
            #dist01 = distance(armor[1],Armor_path_Matrix[0][Armor_path_Matrix_info[0]])  # T0 with D1
            dist10 = distance(armor[0],Armor_path_Matrix[1][Armor_path_Matrix_info[1]])  # T1 with D0
            #dist11 = distance(armor[1],Armor_path_Matrix[Armor_path_Matrix_info[1]]) # T1 with D1
            [ID0,ID1] = tracking(dist10=dist10, case=9,dist_threthold=dist_threthold);
            Armor_path_Matrix[0][ith] = armor[1]
            if ID0[1] == -1:
                Armor_path_Matrix[1][ith] = armor[0]
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
            Armor_path_Matrix[0][ith] = armor[0]
            if ID1[1] == -1:
                Armor_path_Matrix[1][ith] = armor[1]
                Armor_path_Matrix_info[1] = ith
            else:
                armor_mend = mend_armor(Armor_path_Matrix[1], Armor_path_Matrix_info[1])
                Armor_path_Matrix[1][ith] = armor_mend
    if (T0 == 0) and (T1 == 0):
        print armor[0], armor[1]
        Armor_path_Matrix[0][ith] = armor[0]
        Armor_path_Matrix[1][ith] = armor[1]

        if D0 != 0:
            Armor_path_Matrix_info[0] = ith
        if D1 != 0:
            Armor_path_Matrix_info[1] = ith












