# Assignment 2
# CHEM ENG/ SEP 787

# XuLiang Qi - 400347697
# Mohammad Kashif Siddiqui - 0755452

import numpy as np
import math
data = np.loadtxt('train.txt')

private = np.array([])
public = np.array([])
n = 0
m = 0
for i in range(0, len(data)):
    if data[i, 0] == 0:
        private = np.append(private, data[i, :])
        n +=1
        private = np.reshape(private, (n, 6))
    else:
        public = np.append(public, data[i, :])
        m += 1
        public = np.reshape(public, (m, 6))

PCpri = len(private) / len(data)
PCpub = len(public) / len(data)

mean_private = np.mean(private, axis = 0)

m0satV = mean_private[1]
m0satM = mean_private[2]
m0A = mean_private[3]
m0S = mean_private[4]
m0Q = mean_private[5]

std_private = np.std(private, axis = 0)

s0satV = std_private[1]
s0satM = std_private[2]
s0A = std_private[3]
s0S = std_private[4]
s0Q = std_private[5]

mean_public = np.mean(public, axis = 0)

m1satV = mean_public[1]
m1satM = mean_public[2]
m1A = mean_public[3]
m1S = mean_public[4]
m1Q = mean_public[5]

std_public = np.std(public, axis = 0)

s1satV = std_public[1]
s1satM = std_public[2]
s1A = std_public[3]
s1S = std_public[4]
s1Q = std_public[5]

a = []

def function(satV, satM, A, S, Q):
    P_satV_private = 1/(2.51*s0satV) * math.exp(-(satV - m0satV)**2 / (2*s0satV**2))
    P_satM_private = 1/(2.51*s0satM) * math.exp(-(satM - m0satM)**2 / (2*s0satM**2))
    P_A_private = 1/(2.51*s0A) * math.exp(-(A - m0A)**2 / (2*s0A**2))
    P_S_private = 1/(2.51*s0S) * math.exp(-(S - m0S)**2 / (2*s0S**2))
    P_Q_private = 1/(2.51*s0Q) * math.exp(-(Q - m0Q)**2 / (2*s0Q**2))
    
    P_satV_public = 1/(2.51*s1satV) * math.exp(-(satV - m1satV)**2 / (2*s1satV**2))
    P_satM_public = 1/(2.51*s1satM) * math.exp(-(satM - m1satM)**2 / (2*s1satM**2))
    P_A_public = 1/(2.51*s1A) * math.exp(-(A - m1A)**2 / (2*s1A**2))
    P_S_public = 1/(2.51*s1S) * math.exp(-(S - m1S)**2 / (2*s1S**2))
    P_Q_public = 1/(2.51*s1Q) * math.exp(-(Q - m1Q)**2 / (2*s1Q**2))
    
    P_private = P_satV_private*P_satM_private*P_A_private*P_S_private*P_Q_private*PCpri
    P_public = P_satV_public*P_satM_public*P_A_public*P_S_public*P_Q_public*PCpub
    
    if P_private < P_public:
        a.append(1)
    else:
        a.append(0)
    return a

test = np.loadtxt('test.txt')
for i in range(0, len(test)):
    function(test[i, 1], test[i, 2], test[i, 3], test[i, 4], test[i, 5])
    
confusion = np.ones((2, 2))
gt = test[:, 0]
TP = 0
FP = 0
FN = 0
TN = 0

for i in range(0, len(a)):
    if a[i] == 0 and gt[i] == 0:
        TP += 1
    elif a[i] == 1 and gt[i] == 1:
        TN += 1
    elif a[i] == 1 and gt[i] == 0:
        FN += 1
    elif a[i] == 0 and gt[i] == 1:
        FP += 1
confusion[0, 0] = TP
confusion[0, 1] = FP
confusion[1, 0] = FN
confusion[1, 1] = TN
print('The confusion matrix is :')
print(confusion)