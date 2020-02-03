# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 14:21:31 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import math


#step1 input the file
raw = pd.read_csv("C:/Users/Lenovo/Desktop/2020 Winter/APM466/Assignment 1/Bonds Price Records_Final.csv")
raw = raw.dropna(axis = 1, how = "all")
raw = raw.dropna(axis = 0, how = "all")
writer = pd.ExcelWriter('C:/Users/Lenovo/Desktop/2020 Winter/APM466/Assignment 1/yield_output.xlsx')

par = 100
years_list = []
coupon_list = []
for i in range(10):
    years = round((pd.to_datetime(raw.iloc[i, 4])-pd.to_datetime('2020/01/15')).days / 365, 3)
    years_list.append(years)
raw["years"] = years_list

def T_calculation(maturity_date, today):
    day = (pd.to_datetime(maturity_date) - pd.to_datetime(today)).days
    T = day//180 + 1
    return T

def bond_ytm(price, par, T, coup, freq=2, guess=0.05):
    """
    price: current price of bond
    par: face value
    T: number of coupon payment left
    coup: coupon payment rate
    """
    freq = float(freq)
#    print(T)
    periods = T
#    print("period for bond"+str(i)+" bond is", periods)
    coupon = coup/100.*par/freq
    dt = [(i+1)/freq for i in range(int(periods))]
#    print("dt for bond"+str(i)+" bond is", dt)
    ytm_func = lambda y: sum([coupon/(1+y/freq)**(freq*t) for t in dt]) + par/(1+y/freq)**(T) - price
        
    return optimize.newton(ytm_func, guess) *100

def zerocoupon_bond(price, today, maturity_date):
    time_maturity = (pd.to_datetime(maturity_date)-pd.to_datetime(today)).days / 365
    notional = 100
    zcb_rate = -math.log(price/notional)/time_maturity
    return zcb_rate * 100
    
def spot_rate(raw, spot_data, date, bond_index):
    
    #dirty price:
    N = (pd.to_datetime(raw.iloc[bond_index, 4]) - pd.to_datetime(today)).days % 180
    accrued_interest = N / 365 * float(raw.iloc[bond_index, 1][:4])
#    print(accrued_interest)
    dirty_price = accrued_interest + raw.iloc[bond_index, date+5]
    #t1 part
    if bond_index == 1:
        t1 = raw.iloc[0, 15]
    else:
        t1 = (pd.to_datetime(raw.iloc[bond_index-1, 4]) - pd.to_datetime(raw.iloc[bond_index-2, 4])).days / 365
    cf_t1 = float(raw.iloc[bond_index-1, 1][:4])/2
    rate_t1 = math.exp((spot_data.iloc[bond_index-1, date])*t1)
    #t2
    if bond_index == 1:
        t2 = raw.iloc[1, 15]
    else:
        t2 = (pd.to_datetime(raw.iloc[bond_index, 4]) - pd.to_datetime(raw.iloc[bond_index-2, 4])).days / 365
    cf_t2 = 100 + float(raw.iloc[bond_index, 1][:4])/2
    #rate for t2
    r_t2 = math.log(cf_t2 / (dirty_price - (cf_t1/rate_t1)))/t2    
    return r_t2 * 100

def forward_rate(spot_data, forward_data):
    #Consider bonds mature in March only 
    for date in range(0, 10):
        base_spot = spot_data.iloc[0, date]/100
        for bond_index in range(0, 4):
            forward = ((1 + spot_data.iloc[(bond_index+1)*2,date]/100)**(bond_index+2)/(1+base_spot))**(1/(bond_index+1))-1
            forward_data.iloc[bond_index, date] = forward
    return forward_data




    
if __name__ == "__main__":
    
    ytm_data = pd.DataFrame(columns = raw.columns[5:15], index = raw["years"]) #ytm data frame
    spot_data = pd.DataFrame(columns = raw.columns[5:15], index = raw["years"]) #spot rate data frame
    forward_data = pd.DataFrame(columns = raw.columns[5:15], index = ["1yr-1yr","1yr-2yr", "1yr-3yr", "1yr-4yr"]) #forward rate data frame
    zcb_list = []
    for date in range(10):
        today = raw.columns[date+5]
        spot_zcb = zerocoupon_bond(raw.iloc[0, date+5], today,raw.iloc[0,4])
        zcb_list.append(spot_zcb)
    spot_data.iloc[0,:] = zcb_list        
    for date in range(10):
        today = raw.columns[date+5]
        for bond_index in range(len(raw)):
            #ytm
            T = T_calculation(raw.iloc[bond_index, 4], today)
            ytm = bond_ytm(raw.iloc[bond_index, date+5], 100, T, float(raw.iloc[bond_index,1][0:4]))
            ytm_data.iloc[bond_index, date] = ytm
            #spot rate
            if bond_index != 0:
                spot = spot_rate(raw, spot_data, date, bond_index)
                spot_data.iloc[bond_index, date] = abs(spot)
    forward_rate(spot_data, forward_data) 
    ytm_data.to_excel(writer, sheet_name = 'yield to maturity')
    spot_data.to_excel(writer, sheet_name = 'spot rate')
    forward_data.to_excel(writer, sheet_name = 'forward rate')
    writer.save()

#Covariance for yield    
    cov_mat1 = np.zeros([9, 5])
    for i in range(0, 10, 2):
        i_new = int(i/2)
        for j in range(1, 10):
            X_ij = math.log((ytm_data.iloc[i, j]) / (ytm_data.iloc[i,j-1]))
            cov_mat1[j-1,i_new] = X_ij
    ytm_cov = np.cov(cov_mat1.T)
    eig_val1, eig_vec1 = np.linalg.eig(ytm_cov)
    print(eig_val1[0]/sum(eig_val1))


#Covariance for forward
    cov_mat2 = np.zeros([9, 4])
    for i in range(0,4):
        for j in range(1, 10):
            X_ij = math.log((forward_data.iloc[i, j]) / (forward_data.iloc[i, j-1]))
            cov_mat2[j-1, i] = X_ij
    forward_cov = np.cov(cov_mat2.T)
    eig_val2, eig_vec2 = np.linalg.eig(forward_cov)
    print(eig_val2[0]/sum(eig_val2))

    
    #plot of ytm        
    plt.figure(figsize=(15,5), dpi= 80)
    fig = plt.subplot(1, 1, 1)
    for date in range(10):
        fig.plot(raw["years"], ytm_data.iloc[:,date], label = raw.columns[date])
    plt.xlim(0, 5)
    plt.xlabel("Time to Maturity(Years)")
    plt.ylabel("Yield to Maturity(%)")
    plt.title("YTM curve")
    fig.legend(ytm_data.columns)
    plt.grid(True, axis = 'both')
    
    #plot of spot rate       
    plt.figure(figsize=(15,5), dpi= 80)
    fig = plt.subplot(1, 1, 1)
    for date in range(10):
        fig.plot(raw["years"], spot_data.iloc[:,date], label = raw.columns[date])
    plt.xlim(0, 5)
    plt.xlabel("Time to Maturity(Years)")
    plt.ylabel("Spot rate(%)")
    plt.title("Spot rate curve")
    fig.legend(spot_data.columns)
    plt.grid(True, axis = 'both')
    
    #plot of forward rate       
    plt.figure(figsize=(15,5), dpi= 80)
    fig = plt.subplot(1, 1, 1)
    for date in range(10):
        fig.plot(forward_data.iloc[:,date])
    plt.xlabel("Time to Maturity(Years)")
    plt.ylabel("Forward rate(%)")
    plt.title("Forward rate curve")
    fig.legend(forward_data.columns)
    plt.grid(True, axis = 'both')