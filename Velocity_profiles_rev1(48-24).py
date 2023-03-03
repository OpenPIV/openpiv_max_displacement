# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:08:18 2022

@author: saaaa331
"""
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Folder management
base_folder ="M:/19_120-OntwOndInfrst/US_scans_KUL/Data 30_08_22/X_Assistance_AL/"

FOL_OUT_plots = base_folder+"/Velocity_profiles"
if not os.path.exists(FOL_OUT_plots):
    os.mkdir(FOL_OUT_plots)

# -----------------------------------------------------------------------------

# VELOCITY EXPERIMENTS (Phased array)
frequencies = ["3.5"]
densities = ["115"]
voltages = ["60"]#,"60","150"

for f in frequencies:
    for d in densities:      
        for U in voltages:
            
            a_data_all_depths = np.zeros((1,5),float)
            
            if U == "60":
                velocities = ["1500"]#,"750","1000","1250","1500","1750","1900"
            elif U == "150":
                velocities = [""]#,"1000","1500"
            else:
                velocities = []
            
            for velocity in velocities:
                FOL_IN = base_folder+"Results_PIV_48-24_S2N1.25/Open_PIV_results_/"
                # DON'T FORGET TO ADJUST NUMBER OF ROWS PER FIELD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # DON'T FORGET TO ADJUST FILE NAME OF OUTPUT
                
                # Determine number of vector fields
                field_numbers = len([entry for entry in os.listdir(FOL_IN) if os.path.isfile(os.path.join(FOL_IN, entry))])/2
                field_numbers = math.floor(field_numbers)
                # field_numbers = 50
                # print(field_numbers)
                
                # Get frame rate
                # META_IN="M:/19_120-OntwOndInfrst/US_scans_KUL/Data 30_08_22/Meta_data"
                # DF = pd.read_csv(META_IN+"/Meta_Velocity_exps_30_08.txt", header = 0,sep="\t")
                # Meta_arr=np.array(DF)
                
                # f_rows=np.where(Meta_arr[:,1]==float(f))
                # d_rows=np.where(Meta_arr[:,2]==int(d))
                # v_rows=np.where(Meta_arr[:,3]==int(U))
                # velo_rows=np.where(Meta_arr[:,4]==int(velocity))
                # rows=np.intersect1d(f_rows, d_rows)
                # rows=np.intersect1d(rows,v_rows)
                # row=np.intersect1d(rows,velo_rows)
                # row=int(row)
                
                # FR=Meta_arr[row,5]
                FR = 354.5847812211757

                # Determine time scale
                Time_scale = np.linspace(0,1/FR*field_numbers,num = field_numbers)
 #----------------------------------------------------------------------------------------------------------------              
 #----------------------------------------------------------------------------------------------------------------               
                # MACRO steps

                # for ro in range(10,int(78/1.2),5):
                for ro in range(10,11,1):
                    # Row = np.around(78-1.2*ro,1)   
                    Row = 54
                    # Depth = np.around(78-Row,1)
                    Depth = 24
                    print (ro, Row, Depth)
 
                    a_temp2 = np.empty((0,2),float)
                
                    for i in range(field_numbers):
                        # print("field_%04d" %i)
                        DF = pd.read_csv(FOL_IN+"field_%04d" %i+".txt", header = 0,sep="\t")
                        vector_data=np.array(DF)
                        
                        # Kick out empty vector fields
                        # U values
                        U_values = vector_data[:,2].astype(float)
                        sum_u_values = np.nansum(U_values)
                        # V values
                        V_values = vector_data[:,3].astype(float)
                        sum_v_values = np.nansum(V_values)
                        if sum_u_values == 0 and sum_v_values == 0:
                            # print("Empty Vector field: field_%04d" %i)
                            VELO_field = np.nan
             
                        else:
                            rows = np.where(vector_data[:,1]==Row)
                            rows = np.asarray(rows[0])
                            a_temp = np.array([])
                            for r in rows:
                                VELO_row= (U_values[r]**2+V_values[r]**2)**0.5
                                a_temp = np.append(a_temp,[VELO_row])                            
                            if np.nansum(a_temp) == 0:
                                VELO_field = np.nan
                            else:
                                # VELO_field is average velocity at depth for each vector field
                                VELO_field = np.nanmean(a_temp)
                            # print(VELO_field)

                        a_temp2 = np.append(a_temp2,np.array([[i,VELO_field]]),axis = 0)

                    # Arange velocities per timestamp                                 
                    a_VELOs = np.column_stack([a_temp2,Time_scale])
                    print ("Depth", Depth, "mm finished")                    



# ----------------------------------------------------------------------------------------------------------------
                    # Plot
                    plt.figure(dpi=100)
                    plt.xlabel("Time [s]")
                    plt.ylabel("Velocity [mm/s]")
                    plt.grid(True)
                    plt.plot(a_VELOs[:,2],a_VELOs[:,1], ".",label="measured data")
                    plt.legend(loc="lower right")
                    plt.title("Velocity "+velocity+" mm/s at "+str(Depth)+" mm depth")
                    
                    plt.savefig(FOL_OUT_plots+"/"+U+"V_"+velocity+"_mmps_"+str(Depth)+"_mm_(48-24).PNG")


            