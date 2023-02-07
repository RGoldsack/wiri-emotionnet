#!/usr/bin/env python
# coding: utf-8


# Importing mocap

import glob
import pandas as pd
import numpy as np
import collections
import re
import time
import sys
import warnings
import gc

warnings.filterwarnings('ignore')

def model_import(path, part = None, valence = "both"):
    #### Importing files that have already been processed so that they can be smooshed together into one big file ####
    sys.stdout.write("Importing Dyads")
    dfBig = pd.DataFrame()
    dyadList = part
    dyadList.sort()
    print(dyadList)
    print(len(dyadList))
    
    if valence == "both":
        for dyad in dyadList:
            sys.stdout.write(dyad)
            sys.stdout.write("\n")
            importPath = path + dyad
            dfDyad = pd.read_csv(importPath)
            dfDyad = dfDyad.dropna()
            dfBig = pd.concat([dfBig, dfDyad], ignore_index = True)
            print(dfBig.shape)

            sys.stdout.flush()
        dfBig = dfBig.reset_index()
        # dfBig.to_csv(path + "dfBig.csv")
        return dfBig
    elif valence == "pos":
        for dyad in dyadList:
            sys.stdout.write(dyad)
            sys.stdout.write("\n")
            importPath = path + dyad
            dfDyad = pd.read_csv(importPath)
            dfDyad = dfDyad.dropna()
            if int(dyad[1:3]) == 3:
                dfDyad = dfDyad.loc[(dfDyad["Trial"] == 2) | (dfDyad["Trial"] == 4)]
            else: 
                dfDyad = dfDyad.loc[(dfDyad["Trial"] == 1) | (dfDyad["Trial"] == 4)]
            dfBig = pd.concat([dfBig, dfDyad], ignore_index = True)
            print(dfBig.shape)
            sys.stdout.flush()
        dfBig = dfBig.reset_index()
        # dfBig.to_csv(path + "dfBig.csv")
        return dfBig
    elif valence == "neg":
        for dyad in dyadList:
            sys.stdout.write(dyad)
            sys.stdout.write("\n")
            importPath = path + dyad
            dfDyad = pd.read_csv(importPath)
            dfDyad = dfDyad.dropna()
            if int(dyad[1:3]) == 3:
                dfDyad = dfDyad.loc[(dfDyad["Trial"] == 1) | (dfDyad["Trial"] == 3)]
            else: 
                dfDyad = dfDyad.loc[(dfDyad["Trial"] == 2) | (dfDyad["Trial"] == 3)]
            dfBig = pd.concat([dfBig, dfDyad], ignore_index = True)
            print(dfBig.shape)
            sys.stdout.flush()
        dfBig = dfBig.reset_index()
        # dfBig.to_csv(path + "dfBig.csv")
        return dfBig
            
        
    print("Done importing and concatting all dyad files")
    dfBig = dfBig.reset_index()
    # dfBig.to_csv(path + "dfBig.csv")
    return dfBig

def importFiles(path, dataToImport = "all", exportList = ["none"], verbose = 0, df_cols = False):
    timeIn = time.time()
            
    #### Importing PANAS Ratings - dfPAN is dataframe PANAS ####
    if dataToImport == "PAN" or dataToImport == "all":
        dfPAN = pd.read_csv(path + "eMOTION_data.csv")
        dfPAN = dfPAN.filter(regex = "Id.|P[1-9]_.")
        dfPAN[["Dyad", "PX"]] = dfPAN["Identification"].str.split("_", expand = True)
        dfPANtemp = dfPAN.filter(regex = "P[1-9]_.")
        colList = dfPANtemp.columns
        colList = colList.tolist()
        
        # Exporting adjusted file
        if "PAN" in exportList or "pre" in exportList:
            outPath = "dfPAN.csv"
            dfPAN.to_csv(outPath, index = False)

    
    #### Importing Six Ratings - dfR6 is dataframe Six Categorical Ratings ####
    R6_emos = list()
    if dataToImport == "R6" or dataToImport == "all":
        globpath = path + "RatingsFiles/" + "*SHORT.csv"
        R6files = glob.glob(globpath)
        R6files.sort()
        for file in R6files:
            dfR6 = pd.read_csv(file)
            dfR6 = dfR6[dfR6["Trial"] == 9999]
            dfR6 = dfR6.filter(regex = "P[AB]\.")
            dfR6[dfR6.isna()] = 0
            for col in dfR6.columns:
                temp = dfR6[dfR6[col] > 0]
                temp.reset_index(inplace = True)
                R6_emos.append(temp.loc[:, col])
        R6_emos = np.array(R6_emos)
        
        # Adding dyad labels
        dyads = list()
        for file in R6files:
            dyads.append(file[-33:-30])
        dyads = dyadList = np.array(dyads)
        dyads = np.repeat(dyads, [6] * len(dyads), axis = 0)
        length = round(len(dyads)/6)
        emotions = np.array(("Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise") * length)
        PX = np.repeat(["PA", "PB"], 6, axis = 0)
        length = round(length / 2)
        PX = np.tile(PX, length)
        dyads = np.column_stack((dyads, emotions, PX))
        R6_emos = np.append(R6_emos, dyads, axis = 1)
        R6_emos = pd.DataFrame(R6_emos, columns = ["T1Self", "T2Self", "T3Self", "T4Self", "T1Other", "T2Other", "T3Other", "T4Other", "Dyad", "Emotions", "PX"])
        
        # Exporting complete file
        if "R6_emos" in exportList or "pre" in exportList:
            outPath = "R6_emos.csv"
            R6_emos.to_csv(outPath)
            
    
    #### Importing Motion Capture files and combining them with other ratings data - dfMC is dataframe motion capture #####
    df_cols = list()
    dfOut = pd.DataFrame()
    if dataToImport == "MC" or dataToImport == "all":
        globpath = path + "MocapFiles/" + "*ALIGNED.csv"
        MCfiles = glob.glob(globpath)
        MCfiles.sort()
        MCcol = 0
        for file in MCfiles:
            if verbose > 1:
                print(file[-30:-27], "file :", file[-20:-18])
            dfMC = pd.read_csv(file)
            dfMC = dfMC.filter(regex = "Frame|Time.|\.Bone\.")
            dfMC.columns = dfMC.columns.str.replace(r"D[0-9][0-9].", "", regex = True)
            dfMC.columns = dfMC.columns.str.replace(r":|_", ".", regex = True)
            #### Adding extra measures to motion capture data ####
            if dataToImport == "all":
                
                #### Adding 6 categorical emotion ratings to motion capture data ####
                colList = list()
                for str3 in [".Self", ".Other"]:
                    for str1 in ["PA.", "PB."]:
                        for str2 in ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]:
                            colList.append(str1 + str2 + str3)
                for item in colList:
                    dfMC[item] = np.NaN
                item = 0
                R6col = int(file[-19:-18]) - 1
                for R6colx in range(2):
                    for R6row in range(12):
                        dfMC[colList[item]] = int(float(R6_emos.iloc[R6row, R6col]))
                        item += 1
                    R6col = R6col + 4
            
                #### Adding PANAS ratings to motion capture data ####
                dyadList = pd.unique(dyadList)
                dfPAN = dfPAN[dfPAN.Dyad.isin(dyadList)]
                regexStr = "P" + file[-19:-18] + "_."
                dfPAN_MC = dfPAN.filter(regex = regexStr)
                colList = list()
                for str1 in ["PA.", "PB."]:
                    for str2 in ["Interested","Distressed","Excited","Upset","Strong","Guilty","Scared","Hostile","Enthusiastic","Proud",
                                 "Irritable","Alert","Ashamed","Inspired","Nervous","Determined","Attentive","Jittery","Active","Afraid"]:
                        colList.append(str1 + "PANAS." + str2)
                for item in colList:
                    dfMC[item] = np.NaN
                item = 0
                for PANrow in range(2):
                    for PANcol in range(20):
                        dfMC[colList[item]] = dfPAN_MC.iloc[PANrow, PANcol]
                        item += 1
        
                #### Adding Continuous Ratings to motion capture data - dfCR is dataframe Continuous Ratings ####
                colList = ["PA.Response.Self", "PB.Response.Self"]
                for item in colList:
                    dfMC[item] = np.NaN
                    importPathCR = path + "RatingsFiles/" + file[-30:-27] + "_" + item[0:2] + "_PSYCHOPY_ALIGNED_LONG.csv"
                    dfCR = pd.read_csv(importPathCR)
                    dfCR = dfCR["Response"][(dfCR["Trial"] == int(file[-19:-18])) & (dfCR["Self/Other"] == "Self")]
                    dfCR = dfCR.reset_index()
                    dfMC[item] = dfCR["Response"]
                
                #### Adding Physio Data to motion capture data - dfPhys is dataframe Physio ####
                colList = ["PA.HR", "PA.ECG", "PA.ChestEx", "PA.SkinTemp", "PA.GSR", "PB.HR", "PB.ECG", "PB.ChestEx", "PB.SkinTemp", "PB.GSR"]
                importPathPhys = path + "PhysioFiles/" + file[-30:-27] + "_PHYS_SHORT.csv"
                dfPhys = pd.read_csv(importPathPhys)
                dfPhys.columns = dfPhys.columns.str.replace("_", ".")
                dfPhys.columns = dfPhys.columns.str.replace("A.", "PA.")
                dfPhys.columns = dfPhys.columns.str.replace("B.", "PB.")
                dfPhys = dfPhys[(dfPhys["Trial"] == int(file[-19:-18]))]
                dfPhys = dfPhys.filter(colList)
                dfPhys = dfPhys.reset_index()
                dfMC = pd.concat([dfMC, dfPhys], axis = 1)

                
                #### Adding a number of extra reference or informative columns ####
                dfMC["Trial"] = np.NaN
                dfMC["Trial"] = int(file[-19:-18])
                dfMC["Dyad"] = np.NaN
                dfMC["Dyad"] = int(file[-29:-27])
                
                
            #### Exporting ####
            df_cols.append(list(dfMC.columns))
            if "MC" in exportList or "pre" in exportList:
                outPath = file[:-30] + file[-30:-4] + "_Emo.csv"
                dfMC.to_csv(outPath, index = False)
            
            #### Concat all files ####
            if ("allIndiv" in exportList) or ("allBig" in exportList):
                fNum = int(file[-19:-18])
#                 print(dfMC.columns.tolist())
                
                # Downsampling
                dfMC["Time (Seconds)"] = pd.to_datetime(dfMC["Time (Seconds)"], unit = "s")
                dfMC["Time (Seconds)"] = pd.DatetimeIndex(dfMC["Time (Seconds)"], dtype = "datetime64[ns]")
                dfMC = dfMC.set_index("Time (Seconds)", drop = False)
                dfMC = dfMC.resample('33.3667ms').mean()
                dfMC.reset_index(drop = True)
                
                # Iterating for a dyad to be exported at a time
                
                if fNum == 1:
                    dfAll = pd.DataFrame()
                    dfAll = pd.concat([dfAll, dfMC], ignore_index = True, axis = 0)
                    dfAll.loc[:,~dfAll.columns.str.contains("Other", case=False)]
                else:
                    dfAll = pd.concat([dfAll, dfMC], ignore_index = True, axis = 0)
                    dfAll.loc[:,~dfAll.columns.str.contains("Other", case=False)]
                if fNum == 4:
                    dfPA = dfAll.filter(regex = "Frame|Time.|PA\.|Trial|Dyad")
                    dfPB = dfAll.filter(regex = "Frame|Time.|PB\.|Trial|Dyad")
#                     print(dfPA.columns.tolist())
#                     print(dfPB.columns.tolist())
                    dfPA["PX"], dfPB["PX"] = "PA", "PB"
                    headers = list(dfPA.columns)
                    strpos = 0
                    for string in headers:
                        if (string.startswith("PA.")):
                            headers[strpos] = string[3:]
                        if (string.startswith("A.")):
                            headers[strpos] = string[2:]
                        strpos += 1
                    if "allIndiv" in exportList:
                        dfPA, dfPB = np.array(dfPA), np.array(dfPB)
                        dfAll = np.append(dfPA, dfPB, axis = 0)
                        dfAll = pd.DataFrame(dfAll, columns = headers)

                        #### Adding PANAS positive and negative sum scores ####
                        dfAll["PANAS.Positive"], dfAll["PANAS.Negative"] = np.NaN, np.NaN
                        dfAll["PANAS.Positive"] = dfAll["PANAS.Interested"] + dfAll["PANAS.Excited"] + dfAll["PANAS.Strong"] + dfAll["PANAS.Enthusiastic"] + dfAll["PANAS.Proud"] + dfAll["PANAS.Alert"] + dfAll["PANAS.Inspired"] + dfAll["PANAS.Determined"] + dfAll["PANAS.Attentive"] + dfAll["PANAS.Active"]
                        dfAll["PANAS.Negative"] = dfAll["PANAS.Distressed"] + dfAll["PANAS.Upset"] + dfAll["PANAS.Guilty"] + dfAll["PANAS.Scared"] + dfAll["PANAS.Hostile"] + dfAll["PANAS.Irritable"] + dfAll["PANAS.Ashamed"] + dfAll["PANAS.Nervous"] + dfAll["PANAS.Jittery"] + dfAll["PANAS.Afraid"]
    #                     print(dfAll["PANAS.Positive"].mean(), "is the sum of Positive mean")
    #                     print(dfAll["PANAS.Negative"].mean(), "is the sum of Negative mean")

                        outPath = path + "DyadFiles/" + file[-30:-27] + "_dfBig.csv"
                        dfAll.to_csv(outPath, index = False)
                        print(dfAll.shape, "second dfAll, file has been exported")
                        del dfAll
                        del dfPA
                        del dfPB
                        gc.collect()
                    if "allBig" in exportList:
                        dfPA, dfPB = np.array(dfPA), np.array(dfPB)
                        dfPX = np.append(dfPA, dfPB, axis = 0)
                        dfPX = pd.DataFrame(dfPX, columns = headers)
                        dfOut = pd.concat([dfOut, dfPX], ignore_index = True)
                        print(dfOut.shape)
                    
            if verbose > 2:
                print("Done ", file[-30:-27], "file :", file[-20:-18])
            
    timeOut = time.time()
    if verbose > 0:
        print("Execution time in minutes for all files: ", ((timeOut - timeIn)/60))
    if df_cols == True:
        return(df_cols)
    if "allBig" in exportList:
        dfOut["PANAS.Positive"],  dfOut["PANAS.Negative"]   = np.NaN, np.NaN
        dfOut["PANAS.Positive"] = dfOut["PANAS.Interested"] + dfOut["PANAS.Excited"] + dfOut["PANAS.Strong"] + dfOut["PANAS.Enthusiastic"] + dfOut["PANAS.Proud"]   + dfOut["PANAS.Alert"]     + dfOut["PANAS.Inspired"] + dfOut["PANAS.Determined"] + dfOut["PANAS.Attentive"] + dfOut["PANAS.Active"]
        dfOut["PANAS.Negative"] = dfOut["PANAS.Distressed"] + dfOut["PANAS.Upset"]   + dfOut["PANAS.Guilty"] + dfOut["PANAS.Scared"]       + dfOut["PANAS.Hostile"] + dfOut["PANAS.Irritable"] + dfOut["PANAS.Ashamed"]  + dfOut["PANAS.Nervous"]    + dfOut["PANAS.Jittery"]   + dfOut["PANAS.Afraid"]
        print("big ol dude shape", dfOut.shape)
#         return(dfOut)
    return("Done!")
        
# path = "/Users/roydon/Desktop/MOCAP/"
#path = "/Volumes/fastt/Data/"
# path = "C:/Users/goldsaro.STAFF/Data/"
# dfAll = importFiles(path, dataToImport = "all", exportList = ["all"])
#importFiles(path, dataToImport = "all", exportList = ["allIndiv"], verbose = 2, df_cols = True)



def columnChecker(df_cols, export = False):
    print("Looking at sets")
    for col in range(0, 15):
        for num in range(0,15):
            if set(df_cols[col]) != set(df_cols[num]):
                print("df_col", col, "and num is", num, "SET")
            if df_cols[col] != df_cols[num]:
                print("df_col", col, "and num is", num, "COLUMNS")
    print("Finished looking at sets")
    if export == True:
        df_cols.to_csv("df_cols.csv", index = False)