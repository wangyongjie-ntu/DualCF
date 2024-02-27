#Filename:	adult-dice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 14 Jul 2021 08:09:39 

import torch
import dice_ml
import pandas as pd
import numpy as np

def load_adult_income(filename):

    income_df = pd.read_csv(filename)
    income_df.replace("?", np.nan, inplace= True)

    for col in income_df.columns:
        if income_df[col].dtype == np.float64:
            income_df[col].fillna(income_df[col].mean()[0], inplace = True)
        elif income_df[col].dtype  == object:
            income_df[col].fillna(income_df[col].mode()[0], inplace = True)
        else:
            continue
    
    income_df.drop(["fnlwgt"], axis = 1, inplace = True)
    income_df.at[income_df[income_df['income'] == '>50K'].index, 'income'] = 1
    income_df.at[income_df[income_df['income'] == '<=50K'].index, 'income'] = 0

    income_df['education'].replace('Preschool', 'dropout',inplace=True)
    income_df['education'].replace('10th', 'dropout',inplace=True)
    income_df['education'].replace('11th', 'dropout',inplace=True)
    income_df['education'].replace('12th', 'dropout',inplace=True)
    income_df['education'].replace('1st-4th', 'dropout',inplace=True)
    income_df['education'].replace('5th-6th', 'dropout',inplace=True)
    income_df['education'].replace('7th-8th', 'dropout',inplace=True)
    income_df['education'].replace('9th', 'dropout',inplace=True)
    income_df['education'].replace('HS-Grad', 'HighGrad',inplace=True)
    income_df['education'].replace('HS-grad', 'HighGrad',inplace=True)
    income_df['education'].replace('Some-college', 'CommunityCollege',inplace=True)
    income_df['education'].replace('Assoc-acdm', 'CommunityCollege',inplace=True)
    income_df['education'].replace('Assoc-voc', 'CommunityCollege',inplace=True)
    income_df['education'].replace('Bachelors', 'Bachelors',inplace=True)
    income_df['education'].replace('Masters', 'Masters',inplace=True)
    income_df['education'].replace('Prof-school', 'Doctorate',inplace=True)
    income_df['education'].replace('Doctorate', 'Doctorate',inplace=True)

    income_df = income_df.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    income_df = income_df.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov':'Government'}})
    income_df = income_df.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    income_df = income_df.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    income_df = income_df.replace({'workclass': {'?': 'Other/Unknown'}})


    occupation_map = {
        "Adm-clerical": "Admin", "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar"
    }

    income_df['occupation'] = income_df['occupation'].map(occupation_map)

    income_df['marital-status'].replace('Never-married', 'NotMarried',inplace=True)
    income_df['marital-status'].replace(['Married-AF-spouse'], 'Married',inplace=True)
    income_df['marital-status'].replace(['Married-civ-spouse'], 'Married',inplace=True)
    income_df['marital-status'].replace(['Married-spouse-absent'], 'NotMarried',inplace=True)
    income_df['marital-status'].replace(['Separated'], 'Separated',inplace=True)
    income_df['marital-status'].replace(['Divorced'], 'Separated',inplace=True)
    income_df['marital-status'].replace(['Widowed'], 'Widowed',inplace=True)

    income_df['native-country'] = income_df['native-country'].apply(lambda el: 1 if el.strip() == "United-States" else 0)
    income_df['education'] = income_df['education'].map({'dropout':1, 'HighGrad':2, 'CommunityCollege':3, 'Bachelors':4, 'Masters':5, "Doctorate":6})

    return income_df

if __name__ == "__main__":

    income_df = load_adult_income("../data/Adult/adult.csv")
    train_df = income_df.iloc[0:int(len(income_df) * 0.5)]
    query_df = income_df.iloc[int(len(income_df) * 0.5):int(len(income_df)*0.65)]
    columns = income_df.columns.tolist()

    outcome_name = columns[-1]
    continuous_features = []

    for col in income_df.columns:
        if income_df[col].dtype != object and col != 'income':
            continuous_features.append(col)

    d = dice_ml.Data(dataframe = train_df, continuous_features = continuous_features, outcome_name = outcome_name)
    
    backend = 'PYT'
    model_path = "./adult.pt"
    m = dice_ml.Model(model_path = model_path, backend = backend)
    exp = dice_ml.Dice(d, m)
    cf_df = None

    for i in range(len(query_df)):
        print(i)
        query_instance = query_df.iloc[i].to_dict()
        dice_exp = exp.generate_counterfactuals(query_instance, total_CFs = 1, desired_class = 'opposite', max_iter = 10000, verbose = False, posthoc_sparsity_algorithm=None)
        if cf_df is None:
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        else:
            cf_df = pd.concat((cf_df, dice_exp.cf_examples_list[0].final_cfs_df))
    
    cf_df.to_csv("adult-cf.csv", index = None)


