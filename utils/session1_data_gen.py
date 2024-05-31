import pandas as pd
import numpy as np
import random
from random import randrange
from datetime import timedelta
import datetime

n_patients = 100
age_dist_split = 0.5
mri_prop = 0.1

start_date = datetime.datetime(2023, 1, 25)
end_date = datetime.datetime(2023, 8, 25)

imaging_mod = [
    ["x-ray", "xr", "XRAY"],
    ["ct", "CT", "Ct"],
    ["mr", "mri", "Mri", "MRI"]
]

def convert_to_prob(x, mu, sigma): 
    return 1 / (sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))  

def select_imaging_mod(pt_age, admission_date):

    # Select an imaging modility based on age, which will be the main insight from this workshop
    prob_xr = convert_to_prob(pt_age, 25, 20.0)
    prob_ct = convert_to_prob(pt_age, 75, 10.0)
    prob_mr = np.random.random(1)[0] * mri_prop
    probs = [prob_xr, prob_ct, prob_mr]

    scale_fac = 1.0 / np.sum(probs)
    probs = np.multiply(probs, scale_fac)

    val = np.random.choice(3, p=probs)

    # Get the length of time for the scan based on the modality
    if int(val) == 0:
        delta = np.random.f(1, 48, 1)[0] * 10
    elif int(val) == 1:
        delta = np.random.f(1, 48, 1)[0] * 10
    else:
        delta = np.random.f(100, 12, 1)[0] * 10

    return np.random.choice(imaging_mod[int(val)]), admission_date + timedelta(days=delta)

def random_date(start_date, end_date):
    # Generate a random date based on two limits
    delta = end_date - start_date
    d = (delta.days * 24 * 60 * 60) + delta.seconds
    act_d = random.randint(0, d)
    return start_date + timedelta(seconds=act_d)

def proc_dates(date_arr):
    # Appropriately format the date
    return np.array([date.strftime("%Y-%m-%d %H:%M") for date in date_arr])

if __name__ == "__main__":

    # Generate the patient ages for the simulated dataset and shuffle them. 
    age = np.concatenate([
        np.random.normal(loc=18.0, scale=10.0, size=int(n_patients * age_dist_split)),
        np.random.normal(loc=65.0, scale=10.0, size=int(n_patients * age_dist_split))
    ])
    np.random.shuffle(age)
    age = np.clip(age, 0, 103)

    # Generate the imaging modality and the time of imaging
    admission_dates = [random_date(start_date, end_date) for _ in range(n_patients)]
    ret = np.array([select_imaging_mod(a, date) for a, date in zip(age, admission_dates)])
    img_mod, imaging_dates = ret[:, 0], ret[:, 1]
    
    # Format the dates for the final dataframe
    admission_dates = proc_dates(admission_dates)
    imaging_dates = proc_dates(imaging_dates)

    df = pd.DataFrame({
        "Patient ID": [i for i in range(n_patients)],
        "Age": age.astype("int32"),
        "Admission Date": admission_dates,
        "Imaging Date": imaging_dates,
        "Imaging Modality": img_mod
    })

    # Save the simulated data to an Excel file
    df.to_excel("notebooks/data/imaging_audit.xlsx")
    df.to_csv("notebooks/data/imaging_audit.csv")