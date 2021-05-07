# Project : Investigating Patients not showing for check-ups 

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
                        <li><a href="#Notes on Data">Notes on Data</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
## Introduction



> The data set in hand, has information about the attendance of patients after making an appointment for a check up, it has 110.527 medical appointments it's 14 associated variables. We will be analysing trends for patients who showed up Vs. Patients who didn't show for the appointment, and how they differ.

Data Glossary from Kaggle @ https://www.kaggle.com/joniarroba/noshowappointments

01 - PatientId:
Identification of a patient

02 - AppointmentID:
Identification of each appointment

03 - Gender:
Male or Female . Female is the greater proportion, woman takes way more care of they health in comparison to man.

04 - AppointmentDay:
The day of the actuall appointment, when they have to visit the doctor.

05 - ScheduledDay:
The day someone called or registered the appointment.

06 - Age:
How old is the patient.

07 - Neighbourhood:
Where the appointment takes place.

08 - Scholarship:
True of False . Observation, financial aid, more info @ https://en.wikipedia.org/wiki/Bolsa_Fam%C3%ADlia

09 - Hipertension:
True or False

10 - Diabetes:
True or False

11 - Alcoholism:
True or False

12 - Handcap:
True or False

13 - SMS_received:
1 or more messages sent to the patient.

14 - No-show:
True or False.



> The question we are after,is as follows, what are the behaviours & charachteristics associated with a patient not showing up.
We will investigate the **No-show** Variable, and see whether if the below have any relationship on attending or not.



<ul>
<li><a href="#Age">Do the Age has any relationship with not attending the appointment?</a></li>.
<li><a href="#Receiving a SMS">If receiving multiple SMS has any relationship with not showing up?</a></li>.
<li><a href="#Type of Illness">Does being Ill affect the attendance ?</a></li>.   
<li><a href="#Monetary Issues">If Having a scholarship makes patient more keen to attend?</a></li>.
<li><a href="#Waiting time">Does Waiting time between reservation and check-up date affect Showing up or not?</a></li>.
</ul>
 


```python
# Importing the libraries and modules to be used.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Magic world for inline plotting
%matplotlib inline
plt.style.use('ggplot')

```

<a id='wrangling'></a>
## Data Wrangling


### General Properties


```python
# Load the data and get to know its properties.
raw_appointments = pd.read_csv(
    'D:\\udacity\\Data Analyst Nano Degree\\Lesson 1\\Project\\Data Sets\\noshowappointments-kagglev2-may-2016.csv')
```

### Inspecting the Data


```python
#getting to know the data structure
raw_appointments.shape
```




    (110527, 14)




```python
# printing the columns names and summary to find columns types and missing values if any.
raw_appointments.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 110527 entries, 0 to 110526
    Data columns (total 14 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   PatientId       110527 non-null  float64
     1   AppointmentID   110527 non-null  int64  
     2   Gender          110527 non-null  object 
     3   ScheduledDay    110527 non-null  object 
     4   AppointmentDay  110527 non-null  object 
     5   Age             110527 non-null  int64  
     6   Neighbourhood   110527 non-null  object 
     7   Scholarship     110527 non-null  int64  
     8   Hipertension    110527 non-null  int64  
     9   Diabetes        110527 non-null  int64  
     10  Alcoholism      110527 non-null  int64  
     11  Handcap         110527 non-null  int64  
     12  SMS_received    110527 non-null  int64  
     13  No-show         110527 non-null  object 
    dtypes: float64(1), int64(8), object(5)
    memory usage: 11.8+ MB
    

###### Some Column types needs to be changed:
Patient Id: float to int

ScheduledDay, AppointmentDay to Datetime

No_show to Boolean


```python
# visually exploring the first rows 
raw_appointments.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.987250e+13</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29T18:38:08Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.589978e+14</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29T16:08:27Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.262962e+12</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29T16:19:04Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.679512e+11</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29T17:29:31Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.841186e+12</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29T16:07:23Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



### A random sample for more inspection


```python
# random sample to inspect data visually
raw_appointments.sample(n = 10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27765</th>
      <td>2.721288e+13</td>
      <td>5694725</td>
      <td>M</td>
      <td>2016-05-13T08:54:59Z</td>
      <td>2016-05-18T00:00:00Z</td>
      <td>38</td>
      <td>RESISTÊNCIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>102586</th>
      <td>1.253283e+13</td>
      <td>5734950</td>
      <td>F</td>
      <td>2016-05-24T16:31:22Z</td>
      <td>2016-06-06T00:00:00Z</td>
      <td>1</td>
      <td>SÃO PEDRO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>46486</th>
      <td>4.636194e+14</td>
      <td>5607104</td>
      <td>F</td>
      <td>2016-04-20T11:05:58Z</td>
      <td>2016-05-24T00:00:00Z</td>
      <td>25</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>90498</th>
      <td>4.772153e+13</td>
      <td>5587928</td>
      <td>F</td>
      <td>2016-04-15T09:19:00Z</td>
      <td>2016-06-07T00:00:00Z</td>
      <td>61</td>
      <td>FONTE GRANDE</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>13793</th>
      <td>7.675349e+13</td>
      <td>5646645</td>
      <td>F</td>
      <td>2016-05-02T11:34:28Z</td>
      <td>2016-05-04T00:00:00Z</td>
      <td>38</td>
      <td>GURIGICA</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>57222</th>
      <td>7.116384e+12</td>
      <td>5737191</td>
      <td>M</td>
      <td>2016-05-25T09:19:23Z</td>
      <td>2016-05-25T00:00:00Z</td>
      <td>19</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>65160</th>
      <td>4.639524e+12</td>
      <td>5683525</td>
      <td>F</td>
      <td>2016-05-11T07:36:22Z</td>
      <td>2016-05-11T00:00:00Z</td>
      <td>32</td>
      <td>SÃO CRISTÓVÃO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>50367</th>
      <td>2.898551e+13</td>
      <td>5674455</td>
      <td>F</td>
      <td>2016-05-09T11:05:48Z</td>
      <td>2016-05-11T00:00:00Z</td>
      <td>58</td>
      <td>JESUS DE NAZARETH</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3523</th>
      <td>2.934141e+12</td>
      <td>5574032</td>
      <td>F</td>
      <td>2016-04-12T14:00:25Z</td>
      <td>2016-05-30T00:00:00Z</td>
      <td>41</td>
      <td>RESISTÊNCIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>18242</th>
      <td>4.462249e+12</td>
      <td>5627091</td>
      <td>M</td>
      <td>2016-04-27T09:01:40Z</td>
      <td>2016-05-04T00:00:00Z</td>
      <td>10</td>
      <td>RESISTÊNCIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



### Inspecting columns for irregular data


```python
# summary of the data statistics
raw_appointments.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Age</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.105270e+05</td>
      <td>1.105270e+05</td>
      <td>110527.000000</td>
      <td>110527.000000</td>
      <td>110527.000000</td>
      <td>110527.000000</td>
      <td>110527.000000</td>
      <td>110527.000000</td>
      <td>110527.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.474963e+14</td>
      <td>5.675305e+06</td>
      <td>37.088874</td>
      <td>0.098266</td>
      <td>0.197246</td>
      <td>0.071865</td>
      <td>0.030400</td>
      <td>0.022248</td>
      <td>0.321026</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.560949e+14</td>
      <td>7.129575e+04</td>
      <td>23.110205</td>
      <td>0.297675</td>
      <td>0.397921</td>
      <td>0.258265</td>
      <td>0.171686</td>
      <td>0.161543</td>
      <td>0.466873</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.921784e+04</td>
      <td>5.030230e+06</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.172614e+12</td>
      <td>5.640286e+06</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.173184e+13</td>
      <td>5.680573e+06</td>
      <td>37.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.439172e+13</td>
      <td>5.725524e+06</td>
      <td>55.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.999816e+14</td>
      <td>5.790484e+06</td>
      <td>115.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



###### We observe the following:
* Age values ranges from -1 to 115, the -ve 1 probably a data entery mistake, 115 to be investigated.
* Handicap has values over 1, which will be converted to appropriate level "1"


```python
# inspecting Gender
raw_appointments.Gender.unique()

raw_appointments.Gender.value_counts()
```




    array(['F', 'M'], dtype=object)






    F    71840
    M    38687
    Name: Gender, dtype: int64



##### Here we find that 64% of Patients are Females.


```python
# further inspection of the range of Age values
raw_appointments.Age.value_counts()
```




     0      3539
     1      2273
     52     1746
     49     1652
     53     1651
            ... 
     115       5
     100       4
     102       2
     99        1
    -1         1
    Name: Age, Length: 104, dtype: int64



##### the Counts of Age values


```python
# inspecting the percentage of zero aged patients 
print(raw_appointments['Age'].value_counts(normalize=True)*100)
```

     0      3.201933
     1      2.056511
     52     1.579705
     49     1.494657
     53     1.493753
              ...   
     115    0.004524
     100    0.003619
     102    0.001810
     99     0.000905
    -1      0.000905
    Name: Age, Length: 104, dtype: float64
    

#### We have about 3% of Age values accounting for under 1 year which might need further investigation, and Ages below 0 doesn't have big weight so they can be dropped


```python
# inspecting No-show
raw_appointments['No-show'].unique()
```




    array(['No', 'Yes'], dtype=object)



###### No errors in the dependent variable, might need to convert it to 'True, or False' / boolean values


```python
# checking for duplicates
duplicated = raw_appointments.duplicated()

duplicated.sum()
```




    0



###### No duplicates found


```python
is_null = raw_appointments.isnull().sum()

is_null
```




    PatientId         0
    AppointmentID     0
    Gender            0
    ScheduledDay      0
    AppointmentDay    0
    Age               0
    Neighbourhood     0
    Scholarship       0
    Hipertension      0
    Diabetes          0
    Alcoholism        0
    Handcap           0
    SMS_received      0
    No-show           0
    dtype: int64



###### No null values found

<a id='Notes on Data'></a>
## Notes on Data
> 1. Readability:
    * Handicap typo can be neglected for now.
    * SMS_received , No-show have different seperators ( unify seperators).
    * Neighbourhood values are all caps should be converted to lower caps for better readability later on.
    * drop column Patient_id , Appointment_id as they will not be of further use( no other data to be joined upon nor the values of these columns are to be used furthermore)
 
    
> 2. Adjusting values:
      * Appointment day to be converted to datetime object.
      * Patient id to be converted to int for better readability.
      * Age inputs of larger than 100 or lower than 0 to be investigated.
      * Handicap values > 1 will be converted to one.
      * Converting the No-show Column Values to 1 and zero.
      
      
>  3. Erronus Values:
      * no-null/missing values discovered.
      * Zero Aged patients abour 3% of the observation.



### Data Cleaning and pre-processing

#### Making a copy of the raw data, incase of need for reference 


```python
# making a copy of the raw data frame
appointments = raw_appointments.copy()
```

### Addressing the first issue (Readability)
> Readability:
- [x] Handicap typo to be corrected.
- [x] SMS_received , No-show have different seperators ( unify seperators).
- [x] Neighbourhood values are all caps should be converted to lower caps for better readability later on
- [x] drop column Patient_id , Appointment_id as they will not be of further use( no other data to be joined upon nor the values of these columns are to be used furthermore)


```python
# renaming Handcap to Handicap
appointments.rename(columns = {'Handcap':'Handicap', 'No-show':'No_show', 
                              'PatientId':'Patient_id', 'AppointmentID': 'Appointment_id'}, inplace = True)
```


```python
appointments['Neighbourhood'] = appointments['Neighbourhood'].str.lower()
```


```python
# dropping non-desired column for analysis
appointments.drop(['Patient_id', 'Appointment_id'], axis = 1, inplace = True)
```


```python
#visualy inspecting the dataframe
appointments.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No_show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>2016-04-29T18:38:08Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>jardim da penha</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



### Addressing the Second issue ( Values adjustments)
> Adjusting values:
- [x] Appointment day & SchedulDay to be converted to datetime object.
- [x] Patient id to be converted to int for better readability.
- [x] Age inputs of larger than 100 or lower than 0 to be removed
- [x] Handicap values >1 to be converted to 1.
- [x] Convert the .


```python
# converting Appointment day and ScheduledDay to date time
date_columns = ['AppointmentDay', 'ScheduledDay']

for col in date_columns:
    appointments[col] = pd.to_datetime(appointments[col])
```


```python
# Handicap values with Value >1 to be converted to one
appointments.replace({'Handicap': {2: 1, 3: 1, 4: 1}}, inplace = True)
```


```python
# removing observations with Age value lower than 0
error_values = appointments[appointments['Age'] <0]

appointments.drop(error_values.index, inplace =True)

#checking the Age Column for Values under zero
appointments[appointments['Age'] <0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No_show</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



##### after inspecting patients older than 100 years, it turned out they belong to few patients and not outliers.


```python
# convert No_show Yes and No to 1 and 0
d ={'Yes': 1, 'No':0}
appointments.No_show= appointments.No_show.replace(d)
```


```python
# Validate changes
appointments.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 110526 entries, 0 to 110526
    Data columns (total 12 columns):
     #   Column          Non-Null Count   Dtype              
    ---  ------          --------------   -----              
     0   Gender          110526 non-null  object             
     1   ScheduledDay    110526 non-null  datetime64[ns, UTC]
     2   AppointmentDay  110526 non-null  datetime64[ns, UTC]
     3   Age             110526 non-null  int64              
     4   Neighbourhood   110526 non-null  object             
     5   Scholarship     110526 non-null  int64              
     6   Hipertension    110526 non-null  int64              
     7   Diabetes        110526 non-null  int64              
     8   Alcoholism      110526 non-null  int64              
     9   Handicap        110526 non-null  int64              
     10  SMS_received    110526 non-null  int64              
     11  No_show         110526 non-null  int64              
    dtypes: datetime64[ns, UTC](2), int64(8), object(2)
    memory usage: 11.0+ MB
    


```python
appointments.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No_show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>37.089219</td>
      <td>0.098266</td>
      <td>0.197248</td>
      <td>0.071865</td>
      <td>0.030400</td>
      <td>0.020276</td>
      <td>0.321029</td>
      <td>0.201934</td>
    </tr>
    <tr>
      <th>std</th>
      <td>23.110026</td>
      <td>0.297676</td>
      <td>0.397923</td>
      <td>0.258266</td>
      <td>0.171686</td>
      <td>0.140943</td>
      <td>0.466874</td>
      <td>0.401445</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>55.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>115.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



##### from the mean of the No_show, it is now clear that 20% of total Appointments didn't take place.


```python
#Checking the datframe visually
appointments.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No_show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>62</td>
      <td>jardim da penha</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>56</td>
      <td>jardim da penha</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Splitting the Data into Abscent & Attended Patients
#### we split the data on the variable No-show to be able to compare the other Independant variables.


```python
# No_show with No value means appointment is completed/successful
appointment_completed = appointments[appointments['No_show'] == 0]

# No_show with Yes value means appointment is not done/ didn't take place
appointment_absent = appointments[appointments['No_show'] == 1]
```


```python
# summary desciription of successful appointment
appointment_completed.describe()

# summary desciription of unsuccessful appointment
appointment_absent.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No_show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>88207.000000</td>
      <td>88207.000000</td>
      <td>88207.000000</td>
      <td>88207.000000</td>
      <td>88207.000000</td>
      <td>88207.000000</td>
      <td>88207.000000</td>
      <td>88207.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>37.790504</td>
      <td>0.093904</td>
      <td>0.204394</td>
      <td>0.073838</td>
      <td>0.030417</td>
      <td>0.020792</td>
      <td>0.291337</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>23.338645</td>
      <td>0.291697</td>
      <td>0.403261</td>
      <td>0.261508</td>
      <td>0.171733</td>
      <td>0.142688</td>
      <td>0.454381</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>38.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>56.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>115.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No_show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>22319.000000</td>
      <td>22319.000000</td>
      <td>22319.000000</td>
      <td>22319.000000</td>
      <td>22319.000000</td>
      <td>22319.000000</td>
      <td>22319.000000</td>
      <td>22319.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>34.317667</td>
      <td>0.115507</td>
      <td>0.169004</td>
      <td>0.064071</td>
      <td>0.030333</td>
      <td>0.018236</td>
      <td>0.438371</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>21.965941</td>
      <td>0.319640</td>
      <td>0.374764</td>
      <td>0.244885</td>
      <td>0.171505</td>
      <td>0.133805</td>
      <td>0.496198</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>16.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>51.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>115.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### An overview summary statistic for No-show Vs Show data sets


```python
# summary desciription of successful appointmentcategorical variables
appointment_completed.describe(include=[np.object])

# summary desciription of unsuccessful appointment categorical variables
appointment_absent.describe(include=[np.object])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>88207</td>
      <td>88207</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>80</td>
    </tr>
    <tr>
      <th>top</th>
      <td>F</td>
      <td>jardim camburi</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>57245</td>
      <td>6252</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>22319</td>
      <td>22319</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>80</td>
    </tr>
    <tr>
      <th>top</th>
      <td>F</td>
      <td>jardim camburi</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>14594</td>
      <td>1465</td>
    </tr>
  </tbody>
</table>
</div>



#### From the statistics summary for the Gender and Neighbourhood variables above, We notice that aprox. 65% of Patients not showing are females and highest Neighbourhood with no-show is Jardim Camburi.
#### but we can't infer any insights from from the gender variable, since the data originaly has higher mix of Female patients


```python
# investigating the Correlation between Variables
appointments.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
```




<style  type="text/css" >
    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col0 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col1 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col2 {
            background-color:  #e2dad5;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col3 {
            background-color:  #a2c1ff;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col4 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col5 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col6 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col7 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col0 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col1 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col2 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col3 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col4 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col5 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col6 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col7 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col0 {
            background-color:  #e8d6cc;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col1 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col2 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col3 {
            background-color:  #cedaeb;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col4 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col5 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col6 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col7 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col0 {
            background-color:  #b1cbfc;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col1 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col2 {
            background-color:  #cfdaea;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col3 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col4 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col5 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col6 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col7 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col0 {
            background-color:  #7295f4;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col1 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col2 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col3 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col4 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col5 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col6 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col7 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col0 {
            background-color:  #6e90f2;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col1 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col2 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col3 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col4 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col5 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col6 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col7 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col0 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col1 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col2 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col3 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col4 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col5 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col6 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col7 {
            background-color:  #7396f5;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col0 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col1 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col2 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col3 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col4 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col5 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col6 {
            background-color:  #6a8bef;
            color:  #000000;
        }    #T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col7 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }</style><table id="T_92f6652e_c6df_11ea_9051_e470b8a9bfef" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Age</th>        <th class="col_heading level0 col1" >Scholarship</th>        <th class="col_heading level0 col2" >Hipertension</th>        <th class="col_heading level0 col3" >Diabetes</th>        <th class="col_heading level0 col4" >Alcoholism</th>        <th class="col_heading level0 col5" >Handicap</th>        <th class="col_heading level0 col6" >SMS_received</th>        <th class="col_heading level0 col7" >No_show</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_92f6652e_c6df_11ea_9051_e470b8a9bfeflevel0_row0" class="row_heading level0 row0" >Age</th>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col0" class="data row0 col0" >1.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col1" class="data row0 col1" >-0.09</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col2" class="data row0 col2" >0.50</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col3" class="data row0 col3" >0.29</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col4" class="data row0 col4" >0.10</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col5" class="data row0 col5" >0.08</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col6" class="data row0 col6" >0.01</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow0_col7" class="data row0 col7" >-0.06</td>
            </tr>
            <tr>
                        <th id="T_92f6652e_c6df_11ea_9051_e470b8a9bfeflevel0_row1" class="row_heading level0 row1" >Scholarship</th>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col0" class="data row1 col0" >-0.09</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col1" class="data row1 col1" >1.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col2" class="data row1 col2" >-0.02</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col3" class="data row1 col3" >-0.02</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col4" class="data row1 col4" >0.04</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col5" class="data row1 col5" >-0.01</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col6" class="data row1 col6" >0.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow1_col7" class="data row1 col7" >0.03</td>
            </tr>
            <tr>
                        <th id="T_92f6652e_c6df_11ea_9051_e470b8a9bfeflevel0_row2" class="row_heading level0 row2" >Hipertension</th>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col0" class="data row2 col0" >0.50</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col1" class="data row2 col1" >-0.02</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col2" class="data row2 col2" >1.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col3" class="data row2 col3" >0.43</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col4" class="data row2 col4" >0.09</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col5" class="data row2 col5" >0.08</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col6" class="data row2 col6" >-0.01</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow2_col7" class="data row2 col7" >-0.04</td>
            </tr>
            <tr>
                        <th id="T_92f6652e_c6df_11ea_9051_e470b8a9bfeflevel0_row3" class="row_heading level0 row3" >Diabetes</th>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col0" class="data row3 col0" >0.29</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col1" class="data row3 col1" >-0.02</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col2" class="data row3 col2" >0.43</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col3" class="data row3 col3" >1.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col4" class="data row3 col4" >0.02</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col5" class="data row3 col5" >0.06</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col6" class="data row3 col6" >-0.01</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow3_col7" class="data row3 col7" >-0.02</td>
            </tr>
            <tr>
                        <th id="T_92f6652e_c6df_11ea_9051_e470b8a9bfeflevel0_row4" class="row_heading level0 row4" >Alcoholism</th>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col0" class="data row4 col0" >0.10</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col1" class="data row4 col1" >0.04</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col2" class="data row4 col2" >0.09</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col3" class="data row4 col3" >0.02</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col4" class="data row4 col4" >1.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col5" class="data row4 col5" >0.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col6" class="data row4 col6" >-0.03</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow4_col7" class="data row4 col7" >-0.00</td>
            </tr>
            <tr>
                        <th id="T_92f6652e_c6df_11ea_9051_e470b8a9bfeflevel0_row5" class="row_heading level0 row5" >Handicap</th>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col0" class="data row5 col0" >0.08</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col1" class="data row5 col1" >-0.01</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col2" class="data row5 col2" >0.08</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col3" class="data row5 col3" >0.06</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col4" class="data row5 col4" >0.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col5" class="data row5 col5" >1.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col6" class="data row5 col6" >-0.03</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow5_col7" class="data row5 col7" >-0.01</td>
            </tr>
            <tr>
                        <th id="T_92f6652e_c6df_11ea_9051_e470b8a9bfeflevel0_row6" class="row_heading level0 row6" >SMS_received</th>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col0" class="data row6 col0" >0.01</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col1" class="data row6 col1" >0.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col2" class="data row6 col2" >-0.01</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col3" class="data row6 col3" >-0.01</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col4" class="data row6 col4" >-0.03</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col5" class="data row6 col5" >-0.03</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col6" class="data row6 col6" >1.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow6_col7" class="data row6 col7" >0.13</td>
            </tr>
            <tr>
                        <th id="T_92f6652e_c6df_11ea_9051_e470b8a9bfeflevel0_row7" class="row_heading level0 row7" >No_show</th>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col0" class="data row7 col0" >-0.06</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col1" class="data row7 col1" >0.03</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col2" class="data row7 col2" >-0.04</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col3" class="data row7 col3" >-0.02</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col4" class="data row7 col4" >-0.00</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col5" class="data row7 col5" >-0.01</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col6" class="data row7 col6" >0.13</td>
                        <td id="T_92f6652e_c6df_11ea_9051_e470b8a9bfefrow7_col7" class="data row7 col7" >1.00</td>
            </tr>
    </tbody></table>



##### from the map above we find intersting insights:
 * Age has -ve correlation, so does getting older makes people attend more ?
 * for the reported illnesses, we see a -ve correlation, which makes sense, the motive to go to the clinique is to become healthy.
 * Receiving Multiple SMS showsa positive reationship, so as well as having a scholarship, but not as strong as a SMS.

###### After processing all the required data adjustments and cleaning, we proceed to the EDA part to discover some insights.

<a id='eda'></a>
## Exploratory Data Analysis

> **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. Compute statistics and create visualizations with the goal of addressing the research questions that you posed in the Introduction section. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables.

### Do the Age has any relationship with not attending the appointment?

### Inspecting the Distribution of the variables

#### we split the data on the variable No-show to be able to compare the other Independant variables, firstly we check the Age Variable


```python
# visualize the whole data set

appointments.hist(figsize = (15,15));
```


![png](output_57_0.PNG)


#### We Observe the Age variable is Right skewed, Along with counts for the other Variables.


```python
# Ploting the Age Distribution for both Attended and Absent data sets

plt.rcParams['figure.figsize'] = [10, 8]
sns.kdeplot(appointment_absent['Age'], shade =True, label = 'No_show')
sns.kdeplot(appointment_completed['Age'], color= 'white', label = 'Attended')
plt.minorticks_on()
plt.grid(color='w', linestyle='-', linewidth=0.5, which ='minor', b=None)
plt.legend(loc = 'best');
```


![png](output_59_0.png)


#### from this graph, it show there is an increase in number of Absent Patient starting from age of 5 Years, and raeching the maximum at the age of 20, then declining and reaching the Age of 45 Years where the Attendance rate of Patients start to increase as they get older.


```python
# inspecting the spread and distribution of Age variable
sns.catplot(x='Age', kind="box", data= appointments, col='No_show')
```




    <seaborn.axisgrid.FacetGrid at 0x2d28e58b940>




![png](output_61_1.png)


#### inspecting patients with age > 100 years



```python
# Inspecting the outliers 
appointments[appointments['Age']>100]
appointments[appointments['Age']>100].shape
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No_show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58014</th>
      <td>F</td>
      <td>2016-05-03 09:14:53+00:00</td>
      <td>2016-05-03 00:00:00+00:00</td>
      <td>102</td>
      <td>conquista</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>63912</th>
      <td>F</td>
      <td>2016-05-16 09:17:44+00:00</td>
      <td>2016-05-19 00:00:00+00:00</td>
      <td>115</td>
      <td>andorinhas</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>63915</th>
      <td>F</td>
      <td>2016-05-16 09:17:44+00:00</td>
      <td>2016-05-19 00:00:00+00:00</td>
      <td>115</td>
      <td>andorinhas</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>68127</th>
      <td>F</td>
      <td>2016-04-08 14:29:17+00:00</td>
      <td>2016-05-16 00:00:00+00:00</td>
      <td>115</td>
      <td>andorinhas</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76284</th>
      <td>F</td>
      <td>2016-05-30 09:44:51+00:00</td>
      <td>2016-05-30 00:00:00+00:00</td>
      <td>115</td>
      <td>andorinhas</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>90372</th>
      <td>F</td>
      <td>2016-05-31 10:19:49+00:00</td>
      <td>2016-06-02 00:00:00+00:00</td>
      <td>102</td>
      <td>maria ortiz</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97666</th>
      <td>F</td>
      <td>2016-05-19 07:57:56+00:00</td>
      <td>2016-06-03 00:00:00+00:00</td>
      <td>115</td>
      <td>são josé</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>






    (7, 12)



#### seems that we can't pin point that they are outliers or not, so we will leave the observations as it is.

##### onto the second variable in our Data set to be analysed.



### If receiving a SMS has any relationship with not showing up?</a></li>. 

#### We explore the counts to get a clear view and perception.


```python
sns.catplot(x='No_show', kind="count", data= appointments, hue='SMS_received')
```




    <seaborn.axisgrid.FacetGrid at 0x2d28e53ef28>




![png](output_68_1.png)


#### From this graph it seems that not receiving a SMS shows a Higher percentage of Attending, but this doesn't mean it the cause, but just an observation, while receiving a SMS has some relationship with altering the percentage of not showing up to be larger in comparison to the state of not reveiving a SMS.

##### Now, Does being Ill affect the attendance, & which Illness Has most completed appointments?

first to get  2 dataframes from the original one. One for those attended their appointment and one for those who did not attend to investigate the impact of the disease on attending:
df_d = df.query('diabetes == 1')
df_nd = df.query('diabetes == 0')

Second step: get the percentage of those attended and those who did not attend for those who have the disease and those who does not have the disease
df_d['no_show'].value_counts()/x*100
df_nd['no_show'].value_counts()/x*100

third step: plot comparison histogram
dhist_dia = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=20)
plt.hist(df_d['age'], **dhist_dia);
plt.hist(df_nd['age'], **dhist_dia)


```python
# plotting the Counts distribution and count of reported illnesses
illness = appointments.columns[6:10]
for series in illness:
    sns.catplot(x=series, kind="count", data= appointments);    
```




    <seaborn.axisgrid.FacetGrid at 0x2d28e3c1e80>






    <seaborn.axisgrid.FacetGrid at 0x2d28e879860>






    <seaborn.axisgrid.FacetGrid at 0x2d2896f2390>






    <seaborn.axisgrid.FacetGrid at 0x2d28a92b550>




![png](output_72_4.png)



![png](output_72_5.png)



![png](output_72_6.png)



![png](output_72_7.png)


##### Interstingly the counts of patients out of the 11.5K registered appointments isn't the majority, it only accounts for approx. 31% of the observations, this leads to another question, what drive healthy people to make an appointment for a reserving an appointment in a clinic

##### Which directs our attention to the Scholarship variable, If Having a scholarship makes patient more keen to attend?


```python
# Vizualize the relationship of No_show, Scolarship along with Illnesses

illness = appointments.columns[6:10]

for series in illness:
    g = sns.catplot(x=series,
                hue="No_show", col="Scholarship",
                data=appointments, kind="count",
                height=4, aspect=.7);

```


![png](output_74_0.png)



![png](output_74_1.png)



![png](output_74_2.png)



![png](output_74_3.png)


#### As shown in the Bar Plot, Most of the who didn't register for the reported illnesses and didn't have a Scholarship, in the data set, are missing the appointments more and we will verify that in the next graph


```python
# filtering for Healthy people only
healthy_people = appointments.query('Handicap== 0 & Alcoholism==0 & Diabetes== 0 & Hipertension== 0')

patients = appointments.query('Handicap== 1 or Alcoholism==1 or Diabetes== 1 or Hipertension== 1')


healthy_people.No_show.hist(alpha=0.5 ,label='Healthy');
patients.No_show.hist(alpha = 0.5 ,label='Patient');
plt.legend();
```


![png](output_76_0.png)


##### From this graph it shows that the majority of the No-show cases isn't related to Illness, as these figures are from healthy individuals, and we will verify that with the below graph



```python
# Ploting the Age Distribution for both Attended and Absent data sets

healthy_no_show = healthy_people.query('No_show== 1')

ill_no_show = patients.query('No_show== 1')

plt.rcParams['figure.figsize'] = [10, 8]
sns.kdeplot(healthy_no_show['Age'], shade =True, label = 'healthy')
sns.kdeplot(ill_no_show['Age'], color= 'white', label = 'ill')
plt.minorticks_on()
plt.grid(color='w', linestyle='-', linewidth=0.5, which ='minor', b=None)
plt.legend(loc = 'best');
```


![png](output_78_0.png)



```python
healthy_no_show.shape
ill_no_show.shape
```




    (17603, 12)






    (4716, 12)



#### We discover that majority about 20% of people who missed the appointment, signed up for a treament, while about 80% are healthy individual which makes a strong association with not showing up, perhaps there is another reason for why the signed up for an appointment at the clinic.

### Does Waiting time between reservation and check-up date affect Showing up or not?

#### Creating a new field, time difference between reservation and actual check-up, to investigate schedule effect on attendance.


```python
# copy for maniuplating datetime columns
appointment_date = appointments.copy()


appointment_date['Waiting_time'] = (appointment_date['AppointmentDay'] - appointment_date['ScheduledDay'])
#appointment_date['Waiting_time'] = appointment_date['Waiting_time'].astype('timedelta64[D]')

appointment_date.info()

# No_show with No value means appointment is completed/successful
#appointment_completed = appointment[appointment['No_show'] == 'No']

# No_show with Yes value means appointment is not done/ didn't take place
#appointment_absent = appointment[appointment['No_show'] == 'Yes']

```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 110526 entries, 0 to 110526
    Data columns (total 13 columns):
     #   Column          Non-Null Count   Dtype              
    ---  ------          --------------   -----              
     0   Gender          110526 non-null  object             
     1   ScheduledDay    110526 non-null  datetime64[ns, UTC]
     2   AppointmentDay  110526 non-null  datetime64[ns, UTC]
     3   Age             110526 non-null  int64              
     4   Neighbourhood   110526 non-null  object             
     5   Scholarship     110526 non-null  int64              
     6   Hipertension    110526 non-null  int64              
     7   Diabetes        110526 non-null  int64              
     8   Alcoholism      110526 non-null  int64              
     9   Handicap        110526 non-null  int64              
     10  SMS_received    110526 non-null  int64              
     11  No_show         110526 non-null  int64              
     12  Waiting_time    110526 non-null  timedelta64[ns]    
    dtypes: datetime64[ns, UTC](2), int64(8), object(2), timedelta64[ns](1)
    memory usage: 11.8+ MB
    


```python
# exploring the range of Waiting time variable
pd.reset_option("display.max_rows")
appointment_date.Waiting_time.value_counts()
```




    -1 days +16:50:07    25
    -1 days +16:50:06    25
    13 days 06:42:14     22
    34 days 06:41:33     22
    6 days 06:42:37      19
                         ..
    35 days 05:17:04      1
    40 days 09:53:54      1
    -1 days +11:40:14     1
    1 days 07:39:30       1
    27 days 16:38:41      1
    Name: Waiting_time, Length: 89711, dtype: int64




```python
#removing observations with Waiting time less than 0
invalid_days = appointment_date[appointment_date['Waiting_time'] < '00:00:00']
invalid_days.describe()

invalid_days.shape
appointment_date.drop(invalid_days.index, inplace =True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No_show</th>
      <th>Waiting_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>38567.000000</td>
      <td>38567.000000</td>
      <td>38567.000000</td>
      <td>38567.000000</td>
      <td>38567.000000</td>
      <td>38567.000000</td>
      <td>38567.0</td>
      <td>38567.000000</td>
      <td>38567</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>34.452174</td>
      <td>0.108642</td>
      <td>0.175513</td>
      <td>0.066534</td>
      <td>0.039879</td>
      <td>0.024218</td>
      <td>0.0</td>
      <td>0.046594</td>
      <td>-1 days +13:18:03.543443</td>
    </tr>
    <tr>
      <th>std</th>
      <td>23.222023</td>
      <td>0.311194</td>
      <td>0.380410</td>
      <td>0.249216</td>
      <td>0.195677</td>
      <td>0.153726</td>
      <td>0.0</td>
      <td>0.210771</td>
      <td>0 days 03:07:36.772976</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-7 days +10:10:40</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>15.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-1 days +10:44:12.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>34.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-1 days +14:14:25</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>52.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-1 days +15:52:27.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>115.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>-1 days +17:50:24</td>
    </tr>
  </tbody>
</table>
</div>






    (38567, 13)



#### There are about 37% of Dates which reports Schedule day is after the appointment day, which might play a role, in the no show appointments


```python
appointment_date.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No_show</th>
      <th>Waiting_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>71959.000000</td>
      <td>71959.000000</td>
      <td>71959.000000</td>
      <td>71959.000000</td>
      <td>71959.000000</td>
      <td>71959.000000</td>
      <td>71959.000000</td>
      <td>71959.000000</td>
      <td>71959</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.502564</td>
      <td>0.092706</td>
      <td>0.208897</td>
      <td>0.074723</td>
      <td>0.025320</td>
      <td>0.018163</td>
      <td>0.493086</td>
      <td>0.285190</td>
      <td>15 days 03:50:06.596145</td>
    </tr>
    <tr>
      <th>std</th>
      <td>22.925421</td>
      <td>0.290021</td>
      <td>0.406523</td>
      <td>0.262946</td>
      <td>0.157096</td>
      <td>0.133542</td>
      <td>0.499956</td>
      <td>0.451508</td>
      <td>16 days 11:46:35.560378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0 days 03:16:20</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>19.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3 days 15:14:50</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8 days 16:25:29</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>57.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>21 days 15:01:04.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>115.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>178 days 13:19:01</td>
    </tr>
  </tbody>
</table>
</div>




```python
# exploring the Invalid days and their counts of no- show appointments
sns.catplot(x='No_show', kind="count", data= invalid_days)
```




    <seaborn.axisgrid.FacetGrid at 0x2d28c9deb38>




![png](output_88_1.png)



```python
invalid_days.shape
```




    (38567, 13)



#### it turns out, that the assumption of the invalid appointment days might have larger no show appointments, is not valid


```python
# creating a bin range for waiting time

def bin(days):
    if days >=1 and days <2:
        return "1 day"
    elif days >=2 and days < 8:
        return "2-7 days"
    elif days >=8 and days <15 :
        return "8-14 days"
    elif days >=15 and days < 22:
        return "15-21 days"
    elif days >=23 and days < 31:
        return "22-30 days"
    elif days >=31 and days < 59:
        return "32-59 days"
    else:
        return "60+ days"
    
appointment_date['Wait_time_bins'] = (appointment_date['Waiting_time'] / np.timedelta64(1, 'D')).astype(int).apply(bin)
```


```python
#visualize time delta
plt.figure(figsize=(20,20))
g=sns.catplot(hue='No_show', x= 'Wait_time_bins', kind="count", data= appointment_date,
            orient="h", order= ['1 day','2-7 days', '8-14 days', '15-21 days','22-30 days','32-59 days', '60+ days'])

g.set_xticklabels(rotation=45, horizontalalignment='right');
```


    <Figure size 1440x1440 with 0 Axes>



![png](output_92_1.png)


#### Now, we can observe that waiting time with 1 days shows the best attendence rate among the other intervals, where 8-14 days interval shows the biggest percentage of not showing up compared to attending for the same interval.


```python
# Continue to explore the data to address your additional research
#   questions. Add more headers as needed if you have more questions to
#   investigate.
plt.figure(figsize=(20,15))
appointments.groupby('Neighbourhood').No_show.mean().plot(kind = 'barh');
```


![png](output_94_0.png)


#### Inspecting the Neighbourhood with 100% no-show


```python
# exploring the statistics for the now_show per neighbourhood
pd.reset_option('all')
appointments.groupby('Neighbourhood').No_show.describe()
```

    
    : boolean
        use_inf_as_null had been deprecated and will be removed in a future
        version. Use `use_inf_as_na` instead.
    
    

    C:\Users\mostafa.elmallah\AppData\Local\Continuum\anaconda3\lib\site-packages\pandas\_config\config.py:620: FutureWarning: 
    : boolean
        use_inf_as_null had been deprecated and will be removed in a future
        version. Use `use_inf_as_na` instead.
    
      warnings.warn(d.msg, FutureWarning)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Neighbourhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aeroporto</th>
      <td>8.0</td>
      <td>0.125000</td>
      <td>0.353553</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>andorinhas</th>
      <td>2262.0</td>
      <td>0.230327</td>
      <td>0.421135</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>antônio honório</th>
      <td>271.0</td>
      <td>0.184502</td>
      <td>0.388611</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ariovaldo favalessa</th>
      <td>282.0</td>
      <td>0.219858</td>
      <td>0.414887</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>barro vermelho</th>
      <td>423.0</td>
      <td>0.215130</td>
      <td>0.411399</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>são josé</th>
      <td>1977.0</td>
      <td>0.216490</td>
      <td>0.411956</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>são pedro</th>
      <td>2448.0</td>
      <td>0.210376</td>
      <td>0.407659</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>tabuazeiro</th>
      <td>3132.0</td>
      <td>0.182950</td>
      <td>0.386687</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>universitário</th>
      <td>152.0</td>
      <td>0.210526</td>
      <td>0.409030</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>vila rubim</th>
      <td>851.0</td>
      <td>0.165687</td>
      <td>0.372018</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>81 rows × 8 columns</p>
</div>



#### particularly the Nieghbourhood of Ilhas oceanicas has the highest count for no show, to fix the above graph, we need to select top 10 Neighbourhood with appointments count and see their statistics ( top 10% percentile )


```python
oceanicas_neighbourhood = appointments[appointments['Neighbourhood']=='ilhas oceânicas de trindade']

oceanicas_neighbourhood.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No_show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.000000</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>54.500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.949747</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>51.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>52.750000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>54.500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>56.250000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>58.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### with 2 appointments only, it is not a very siginifcant indication.


```python
# Investigate top Neighbhourhood with No show counts
plt.figure(figsize=(20,15))
appointments.groupby('Neighbourhood').No_show.count().rank(pct = True).sort_values(ascending = False).plot(kind = 'barh');
```


![png](output_100_0.png)



```python
jardim_neighbourhood = appointments[appointments['Neighbourhood']=='jardim camburi']
jardim_neighbourhood.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No_show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7717.000000</td>
      <td>7717.000000</td>
      <td>7717.000000</td>
      <td>7717.000000</td>
      <td>7717.000000</td>
      <td>7717.000000</td>
      <td>7717.000000</td>
      <td>7717.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>43.731502</td>
      <td>0.020604</td>
      <td>0.065958</td>
      <td>0.032526</td>
      <td>0.001296</td>
      <td>0.000778</td>
      <td>0.333679</td>
      <td>0.189841</td>
    </tr>
    <tr>
      <th>std</th>
      <td>22.537394</td>
      <td>0.142063</td>
      <td>0.248225</td>
      <td>0.177403</td>
      <td>0.035977</td>
      <td>0.027875</td>
      <td>0.471557</td>
      <td>0.392200</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>49.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>97.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### with 18% No-show for the largest Neighbourhood count of no show, Vs. the 20% of No-show as an average for the raw data set, Nieghbhourhood seems not to have a strong association with No-show variable

<a id='conclusions'></a>
## Conclusions


<a id='Age'></a>
#### Do the Age has any relationship with not attending the appointment?


```python
#distribution of Age variable

sns.catplot(x='Age', kind="box", data= appointments, col='No_show');
plt.tight_layout();
plt.title('Age and attending behaviour')
```




    <seaborn.axisgrid.FacetGrid at 0x2d29032e5c0>






    Text(0.5, 1, 'Age and attending behaviour')




![png](output_105_2.png)


#### Here we have our first finding, for Age having a sort of relation ship on showing up or not, from the right graph we notice, that the IQR and the median are shifted to the left Vs. the left figure, which gives some indication that younger people might have higher chance to miss an appointment, and we notice for ages starting 39 years old and above people are more likely to adhere to their appointment.

>#### A limitation we faced with this data, is the patients with Age less than a year, not knowing the demographic or the parameters of the Patients we could have removed patients with Ages that don't match the type treatment for example Alcholoism and Diabetes, so I chose to keep the observations of younger people, in order not to affect the relation ships

<a id='Receiving a SMS'></a>
#### If receiving multiple SMS has any relationship with not showing up?


```python
# Plotting SMS count for No_show patients
sns.catplot(x='No_show', kind="count", data= appointments, hue='SMS_received')
plt.title('Multiple SMS received for each patient\'s Show and No_show status');
```


![png](output_108_0.png)


#### It is noticable that percentage of patients not showing up having received multiple SMS, shows a much stronger association compared for patients attending, and it is one the Variables having largest association with Not showing to the appointment.

>For the No_show variable, there is some ambiguity with the SMS sent, if a criteria like how many SMS were sent, or the content for the sent of SMS, it would furtherly cleared the relationship for patients receiving Multiple SMS would have higher association for not showing up.

<a id='Monetary Issues'></a>
#### If Having a scholarship makes patient more keen to attend?


```python
# PLotting NO-show appointments inline with Scholarships
sns.catplot(x='Scholarship', kind="count", data= appointments, hue='No_show');
plt.title('Patients with Scholarships behaviour');
```


![png](output_111_0.png)


#### Here We find an interesting insight, that people with scholarships might have a relationship, with not showing, as the number of patients having a scholarship and missed the appointment increased a little bit compared to patients with no Scholarships,after refering back to the [wikipedia page for the scholarship](https://en.wikipedia.org/wiki/Bolsa_Fam%C3%ADlia).

> A limitation I faced with this variable, that it is comprised of two components School attendence and vaccines, for better understanding the association of this variable with not showing up, further data detailing whether a patient was dropped out of the Scholarship or not, could have provided a more solid insight.

<a id='Type of Illness'></a>
#### Does being Ill affect the attendance ?


```python
#Healthy people and their attendence rate
healthy_people.No_show.hist(alpha=0.5 ,label='Healthy');
patients.No_show.hist(alpha = 0.5 ,label='Patient');
plt.legend();
plt.legend(loc = 'best')
plt.title('Healthy Vs Patient attendence behaviour');
```


![png](output_114_0.png)


#### Now after knowing that Scholarship variable, is an indication of funding based on School attendance, it clears the logic, behind healthy people reserving an appointment in a clinic, perhaps for Vacination or to check on the students attendance at school, and having about 80% of Healthy people making up the bulk of the attendies missing the appointment, it is more likely for healthy individual to miss an appointment in comparison to a patient going for a treatment.

> A limitation I faced for the above, that we can differentiate between healthy individuals and patients having Scholarship or not, so we can confirm if a healthy individual with a scholarship is more likely to miss the appointment, but I couldn't add a visualize it on the same plot.

<a id='Waiting time'></a>
#### Does Waiting time between reservation and check-up date affect Showing up or not?


```python
#visualize time delta
plt.figure(figsize=(20,20))
g=sns.catplot(hue='No_show', x= 'Wait_time_bins', kind="count", data= appointment_date,
            orient="h", order= ['1 day','2-7 days', '8-14 days', '15-21 days','22-30 days','32-59 days', '60+ days'])

g.set_xticklabels(rotation=45, horizontalalignment='right');
```


    <Figure size 1440x1440 with 0 Axes>



![png](output_117_1.png)


#### Here we find that waiting time do have some relationship with missing an appointment, as people are more associated to attend if the time between the reservation date and actual appointment is only one day, while the No-show is observed to have higher percentages of not showing up if the period is larger than 7 days.

>Some limitation that we observed here, is there are some appointment with reported -ve waiting time having patients attending these appointments, which is not consistent, if the date is wrong, it is supposed to have a higher observations of not showing up, but i choose to drop them to have a better view of what might be considered a somewhat error free data.
and the graph above can be improved if it shows the % of No_show per interval, to have an easier and more informative graph.
Also the timedelta variable is tough to maniuplate as I tried to plot and calculate correlation with the dependant variable, bit it didn't show in the heatmap.


```python
from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])
```




    0



# References and hardships encountered

Replace specific values 
https://kanoki.org/2019/07/17/pandas-how-to-replace-values-based-on-conditions/
  
  
counting a value in a series
https://stackoverflow.com/questions/35277075/python-pandas-counting-the-occurrences-of-a-specific-value
  
  
Plotting several columns using loops in Cell 82
https://seaborn.pydata.org/generated/seaborn.barplot.html
  
maniplation of time delta series,
  
Waiting time shows -ve values
 
plotting correlation map for categorical variables
https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas

Pair plot always hangs the notebook and render it not usable

Converting Timedelta to bin intervals
https://towardsdatascience.com/hands-on-python-data-visualization-seaborn-count-plot-90e823599012


Rotating the x-ticks 
https://www.drawingfromdata.com/how-to-rotate-axis-labels-in-seaborn-and-matplotlib

showing all values for pandas methods
https://dev.to/chanduthedev/how-to-display-all-rows-from-data-frame-using-pandas-dha

rank of a column
http://www.datasciencemadesimple.com/percentile-rank-column-pandas-python-2/

