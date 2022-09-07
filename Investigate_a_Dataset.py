#!/usr/bin/env python
# coding: utf-8

# 
# # Project: No-show appointments
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# In  this project ,we are going to investigate the findings within the NO_show_appointments dataset based on the Neighbourhood relationship within the standard determining outcomes od PaymentID and PatientId.
# Scholarship allocation relates directly to Neighbourhood allocation derived from the patients admitted for appointments.
# Gender and Age tables also decribe the data in different view relations but all depend on Neighbourhood and AppointmentDay.
# 
# ### Question(s) for Analysis
# Question1.
#            Identify the factors that are important indoder to decide on future appointments scheduled in a Neighbourhood.
#            According to this, i will analyze the main Neighbouhood table and relate it to Hipertension,Alcoholism and Diabetes 
#            related data in an appointments.
#         
# Question2:
#             In relation to appointments which categories are the highest appointments and priority.
#            In corelation, the Hipertension, Alcoholism and Diabetes appointments appear to display different results as per the
#            mean of appointments.
#            Exploring the main mean distribuion of the dependant variables in a dataset in all No-show_apointments.
# 

# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('No_show_appointments.csv')
df.head()


# In[208]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling

# # Load and inspect data

# Read through the dataset 

# In[5]:


df.head()


# In[56]:


# Locate the dataset  information.
#Inspect the shape
df.info()


# In[57]:


#  check for statistics within the No_show_appointments
df.describe()


# # Describe and read through missing values in No_show_appointments over time

# In[58]:


#Missing values in the No_show_appointments
df.info()


# #Analyze the dataset to check the full data available values 

# In[59]:


#missing values in dataset
df[df.Age.isnull()].sum()
df.info()


# In[60]:


#Deriving the mean for first variable 'Age'
df.groupby('Age').mean()


# In[61]:


#Derive mean for appointmentID and PatientID
df.groupby('Age').mean().sum()


# Groupby mean using with sum. using groupby

# In[62]:


df.groupby('AppointmentID').mean().sum()


# Assesing for duplicates in the dataset distributions

# In[63]:


#use means to fill in missing values
df.groupby('Alcoholism').mean().sum()


# In[15]:


#Checking for duplicates in the No_show_appointments.csv file dataset
sum(df.duplicated())


# In[16]:


#Drop duplicates if any in the appointments dataset
df.drop_duplicates(inplace=True)


# In[17]:


#Try to check if info is effected in the dataset
sum(df.duplicated())


# #Checking for unique and not unique values in the variables dataset

# In[54]:


sum(df.nunique())


# In[56]:


sum(df.Age.unique())


# In[62]:


sum(df.Alcoholism.unique())


# In[68]:


sum(df.PatientId.unique())


# In[71]:


sum(df.Scholarship.unique())


# In[72]:


sum(df.Hipertension.unique())


# #Create and read datatypes within the .csv No_show_appointments dataset

# In[2]:


df.dtypes


# In[ ]:





# 
# ### Data Cleaning

# #Further draw cleaning dataset infoamtions

# In[ ]:


# Fill missing values
df.fillna(df.Age(), inplace=True)


# In[76]:


#Describe the current dataset information 
df.info()


# In[78]:


#Check for the derived duplicates
sum(df.duplicated())


# In[79]:


#From the dataset drop out the duplicates
df.drop_duplicates(inplace=True)


# In[80]:


#Confirm the corrections initiated in the dataset
sum(df.duplicated())


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1 (What factors are important to predict if patients will be scheduled appointment)

# #Read through the remaining data and further explore unique statistics.

# In[65]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
df.head()


# In[66]:


df.drop(['PatientId','AppointmentID','Gender','Age','No-show','ScheduledDay'], axis=1,inplace=True)
df.head()


# In[67]:


df.info()


# #Visualize the data explored after initial analysis.

# In[68]:


df.hist(figsize=(10,8))


# In[78]:


df[df.AppointmentDay.isnull()].hist(figsize=(10,9));
df.fillna(df.mean(),inplace=True)
df.info()


# In[77]:


df.head()


# In[26]:


df[df.Scholarship.isnull()]
df.dropna(inplace=True)
df.info()


# #Draw conclusion bar of data vaiables mean in relation to the the scholarhip

# In[8]:


def mean_plotting(df,Scholarship):
    df.groupby(Diabetes).plot(kind='bar')
    plt.show()
    df.head()


# In[7]:


df.groupby('Diabetes').Scholarship.mean()df.groupby('Diabetes').Scholarship.mean().plot(kind='bar');plt.xlabel('Appointments Diabetes(s)')plt.ylabel('mean(m)')plt.title('Mean Distribution')plt.grid(True)
plt.show


# In[69]:


df.groupby('Hipertension').Scholarship.mean()
df.groupby('Hipertension').Scholarship.mean().plot(kind='bar');plt.xlabel('Appointments Hipertension(s)')plt.ylabel('mean(m)')plt.title('Mean Distribution')plt.grid(True)
plt.show


# In[67]:


df.groupby('Alcoholism').Scholarship.mean()df.groupby('Alcoholism').Scholarship.mean().plot(kind='bar');
plt.xlabel('Appointments Alcoholism(s)')plt.ylabel('mean(m)')plt.title('Mean Distribution')plt.grid(True)plt.show


# #According to the main AppointmentDay variable.

# In[70]:


df.groupby('AppointmentDay').Hipertension.mean().plot(kind='bar');
plt.xlabel('No-show_Appointment(s)')plt.ylabel('mean(m)')plt.title('Mean Distribution')plt.grid(True)
plt.show


# Mean value as per patient appointments is derived from the appointmentDay and specific Neighbourhood.

# In[139]:


df.AppointmentDay.value_counts()


# #Check if the variables can be represented as a Number.

# In[209]:


df.query('AppointmentDay=="Diabetes"')['Scholarship'].mean(), df.query('AppointmentDay=="Hipertension"')['Scholarship'].mean()


# #check for mean standings as per each variable in a scholarship and visualize.

# In[73]:


df.groupby(['Alcoholism', 'Hipertension']).Scholarship.mean().plot(kind='bar');plt.xlabel(' Alcoholism,Hipertension(s)')plt.ylabel('mean(m)')plt.title('Mean Distribution')plt.grid(True)
plt.show


# In[78]:


df.groupby(['Diabetes', 'Handcap']).Scholarship.mean().plot(kind='bar');plt.xlabel('Appointments(s)')plt.ylabel('mean(m)')plt.title('Mean Distribution')plt.grid(True)
plt.show


# In[77]:


df['Handcap'].value_counts().plot(kind='bar', alpha=0.5,color='blue')plt.xlabel('Appointments Handcap(s)')plt.ylabel('mean(m)')plt.title('Mean_value(s)')plt.grid(True)
plt.show()


# ### Research Question 2  (What  determines the Neighbouhood appointment )

# # Explore appointments data in the Neighbourhood variable

# In[147]:


#View the current dataset information available on NO-Show_Appointments dataset
df.info()


# In[148]:


df.fillna(df.mean(),inplace=True)
df.info()


# In[149]:


Hipertension=df.Neighbourhood==True
Diabetes=df.Neighbourhood==False


# #Deduce the mean distribution from the full dataset per appointments .

# In[194]:


df.Alcoholism.mean()


# In[168]:


df.Diabetes.mean()


# In[169]:


df.Hipertension.mean()


# #Describe the distribution of patients mean attandance in a Neighbourhood.
# #Visualize the highest mean of patients in a Neighbourhood.

# In[81]:


df.groupby('Neighbourhood').mean().sum().plot(kind='bar',alpha=0.5, )plt.xlabel('independent variable(s)')plt.ylabel('mean(m)')plt.title('Mean Distribution')plt.grid(True)
plt.show()


# #Plot Patients as per the scholarship allocation in mean distribution.
# #Analyze mean per scholarship allocation.

# In[88]:


df.groupby('PatientId').mean().sum().plot(kind='bar',alpha=0.5,)plt.xlabel('Schorlaship sum(s)')plt.ylabel('mean().sum()')plt.title('Mean Distribution')plt.grid(True)
plt.show()


# In[ ]:


<a id='conclusions'></a>
## Conclusions
Findings:
    The Hipertension variable distributes highest in either of the scholarship allocation and the Neighbourhood.
    
    Handcap variable is least focused in the appointments and records the lowest turnouts in every set of data.
    
    According to the data, Hipertention table records a highest mean distribution of 0.2 as opposed to Diabetes 0.07 and 
    Alcoholism mean appointments of 0.03.
    This describes that there is least number of allocations in the variable of Alcoholism addmitted per AppointmentDay and ID.
    
    datatypes of variables are related to integer types as opposed to the Neighbouhood and AppointmentDa which are Object datatypes.
    
    
    In relation to the main No_show appointments, the dataset relates to the columns and rows with different No_show commands as 
    per the Variable of alloction.
Limitations:
    No-Show_appointments data would have been more relevent to include time timestamps which when referenced to the Age and Gender 
    then can be used to detrermine how groups respond to and how fast the response of patients is disitributed.
    This would also create an anlysis argument whish can relate No-show attribute netween the younger category and older reactions to show-up fpr appointments.
    
    
    


# In[9]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




