#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd

#this is the generaic path 
path = "/Users/carlinoreilly/Desktop"

# imports csv files to thier corosponding varriables 

dataAccount = pd.read_csv (path+'/lightship_csv/account.csv')  
dataMember = pd.read_csv (path+'/lightship_csv/membership.csv')  
dataTask_event_observers = pd.read_csv (path+'/lightship_csv/task_event_observers.csv')  
dataTask_event_old_observers = pd.read_csv (path+'/lightship_csv/task_event_old_observers.csv')  
dataTask_event_old_teams = pd.read_csv (path+'/lightship_csv/task_event_old_teams.csv')   
dataTask_event_teams = pd.read_csv (path+'/lightship_csv/task_event_teams.csv')  
dataTask_event = pd.read_csv (path+'/lightship_csv/task_event.csv') 
dataTask_observers = pd.read_csv (path+'/lightship_csv/task_observers.csv') 
dataTask_teams = pd.read_csv (path+'/lightship_csv/task_teams.csv') 
dataTask = pd.read_csv (path+'/lightship_csv/task.csv')
dataTask_details_keywords = pd.read_csv (path+'/lightship_csv/task_details_keyword_hashes.csv')
dataTask_title_keyword = pd.read_csv (path+'/lightship_csv/task_title_keyword_hashes.csv')
dataTask_comment_event_keyword = pd.read_csv (path+'/lightship_csv/task_comment_event_keyword_hashes.csv')

# reads only predufined data i thought was usefull for each sheet. 

dfdataAccount = pd.DataFrame(dataAccount, columns= ['id','authority'])

dfdataMember = pd.DataFrame(dataMember, columns= ['id','creator_id', 'account_id','team_id',"admin"])

dfdataTask_event_observers = pd.DataFrame(dataTask_event_observers, columns= ['task_event_id', 'account_id'])

dfdataTask_event_old_observers = pd.DataFrame(dataTask_event_old_observers, columns= ['task_event_id', 'account_id'])

dfTask_event_old_teams = pd.DataFrame(dataTask_event_old_teams, columns= ['task_event_id', 'team_id'])

dfTask_event_teams = pd.DataFrame(dataTask_event_teams, columns= ['task_event_id', 'team_id'])

dfTask_event = pd.DataFrame(dataTask_event, columns= ['id','creator_id', 'task_id', 'descriptor_id'])

dfdataTask_observers = pd.DataFrame(dataTask_observers, columns= ['task_id', 'account_id'])

dfdataTask_teams = pd.DataFrame(dataTask_teams, columns= ['team_id'])

dfdataTask = pd.DataFrame(dataTask, columns= ['id', 'project_id', 'assignee_id', 'creator_id', 'title', 'details'])

dfdataTask_details_keywords = pd.DataFrame(dataTask_details_keywords, columns= ['token', 'token hash'])

dfdataTask_title_keyword = pd.DataFrame(dataTask_title_keyword, columns= ['token', 'token_hash'])

dataTask_comment_event_keyword = pd.read_csv (r'/Users/carlinoreilly/Desktop/lightship_csv/task_comment_event_keyword_hashes.csv')

dfdataTask_comment_event_keyword = pd.DataFrame(dataTask_comment_event_keyword, columns= ['token', 'token_hash'])

dfdataTask = pd.DataFrame(dataTask, columns= ['id', 'project_id', 'assignee_id', 'creator_id', 'title', 'details'])

# prints the data from each sheet 

print (dfdataAccount)
print (dfdataMember)
print (dfdataTask_event_observers)
print (dfdataTask_event_old_observers)
print (dfTask_event_old_teams)
print (dfTask_event_teams)
print (dfTask_event)
print (dfdataTask_observers)
print (dfdataTask_teams)
print (dfdataTask)
print (dataTask_details_keywords)
print (dfdataTask_title_keyword)
print (dfdataTask_comment_event_keyword)
 
                


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




