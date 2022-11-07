#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

#this is the generaic path 
path = "../../data/set_1/"

# imports csv files to thier corosponding varriables 

dataAccount = pd.read_csv (path+'account.csv',keep_default_na=False, na_values=['_'])  
dataMember = pd.read_csv (path+'membership.csv',keep_default_na=False, na_values=['_'])  
dataTask_event_observers = pd.read_csv (path+'task_event_observers.csv',keep_default_na=False, na_values=['_'])  
dataTask_event_old_observers = pd.read_csv (path+'task_event_old_observers.csv',keep_default_na=False, na_values=['_'])  
dataTask_event_old_teams = pd.read_csv (path+'task_event_old_teams.csv',keep_default_na=False, na_values=['_'])   
dataTask_event_teams = pd.read_csv (path+'task_event_teams.csv',keep_default_na=False, na_values=['_'])  
dataTask_event = pd.read_csv (path+'task_event.csv',keep_default_na=False, na_values=['_']) 
dataTask_observers = pd.read_csv (path+'task_observers.csv',keep_default_na=False, na_values=['_']) 
dataTask_teams = pd.read_csv (path+'task_teams.csv',keep_default_na=False, na_values=['_']) 
dataTask = pd.read_csv (path+'task.csv',keep_default_na=False, na_values=['_'])
dataTask_details_keywords = pd.read_csv (path+'keyword_data/task_details_keyword_hashes.csv',keep_default_na=False, na_values=['_'])
dataTask_title_keyword = pd.read_csv (path+'keyword_data/task_title_keyword_hashes.csv',keep_default_na=False, na_values=['_'])
dataTask_comment_event_keyword = pd.read_csv (path+'keyword_data/task_comment_event_keyword_hashes.csv',keep_default_na=False, na_values=['_'])

# reads only predufined data i thought was usefull for each sheet. 

dfDataAccount = pd.DataFrame(dataAccount, columns= ['id','authority'])
dfDataMember = pd.DataFrame(dataMember, columns= ['id','creator_id', 'account_id','team_id',"admin"])
dfDataTask_event_observers = pd.DataFrame(dataTask_event_observers, columns= ['task_event_id', 'account_id'])
dfDataTask_event_old_observers = pd.DataFrame(dataTask_event_old_observers, columns= ['task_event_id', 'account_id'])
dfDataTask_event_old_teams = pd.DataFrame(dataTask_event_old_teams, columns= ['task_event_id', 'team_id'])
dfDataTask_event_teams = pd.DataFrame(dataTask_event_teams, columns= ['task_event_id', 'team_id'])
dfDataTask_event = pd.DataFrame(dataTask_event, columns= ['id','creator_id', 'task_id', 'descriptor_id', 'content'])
dfDataTask_observers = pd.DataFrame(dataTask_observers, columns= ['task_id', 'account_id'])
dfDataTask_teams = pd.DataFrame(dataTask_teams, columns= ['team_id'])
dfDataTask = pd.DataFrame(dataTask, columns= ['id', 'project_id', 'assignee_id', 'creator_id', 'title', 'details'])
dfDataTask_details_keywords = pd.DataFrame(dataTask_details_keywords, columns= ['token', 'token_hash'])
dfDataTask_title_keyword = pd.DataFrame(dataTask_title_keyword, columns= ['token', 'token_hash'])
dfDataTask_comment_event_keyword = pd.DataFrame(dataTask_comment_event_keyword, columns= ['token', 'token_hash'])
dfDataTask = pd.DataFrame(dataTask, columns= ['id', 'project_id', 'assignee_id', 'creator_id', 'title', 'details'])



# In[2]:


#sets the index of that Dataframe
dfDataTask_title_keyword.set_index('token_hash', inplace=True)
dfDataTask_details_keywords.set_index('token_hash', inplace=True)
dfDataTask_comment_event_keyword.set_index('token_hash', inplace=True)


# In[3]:



# sets arrays from the task table for the hashed title and details column
taskTitleArray=[]
taskDetailArray=[]
taskCommentArray=[]

#sets arrays from the task_title_keyword table
titleHashArray=[]
titleKeyArray=[]

#sets arrays from the task_details_keyword table
detailHashArray=[]
detailKeyArray=[]

#sets arrays from the task_comment_event_keyword table
commentHashArray=[]
commentKeyArray=[]

titleValueArray=[]
detailValueArray=[]
commentValueArray=[]


# In[4]:


#sets counts
taskCount = len(dfDataTask.index)
taskEventCommentCount= len(dfDataTask_event.index)
hashTitleCount = len(dfDataTask_title_keyword.index)
hashDetailCount = len(dfDataTask_details_keywords.index)
hashCommentCount = len(dfDataTask_comment_event_keyword.index)


# In[5]:


#populates the titleHash and TitleKey arrays
for x in range(hashTitleCount):

    titleKeyword = dfDataTask_title_keyword['token'][x]
    titleKeywordHash = dfDataTask_title_keyword.index[x]
    
    titleHashArray.append(titleKeyword) 
    titleKeyArray.append(titleKeywordHash)


# In[6]:


#populates the detailHash and detailKey arrys
for x in range(hashDetailCount):

    detailKeyword = dfDataTask_details_keywords['token'][x]
    detailKeywordHash = dfDataTask_details_keywords.index[x]
    
    detailHashArray.append(detailKeyword)
    detailKeyArray.append(detailKeywordHash)


# In[7]:


#populates the commentHash and commentKey arrys
for x in range(hashCommentCount):
    
    commentKeyword = dfDataTask_comment_event_keyword['token'][x]
    commentKeywordHash = dfDataTask_comment_event_keyword.index[x]
    
    commentHashArray.append(commentKeyword)
    commentKeyArray.append(commentKeywordHash)


# In[8]:


#populates the taskTitle and taskDetail Arrays
for x in range(taskCount):

    keyword = dfDataTask['title'][x]
    keyword2 = dfDataTask['details'][x]
    
    taskTitleArray.append(keyword)
    taskDetailArray.append(keyword2)


# In[9]:


#populates the taskComment array
for x in range(taskEventCommentCount):

    keyword3 = dfDataTask_event['content'][x]
    taskCommentArray.append(keyword3)
    


# In[10]:


#these are the keywords for the title column of the task table
#takes the first hash value in task_title_keyword and compairs it to all values for task title. 
#iterates though all values in task_title_keyword

for x in range(len(taskTitleArray)):
    
    title =''
    for i in range(len(titleKeyArray)):
        keyTrue = int(taskTitleArray[x].find(titleKeyArray[i]))
        if keyTrue>0:
            title += ' '+titleHashArray[i]
            
    #fills an arry with the real world values for the hashed inputs
    titleValueArray.append(title)


# In[11]:


#these are the keywords for the details column of the task table
#takes the first hash value in task_details_keyword and compairs it to all values for task title. 
#iterates though all values in task_details_keyword

for x in range(len(taskDetailArray)):
    
    detail =''
    for i in range(len(detailKeyArray)):
        keyTrue = int(taskDetailArray[x].find(detailKeyArray[i]))
        if keyTrue>0:
            detail += ' '+detailHashArray[i]
            
   #fills an arry with the real world values for the hashed inputs
    detailValueArray.append(detail)


# In[12]:


#these are the keywords for the contnent column of the task_event table
#takes the first hash value in task_comment_event_keyword and compairs it to all values for task_event content. 
#iterates though all values in task_comment_event_keyword
for x in range(len(taskCommentArray)):
    
    comment='' 
    for i in range(len(commentKeyArray)):
        keyTrue = int(taskCommentArray[x].find(commentKeyArray[i]))
        if keyTrue>0:
            comment+= ' '+commentHashArray[i]
            
    #fills an arry with the real world values for the hashed inputs   
    commentValueArray.append(comment)
    


# In[13]:


#prints real world values for comments
for x in range(len(detailValueArray)):
    print(str(x+2)+'\t'+ detailValueArray[x])


# In[14]:


#prints real world values for comments
for x in range(len(titleValueArray)):
    print(str(x+2)+'\t'+ titleValueArray[x])


# In[15]:


#prints real world values for comments
for x in range(len(commentValueArray)):
    print(str(x+2)+'\t'+ commentValueArray[x])


# In[ ]:




