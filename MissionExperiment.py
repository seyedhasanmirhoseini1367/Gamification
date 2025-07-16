import pandas as pd
import numpy as np

# -------------------------------- Description of the Project --------------------------------#
'''
1. Calculate how many missions each student completed 
2. Calculate how many tasks each student completed 
3. Calculate each student's total time to finish each mission 
4. Calculate the total number of errors for each student 
5. Calculate the total number of skips for each student
6. Mission Completion Time: o Formula: Mission Completion Time = missionFinished - missionStartedAt 
This metric provides the total time taken by a student to complete a mission. 
7. Task Completion Time: o Formula: Task Completion Time = taskFinishedAt - taskStartedAt 
This metric provides the time taken by a student to complete a specific task within a mission. 
8. Average Task Completion Time: o Formula: Average Task Completion Time = Σ(Task Completion Time) / Number of Tasks 
This metric provides the average time taken to complete tasks within a mission. 
9. Performance Score: o Formula: Performance Score = 100 - (skips + mistakes) / (Total Tasks) * 100 
This metric combines the number of skips and mistakes to calculate a performance score. The lower the score, the better the performance. 
10. Efficiency Index: o Formula: Efficiency Index = 1 / (1 + (skips + mistakes) / (Total Tasks)) 
This metric provides an efficiency index, where higher values indicate better efficiency by considering skips and mistakes. 
11. Task Completion Rate: o Formula: Task Completion Rate = (Total Tasks - skips) / Total Tasks * 100 
This metric calculates the percentage of tasks completed, considering skips. 
12. Mistake Rate: o Formula: Mistake Rate = mistakes / (Total Tasks - skips) * 100

a few comments about the data processing:

It is necessary to separate students from each course (i.e., "autonomy-and-competence" and "free"), following the two datasets that I sent you.
This is necessary because we will be comparing students from these two courses, and we need the data analysis software to understand this in some way.
for instance, define the courses as -1 (for students in the course “free”) and 1 (for students in the course “autonomy-and-competence”).
"TaskCompletionTime", "MissionCompletionTime", and  "AverageTaskCompletionTime" should be in seconds (e.g., 1480 seconds). 
This is because analyses such as mean comparison and regression do not understand values such as date and time, requesting a pure numerical variable (e.g., average or total).
If "TaskCompletionRate" and "MistakeRate" are percentages, they need to be represented in values between 0 and 1. For example, 0.9 instead of 90 or 0.97 instead of 97.
Always calculate values of this type with 3 decimal places.
'''


# -------------------------  Mission Completion Time ----------------------------#

def process_student_data(data):
    # -------------------------  Mission Completion Time ----------------------------#
    data['missionStartedAt'] = pd.to_datetime(data['missionStartedAt'])
    data['missionFinished'] = pd.to_datetime(data['missionFinished'])
    data['TotalTimePerStudentMission'] = (data['missionFinished'] - data['missionStartedAt']).dt.total_seconds()
    MissionCompletionTime = data.groupby(['idStudents'])['TotalTimePerStudentMission'].sum().reset_index()

    # -------------------------  Task Completion Time ----------------------------#
    data['taskStartedAt'] = pd.to_datetime(data['taskStartedAt'])
    data['taskFinishedAt'] = pd.to_datetime(data['taskFinishedAt'])
    data['TotalTimePerStudentTask'] = (data['taskFinishedAt'] - data['taskStartedAt']).dt.total_seconds()
    TaskCompletionTime = data.groupby(['idStudents'])['TotalTimePerStudentTask'].sum().reset_index()

    # ----------------------- Number Of Missions ----------------------------#
    NumberOfMissions = data.groupby('idStudents')['idMissions'].nunique().reset_index()

    # ----------------------- Number Of Tasks ----------------------------#
    NumberOfTasks = data.groupby('idStudents')['idTask'].nunique().reset_index()

    # ----------------------- Total Errors and Total Skips ----------------------------#
    TotalErrors = data.groupby('idStudents')['mistakes'].sum().reset_index()
    TotalSkips = data.groupby('idStudents')['skips'].sum().reset_index()

    # ----------------------- Average Task Completion Time ----------------------------#
    result_dataframe = pd.merge(TaskCompletionTime, NumberOfTasks, on='idStudents', how='left')
    result_dataframe.columns = ['idStudents', 'TaskCompletionTime', 'HowManyTasks']
    result_dataframe['AverageTaskCompletionTime'] = result_dataframe['TaskCompletionTime'] / result_dataframe[
        'HowManyTasks']

    # ----------------------- Performance Score ----------------------------#
    PerformanceScore = 100 - (TotalSkips['skips'] + TotalErrors['mistakes']) / (NumberOfTasks['idTask']) * 100

    # ----------------------- Efficiency Index ----------------------------#
    EfficiencyIndex = 1 / (1 + (TotalSkips['skips'] + TotalErrors['mistakes']) / (NumberOfTasks['idTask']))

    # ----------------------- Task Completion Rate ----------------------------#
    TaskCompletionRate = (NumberOfTasks['idTask'] - TotalSkips['skips']) / NumberOfTasks['idTask'] * 100

    # ----------------------- Mistake Rate ----------------------------#
    MistakeRate = TotalErrors['mistakes'] / (NumberOfTasks['idTask'] - TotalSkips['skips']) * 100

    # ----------------------- Create an updated DataFrame ----------------------------#
    UpdatedData = pd.DataFrame({
        'idStudents': TaskCompletionTime['idStudents'],
        'NumberOfTasks': NumberOfTasks['idTask'],
        'NumberOfMissions': NumberOfMissions['idMissions'],
        'TotalErrors': TotalErrors['mistakes'],
        'TotalSkips': TotalSkips['skips'],
        'TaskCompletionTime': TaskCompletionTime['TotalTimePerStudentTask'],
        'MissionCompletionTime': MissionCompletionTime['TotalTimePerStudentMission'],
        'AverageTaskCompletionTime': result_dataframe['AverageTaskCompletionTime'],
        'PerformanceScore': PerformanceScore,
        'EfficiencyIndex': round(EfficiencyIndex, 3),
        'TaskCompletionRate': round(TaskCompletionRate / 100, 3),
        'MistakeRate': round(MistakeRate / 100, 3),
    })

    return UpdatedData


free_dataset = pd.read_csv('C:/Users/HP/Desktop/Missions and Tasks/Missions + Tasks session.csv')
autonomy_competence_dataset = pd.read_csv(
    'C:/Users/HP/Desktop/Missions and Tasks/Missions and Tasks(autonomy-and-competence).csv')

free_data = process_student_data(free_dataset)
free_data['idStudents'] = free_data['idStudents'].apply(lambda x: 1)

autonomy_competence_data = process_student_data(autonomy_competence_dataset)
autonomy_competence_data['idStudents'] = autonomy_competence_data['idStudents'].apply(lambda x: -1)

free_data['idStudents'] = free_data['idStudents'].astype('category')
autonomy_competence_data['idStudents'] = autonomy_competence_data['idStudents'].astype('category')

concat_dataset = pd.concat([free_data, autonomy_competence_data], axis=0)
concat_dataset['idStudents'] = concat_dataset['idStudents'].astype('category')

# Shuffle the rows
newdataset = concat_dataset.sample(frac=1).reset_index(drop=True)

free_data.to_csv('C:/Users/HP/Desktop/free_data.csv', index=False)
autonomy_competence_data.to_csv('C:/Users/HP/Desktop/autonomy_competence_data.csv', index=False)

newdataset.to_csv('C:/Users/HP/Desktop/MissionsTasks.csv', index=False)
data = pd.read_csv('C:/Users/HP/Desktop/Missions and Tasks/MissionsTasks.csv')
freedata = pd.read_csv('C:/Users/HP/Desktop/Missions and Tasks/free_data.csv')
notdata = pd.read_csv('C:/Users/HP/Desktop/Missions and Tasks/autonomy_competence_data.csv')

freedata.columns

from scipy import stats

stest=stats.shapiro(data.PerformanceScore)
stest=stats.shapiro(freedata.PerformanceScore)
stest=stats.shapiro(notdata.PerformanceScore)
print(stest)

import seaborn as sn

sn.distplot(newdataset.PerformanceScore)
sn.distplot(freedata.PerformanceScore)
sn.distplot(notdata.PerformanceScore)

sn.boxplot(freedata.PerformanceScore)
sn.boxplot(notdata.PerformanceScore)