

import pandas as pd

data1 = pd.read_csv('C:/Users/HP/Desktop/Missions + Tasks session.csv')
data2 = pd.read_csv('C:/Users/HP/Desktop/Missions and Tasks(autonomy-and-competence).csv')

# -------------------------------- Preprocessing --------------------------------#
data1.info()
data1.isnull().sum()
data1.isna().sum()
# there are no NA or Null values in dataset

data2.info()
data2.isnull().sum()
data2.isna().sum()
# there are no NA or Null values in dataset

data1.shape
data2.shape

# Concatenate data1 and data2 along rows and reset the index
data = pd.concat([data1, data2], axis=0, ignore_index=True)

# Check if there are duplicates based on specified columns
duplicates = data[data.duplicated(keep=False)]
print(duplicates)

NumberStudent = len(data['idStudents'].unique())
print(f'Number of Students: {NumberStudent}')
NumberMission = len(data['idMissions'].unique())
print(f'Number of Missions: {NumberMission}')


# Convert the 'missionStartedAt' and 'missionFinished' columns to datetime format .
data['missionStartedAt'] = pd.to_datetime(data['missionStartedAt'])
data['missionFinished'] = pd.to_datetime(data['missionFinished'])
data['TotalTimePerStudentMission'] = (data['missionFinished'] - data['missionStartedAt']).dt.total_seconds()
# Group by 'idStudents' and 'idMissions' and sum the time taken for each group
MissionCompletionTime = data.groupby(['idStudents'])['TotalTimePerStudentMission'].sum().reset_index()
MissionCompletionTime
# -------------------------  Task Completion Time ----------------------------#

# Convert the 'taskStartedAt' and 'taskFinishedAt' columns to datetime format .
data['taskStartedAt'] = pd.to_datetime(data['taskStartedAt'])
data['taskFinishedAt'] = pd.to_datetime(data['taskFinishedAt'])
data['TotalTimePerStudentTask'] = (data['taskFinishedAt'] - data['taskStartedAt']).dt.total_seconds()
# Group by 'idStudents' and 'idTask' and sum the time taken for each group
TaskCompletionTime = data.groupby(['idStudents'])['TotalTimePerStudentTask'].sum().reset_index()
# AverageTaskCompletionTime = TotalTimeFinishTask.groupby('idStudents').head()
TaskCompletionTime.info()
# ----------------------- Number Of Missions ----------------------------#

NumberOfMissions = data.groupby('idStudents')['idMissions'].count().reset_index()
# Calculate the number of unique missions for each student
NumberOfMissions = data.groupby('idStudents')['idMissions'].nunique().reset_index()

# ----------------------- Number Of Tasks ----------------------------#

NumberOfTasks = data.groupby('idStudents')['idTask'].count().reset_index()
# Calculate the number of unique missions for each student
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
# 100 - (skips + mistakes) / (Total Tasks) * 100
PerformanceScore = 100 - (TotalSkips['skips'] + TotalErrors['mistakes']) / (NumberOfTasks['idTask']) * 100

# ----------------------- Efficiency Index ----------------------------#
# Efficiency Index = 1 / (1 + (skips + mistakes) / (Total Tasks))
EfficiencyIndex = 1 / (1 + (TotalSkips['skips'] + TotalErrors['mistakes']) / (NumberOfTasks['idTask']))

# ----------------------- Task Completion Rate ----------------------------#
# Task Completion Rate = (Total Tasks - skips) / Total Tasks * 100
TaskCompletionRate = (NumberOfTasks['idTask'] - TotalSkips['skips']) / NumberOfTasks['idTask'] * 100

# ----------------------- Mistake Rate ----------------------------#
# Mistake Rate = mistakes / (Total Tasks - skips) * 100
MistakeRate = TotalErrors['mistakes'] / (NumberOfTasks['idTask'] - TotalSkips['skips']) * 100

# ----------------------- Create an updated DataFrame ----------------------------#

# Create an updated DataFrame by combining relevant columns
# The 'idStudents' column is used as the common key for merging the information.
UpdatedData = pd.DataFrame({
    'idStudents': TaskCompletionTime['idStudents'],  # Student Unique IDs
    'NumberOfTasks': NumberOfTasks['idTask'],  # Number of tasks per student
    'NumberOfMissions': NumberOfMissions['idMissions'],  # Number of missions per student
    'TotalErrors': TotalErrors['mistakes'],  # Total mistakes made by each student
    'TotalSkips': TotalSkips['skips'],  # Total skips by each student
    'TaskCompletionTime': TaskCompletionTime['TotalTimePerStudentTask'].astype(str).str.split().str[-1],
    # Total time taken to complete Tasks per student
    'MissionCompletionTime': MissionCompletionTime['TotalTimePerStudentMission'].astype(str).str.split().str[-1],
    # Total time taken to complete missions per student
    'AverageTaskCompletionTime': result_dataframe['AverageTaskCompletionTime'].astype(str).str.split().str[-1],
    # Average Total time taken to complete Tasks per student
    'PerformanceScore': PerformanceScore,
    'EfficiencyIndex': EfficiencyIndex,
    'TaskCompletionRate': TaskCompletionRate/100,
    'MistakeRate': MistakeRate/100,
})

print(UpdatedData)
print(UpdatedData.shape)
print(UpdatedData.columns)



def process_student_data(data, alpha, beta, gamma):
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

    CorrectlyCompletedTasks = NumberOfTasks['idTask'] - (TotalErrors['mistakes']+TotalSkips['skips'])
    AccuracyRate = (CorrectlyCompletedTasks / NumberOfTasks['idTask']) * 100
    TimeEfficiency = 1 / result_dataframe['AverageTaskCompletionTime']

    EfficiencyIndex = 1 / (1 + alpha * TaskCompletionRate +
                           beta * AccuracyRate +
                           gamma * TimeEfficiency)

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

free_data = process_student_data(free_dataset, 0.5, 0.3, 0.2)
free_data['idStudents'] = free_data['idStudents'].apply(lambda x: 1)

autonomy_competence_data = process_student_data(autonomy_competence_dataset, 0.5, 0.3, 0.2)
autonomy_competence_data['idStudents'] = autonomy_competence_data['idStudents'].apply(lambda x: -1)

free_data['idStudents'] = free_data['idStudents'].astype('category')
autonomy_competence_data['idStudents'] = autonomy_competence_data['idStudents'].astype('category')

concat_dataset = pd.concat([free_data, autonomy_competence_data], axis=0)
concat_dataset['idStudents'] = concat_dataset['idStudents'].astype('category')

