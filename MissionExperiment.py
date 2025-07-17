import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------------------
# Project Description: Student Performance Analysis in Experimental Conditions
# ------------------------------------------------------------------------------------------

"""
This project analyzes student interaction data collected under two experimental conditions:
1. "autonomy-and-competence"
2. "free"

Each student belongs to one of these experimental groups. For comparison purposes, group labels are encoded as:
    - "free" group                   → -1
    - "autonomy-and-competence" group → 1

The following metrics are computed for each student:

1. **Missions Completed**:
   - Total number of missions completed by the student.

2. **Tasks Completed**:
   - Total number of tasks completed by the student.

3. **Mission Completion Time**:
   - Formula: `missionFinishedAt - missionStartedAt`
   - Total time in **seconds** taken to complete each mission.

4. **Task Completion Time**:
   - Formula: `taskFinishedAt - taskStartedAt`
   - Time in **seconds** taken to complete each individual task.

5. **Average Task Completion Time**:
   - Formula: `Σ(Task Completion Time) / Number of Tasks`
   - Mean time in **seconds** per task.

6. **Total Errors**:
   - Total number of mistakes made by the student.

7. **Total Skips**:
   - Total number of tasks skipped by the student.

8. **Performance Score**:
   - Formula: `100 - ((skips + mistakes) / Total Tasks) * 100`
   - A percentage score where **lower values indicate better performance**.
   - Calculated with **three decimal places**.

9. **Efficiency Index**:
   - Formula: `1 / (1 + ((skips + mistakes) / Total Tasks))`
   - A normalized index where **higher values indicate greater efficiency**.

10. **Task Completion Rate**:
    - Formula: `(Total Tasks - skips) / Total Tasks`
    - A proportion between 0 and 1, representing the percentage of completed tasks.
    - Calculated with **three decimal places**.

11. **Mistake Rate**:
    - Formula: `mistakes / (Total Tasks - skips)`
    - A proportion between 0 and 1, indicating the error rate for completed tasks.
    - Calculated with **three decimal places**.

### Notes on Data Processing:

- Students must be **categorized by experimental group** (`-1` or `1`) for accurate comparisons.
- All time-based metrics (`TaskCompletionTime`, `MissionCompletionTime`, `AverageTaskCompletionTime`) must be converted to **numerical seconds**.
- Rates and percentages (e.g., `TaskCompletionRate`, `MistakeRate`) must be stored as **floating-point values between 0 and 1**, rounded to **three decimal places**.

"""



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
