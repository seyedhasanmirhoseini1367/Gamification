# Preprocessing:

import numpy as np
from scipy.signal import butter, filtfilt

# ================================================================


import numpy as np
from scipy import signal as sp_signal
from scipy.signal import cheby2, filtfilt, find_peaks, butter
import matplotlib.pyplot as plt

import pandas as pd
import os
import bioread

channel_mapping = {
    '01.11.23':
        {'AA-PPG': 'S001', 'AA-EDA': 'S001',
         'BB-PPG': 'S002', 'BB-EDA': 'S002',
         'CC-PPG': 'S003', 'CC-EDA': 'S003',
         'DD-PPG': 'S004', 'DD-EDA': 'S004',
         'EE-PPG': 'S005', 'EE-EDA': 'S005',
         'FF-PPG': 'S006', 'FF-EDA': 'S006'},

    '02.11.23a':
        {'BB-PPG': 'S008', 'BB-EDA': 'S008',
         'CC-PPG': 'S009', 'CC-EDA': 'S009',
         'DD-PPG': 'S010', 'DD-EDA': 'S010',
         'EE-PPG': 'S011', 'EE-EDA': 'S011'},

    '02.11.23b':
        {'AA-PPG': 'S013', 'AA-EDA': 'S013',
         'BB-PPG': 'S014', 'BB-EDA': 'S014',
         'CC-PPG': 'S015', 'CC-EDA': 'S015',
         'DD-PPG': 'S016', 'DD-EDA': 'S016',
         'EE-PPG': 'S017', 'EE-EDA': 'S017',
         'FF-PPG': 'S018', 'FF-EDA': 'S018'},

    '03.11.23':
        {'AA-PPG': 'S019', 'AA-EDA': 'S019',
         'BB-PPG': 'S020', 'BB-EDA': 'S020',
         'CC-PPG': 'S021', 'CC-EDA': 'S021',
         'DD-PPG': 'S022', 'DD-EDA': 'S022',
         'EE-PPG': 'S023', 'EE-EDA': 'S023',
         'FF-PPG': 'S024', 'FF-EDA': 'S024'},

    '13.11.23a':
        {'AA-PPG': 'S025', 'AA-EDA': 'S025',
         'BB-PPG': 'S026', 'BB-EDA': 'S026',
         'CC-PPG': 'S027', 'CC-EDA': 'S027',
         'DD-PPG': 'S028', 'DD-EDA': 'S028',
         'FF-PPG': 'S030', 'FF-EDA': 'S030'},

    '13.11.23b':
        {'BB-PPG': 'S032', 'BB-EDA': 'S032',
         'CC-PPG': 'S033', 'CC-EDA': 'S033',
         'FF-PPG': 'S036', 'FF-EDA': 'S036'},

    '14.11.23a':
        {'FF-PPG': 'S038', 'FF-EDA': 'S038',
         'BB-PPG': 'S039', 'BB-EDA': 'S039',
         'CC-PPG': 'S041', 'CC-EDA': 'S041'},

    '14.11.23b':
        {'DD-PPG': 'S046', 'DD-EDA': 'S046'},

    '15.11.23a':
        {'BB-PPG': 'S050', 'BB-EDA': 'S050',
         'CC-PPG': 'S051', 'CC-EDA': 'S051',
         'DD-PPG': 'S052', 'DD-EDA': 'S052',
         'EE-PPG': 'S053', 'EE-EDA': 'S053'},

    '15.11.23b':
        {'BB-PPG': 'S056', 'BB-EDA': 'S056',
         'DD-PPG': 'S058', 'DD-EDA': 'S058',
         'EE-PPG': 'S059', 'EE-EDA': 'S059'},

    '16.11.23':
        {'DD-PPG': 'S064', 'DD-EDA': 'S064',
         'EE-PPG': 'S065', 'EE-EDA': 'S065',
         'FF-PPG': 'S066', 'FF-EDA': 'S066'},

    '17.11.23':
        {'BB-PPG': 'S068', 'BB-EDA': 'S068',
         'CC-PPG': 'S069', 'CC-EDA': 'S069'},

    '20.11.23':
        {'BB-PPG': 'S074', 'BB-EDA': 'S074',
         'CC-PPG': 'S075', 'CC-EDA': 'S075',
         'EE-PPG': 'S077', 'EE-EDA': 'S077'},

    '21.11.23':
        {'CC-PPG': 'S081', 'CC-EDA': 'S081',
         'DD-PPG': 'S082', 'DD-EDA': 'S082',
         'EE-PPG': 'S083', 'EE-EDA': 'S083'},

    '22.11.23':
        {'AA-PPG': 'S085', 'AA-EDA': 'S085',
         'BB-PPG': 'S086', 'BB-EDA': 'S086',
         'CC-PPG': 'S087', 'CC-EDA': 'S087',
         'DD-PPG': 'S088', 'DD-EDA': 'S088',
         'EE-PPG': 'S089', 'EE-EDA': 'S089',
         'FF-PPG': 'S090', 'FF-EDA': 'S090'},

    '29.11.23':
        {'BB-PPG': 'S092', 'BB-EDA': 'S092',
         'DD-PPG': 'S094', 'DD-EDA': 'S094',
         'EE-PPG': 'S095', 'EE-EDA': 'S095'},

    'D1.12.23':
        {'BB-PPG': 'S098', 'BB-EDA': 'S098',
         'CC-PPG': 'S099', 'CC-EDA': 'S099',
         'DD-PPG': 'S100', 'DD-EDA': 'S100',
         'EE-PPG': 'S101', 'EE-EDA': 'S101'},

    'D4.11.23':
        {'BB-PPG': 'S104', 'BB-EDA': 'S104',
         'CC-PPG': 'S105', 'CC-EDA': 'S105',
         'EE-PPG': 'S107', 'EE-EDA': 'S107'}
}

interval_time = {
    'S001': {'start_at': '7:40', 'ended_at': '25:40'},
    'S002': {'start_at': '4:10', 'ended_at': '14:30'},
    'S003': {'start_at': '7:40', 'ended_at': '24:25'},
    'S004': {'start_at': '4:25', 'ended_at': '21:00'},
    'S005': {'start_at': '4:45', 'ended_at': '21:10'},
    'S006': {'start_at': '21:40', 'ended_at': '34:24'},
    'S008': {'start_at': '4:10', 'ended_at': '14:30'},
    'S009': {'start_at': '2:55', 'ended_at': '21:05'},
    'S010': {'start_at': '6:40', 'ended_at': '18:35'},
    'S011': {'start_at': '5:50', 'ended_at': '23:40'},
    'S013': {'start_at': '4:00', 'ended_at': '7:00'},
    'S014': {'start_at': '4:15', 'ended_at': '20:15'},
    'S015': {'start_at': '3:15', 'ended_at': '11:00'},
    'S016': {'start_at': '4:55', 'ended_at': '25:50'},
    'S017': {'start_at': '7:20', 'ended_at': '22:10'},
    'S018': {'start_at': '2:25', 'ended_at': '27:45'},
    'S019': {'start_at': '0:05', 'ended_at': '14:0'},
    'S020': {'start_at': '00:00', 'ended_at': '20:0'},
    'S021': {'start_at': '0:0', 'ended_at': '4:40'},
    'S022': {'start_at': '0:10', 'ended_at': '15:50'},
    'S023': {'start_at': '1:55', 'ended_at': '17:40'},
    'S024': {'start_at': '0:0', 'ended_at': '13:25'},
    'S025': {'start_at': '4:0', 'ended_at': '10:30'},
    'S026': {'start_at': '3:35', 'ended_at': '19:15'},
    'S027': {'start_at': '5:20', 'ended_at': '15:30'},
    'S028': {'start_at': '5:40', 'ended_at': '18:50'},
    'S030': {'start_at': '7:0', 'ended_at': '21:0'},
    'S032': {'start_at': '0:20', 'ended_at': '7:55'},
    'S033': {'start_at': '1:10', 'ended_at': '22:30'},
    'S036': {'start_at': '0:50', 'ended_at': '19:30'},
    'S038': {'start_at': '2:45', 'ended_at': '4:20'},
    'S039': {'start_at': '2:40', 'ended_at': '15:30'},
    'S041': {'start_at': '2:50', 'ended_at': '15:40'},
    'S046': {'start_at': '4:00', 'ended_at': '19:20'},
    'S050': {'start_at': '5:30', 'ended_at': '22:10'},
    'S051': {'start_at': '6:00', 'ended_at': '14:45'},
    'S052': {'start_at': '6:15', 'ended_at': '20:40'},
    'S053': {'start_at': '6:45', 'ended_at': '22:50'},
    'S056': {'start_at': '10:40', 'ended_at': '19:55'},
    'S058': {'start_at': '10:5', 'ended_at': '26:50'},
    'S059': {'start_at': '8:00', 'ended_at': '22:25'},
    'S064': {'start_at': '5:00', 'ended_at': '21:10'},
    'S065': {'start_at': '26:25', 'ended_at': '26:40'},
    'S066': {'start_at': '4:30', 'ended_at': '22:25'},
    'S068': {'start_at': '3:25', 'ended_at': '21:40'},
    'S069': {'start_at': '16:40', 'ended_at': '31:10'},
    'S074': {'start_at': '5:25', 'ended_at': '18:40'},
    'S075': {'start_at': '4:55', 'ended_at': '18:15'},
    'S077': {'start_at': '5:50', 'ended_at': '20:55'},
    'S081': {'start_at': '4:05', 'ended_at': '23:5'},
    'S082': {'start_at': '17:05', 'ended_at': '32:40'},
    'S083': {'start_at': '11:50', 'ended_at': '33:30'},
    'S085': {'start_at': '6:00', 'ended_at': '16:00'},
    'S086': {'start_at': '5:16', 'ended_at': '25:00'},
    'S087': {'start_at': '5:40', 'ended_at': '41:50'},
    'S088': {'start_at': '5:45', 'ended_at': '22:15'},
    'S089': {'start_at': '6:03', 'ended_at': '14:50'},
    'S090': {'start_at': '7:35', 'ended_at': '20:25'},
    'S092': {'start_at': '7:54', 'ended_at': '23:45'},
    'S094': {'start_at': '5:35', 'ended_at': '19:30'},
    'S095': {'start_at': '5:25', 'ended_at': '18:45'},
    'S098': {'start_at': '4:50', 'ended_at': '19:00'},
    'S099': {'start_at': '9:20', 'ended_at': '26:35'},
    'S100': {'start_at': '14:45', 'ended_at': '32:35'},
    'S101': {'start_at': '10:26', 'ended_at': '26:30'},
    'S104': {'start_at': '4:17', 'ended_at': '31:48'},
    'S105': {'start_at': '4:40', 'ended_at': '21:50'},
    'S107': {'start_at': '5:50', 'ended_at': '23:40'}
}

path = 'D:/Gamification/BioPac/Biopac-Dataset'

data = {}

for file_date, values in channel_mapping.items():

    current_path = os.path.join(path, file_date)
    for root, dirs, files in os.walk(current_path):  # go through folders to extract files
        for file in files:
            if file.endswith('.acq'):
                file_path = os.path.join(root, file)
                file_content = bioread.read(file_path)  # read files in folder
                channels = file_content.channels
                df = pd.DataFrame()

                for channel in channels:
                    feature = str(channel).split(' ')
                    features = f'{feature[1]}{feature[2]}{feature[3]}'
                    df[features[:-1]] = channel.data

                for key, value in values.items():
                    if value not in data:
                        data[value] = {'PPG': None, 'EDA': None}

                    if key in df.columns:
                        if key.split('-')[-1] == 'PPG':
                            data[value]['PPG'] = df[key]
                        elif key.split('-')[-1] == 'EDA':
                            data[value]['EDA'] = df[key]

print(data.keys())
len(data.keys())


path = 'D:/Gamification/BioPac/Biopac-Dataset/01.11.23/01-11-2023.acq'
df = bioread.read(path)

channels = df.channels


# --------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# ------------------------ Extracting Pure Time for all Participants -----------------------------

intr_time = pd.DataFrame(interval_time).T.reset_index()

# Initialize dictionaries to store the extracted data
extracted_data = {key: {'EDA': None, 'PPG': None} for key in data.keys()}
baseline_data = {key: {'EDA': None, 'PPG': None} for key in data.keys()}

# Time conversion constants
MS_PER_SECOND = 1000
SECONDS_PER_MINUTE = 60
SAMPLING_RATE_HZ = 2000  # 2000 Hz (samples per second)
MS_PER_SAMPLE = 0.5  # 0.5 milliseconds per sample

# Iterate over the ids in the above interval time dictionary
for key, value in interval_time.items():
    # Convert start_at and ended_at to milliseconds
    start_time_split = interval_time[key]['start_at'].split(':')
    end_time_split = interval_time[key]['ended_at'].split(':')

    # Convert MM:SS format to milliseconds
    # First part: minutes to milliseconds (minutes * 60 seconds/minute * 1000 ms/second)
    # Second part: seconds to milliseconds (seconds * 1000 ms/second)
    start_time_ms = (int(start_time_split[0]) * SECONDS_PER_MINUTE * MS_PER_SECOND +
                     int(start_time_split[1]) * MS_PER_SECOND)
    end_time_ms = (int(end_time_split[0]) * SECONDS_PER_MINUTE * MS_PER_SECOND +
                   int(end_time_split[1]) * MS_PER_SECOND)

    # Convert milliseconds to sample indices
    # Since each sample is 0.5 ms, multiply by 2 to get the number of samples
    start_index = int(start_time_ms / MS_PER_SAMPLE)  # equivalent to start_time_ms * 2
    end_index = int(end_time_ms / MS_PER_SAMPLE)  # equivalent to end_time_ms * 2

    # Extract the relevant portion of the data
    extracted_ppg_data = data[key]['PPG'][start_index:end_index]
    extracted_eda_data = data[key]['EDA'][start_index:end_index]

    extracted_data[key]['PPG'] = extracted_ppg_data
    extracted_data[key]['EDA'] = extracted_eda_data

    # Extract the relevant portion of the baseline
    baseline_ppg_data = data[key]['PPG'][:start_index]
    baseline_eda_data = data[key]['EDA'][:start_index]

    if len(baseline_ppg_data) == 0:
        baseline_ppg_data = data[key]['PPG'][end_index:]

    if len(baseline_eda_data) == 0:
        baseline_eda_data = data[key]['EDA'][end_index:]

    baseline_data[key]['PPG'] = baseline_ppg_data
    baseline_data[key]['EDA'] = baseline_eda_data

len(extracted_data.keys())

# ================================================================

def low_pass_filter(data, cutoff=0.5, fs=2000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# =========================== Baseline Correction: ========================

baseline_filtered_eda = {key: {'EDA': None, 'PPG': None} for key in data.keys()}

for key, value in baseline_data.items():
    if len(baseline_data[key]['EDA']) > 18:  # Ensure sufficient data length
        filtered = low_pass_filter(baseline_data[key]['EDA'])
        baseline_filtered_eda[key]['EDA'] = filtered
    else:
        print(f"Data length for {key} is too short for filtering.")

# =========================== Baseline Correction: ========================
# Baseline Correction:

corrected_eda = {key: {'EDA': None, 'PPG': None} for key in baseline_filtered_eda.keys()}

for key, value in baseline_filtered_eda.items():
    baseline = np.mean(baseline_filtered_eda[key]['EDA'])  # Define baseline period
    correct_eda = baseline_filtered_eda[key]['EDA'] - baseline
    corrected_eda[key]['EDA'] = correct_eda


# ==================================== Visualization =================================
def ploting_EDA(id:str):

    # Plot the original and smoothed EEG data
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_data[id]['EDA'].index, baseline_data[id]['EDA'], label='Original Baseline EDA',
             alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(f'EDA {id} ')
    plt.show()

ploting_EDA('S011')

plt.figure(figsize=(10, 6))
plt.plot(baseline_data['S011']['EDA'].index, baseline_filtered_eda['S011']['EDA'], label='Filtered Baseline EDA',
         alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title(f'EDA S011 ')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(baseline_data['S011']['EDA'].index, corrected_eda['S011']['EDA'], label='Corrected Baseline EDA',
         alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title(f'EDA S011 ')
plt.show()


# ==================================== Calculate Mean EDA =================================


exprimental = ['S004', 'S005', 'S006', 'S010', 'S011', 'S016', 'S017', 'S018', 'S022', 'S023', 'S024', 'S028', 'S030',
               'S036', 'S041', 'S046', 'S052', 'S053', 'S058', 'S059', 'S064', 'S065', 'S066', 'S077', 'S082', 'S083',
               'S088', 'S089', 'S090', 'S094', 'S095', 'S100', 'S101', 'S107']

free = ['S001', 'S002', 'S003', 'S008', 'S009', 'S013', 'S014', 'S015', 'S019', 'S020', 'S021', 'S025', 'S026', 'S027',
        'S032', 'S033', 'S038', 'S039', 'S050', 'S051', 'S056', 'S068', 'S069', 'S074', 'S075', 'S081', 'S086', 'S085',
        'S087', 'S092', 'S098', 'S099', 'S104', 'S105']

corrected_expr_data = {k: {'PPG': None, 'EDA': None} for k in exprimental}
corrected_free_data = {k: {'PPG': None, 'EDA': None} for k in free}

for key, value in baseline_filtered_eda.items():
    if key in exprimental:
        corrected_expr_data[key]['PPG'] = corrected_eda[key]['PPG']
        corrected_expr_data[key]['EDA'] = corrected_eda[key]['EDA']
    elif key in free:
        corrected_free_data[key]['PPG'] = corrected_eda[key]['PPG']
        corrected_free_data[key]['EDA'] = corrected_eda[key]['EDA']

print("Experimental Data:", corrected_expr_data.keys())
print("Free Data:", corrected_expr_data.keys())


# ==================================== # Group Averages EDA =================================

# Calculate Mean EDA:

experimental_group_eda = [np.mean(corrected_expr_data[key]['EDA']) for key in exprimental]
control_group_eda = [np.mean(corrected_free_data[key]['EDA']) for key in free]


plt.figure(figsize=(10, 6))
plt.plot(experimental_group_eda.index, experimental_group_eda, label='Corrected Mean EDA',
         alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Mean')
plt.legend()
plt.title(f'Mean experimental_group_eda')
plt.show()



overall_avg_experimental = np.mean(experimental_group_eda)
overall_avg_control = np.mean(control_group_eda)


# Statistical Comparison:
from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(experimental_group_eda, control_group_eda)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# ==================================== # Group Averages EDA =================================


from scipy.stats import shapiro

# Perform the Shapiro-Wilk test for normality
shapiro_eda_expr = shapiro(experimental_group_eda)
shapiro_eda_free = shapiro(control_group_eda)

# shapiro_ppg_expr = shapiro(ppg_expr_mean)
# shapiro_ppg_free = shapiro(ppg_free_mean)

shapiro_eda_expr, shapiro_eda_free
# shapiro_ppg_expr, shapiro_ppg_free

print(f"P_Value of EDA Expr: {shapiro_eda_expr.pvalue}")
print(f"P_Value of EDA Free: {shapiro_eda_free.pvalue}")

# print(f"P_Value of PPG Expr: {shapiro_ppg_expr.pvalue}")
# print(f"P_Value of PPG Free: {shapiro_ppg_free.pvalue}")

# --------------- Test for Experimental and FREE groups -----------------

from scipy.stats import mannwhitneyu

# Mann-Whitney U test for EDA between Experimental and FREE groups
mannwhitney_eda = mannwhitneyu(experimental_group_eda, control_group_eda)

# Mann-Whitney U test for PPG between Experimental and FREE groups
# mannwhitney_ppg = mannwhitneyu(ppg_expr_mean, ppg_free_mean)

print(f"P_Value of Mann-Whitney U test for EDA Free: {mannwhitney_eda.pvalue}")
# print(f"P_Value of Mann-Whitney U test for PPG Free: {mannwhitney_ppg.pvalue}")

































import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =============================================
# DATA PREPARATION
# =============================================
# Assuming:
# experimental_group_eda = list of all EDA values (flattened) for experimental group
# control_group_eda = list of all EDA values (flattened) for control group

# Calculate statistics
exp_mean, exp_std = np.mean(experimental_group_eda), np.std(experimental_group_eda)
ctrl_mean, ctrl_std = np.mean(control_group_eda), np.std(control_group_eda)

# Perform statistical test (Mann-Whitney for non-normal data)
stat, p_value = stats.mannwhitneyu(experimental_group_eda, control_group_eda)

# =============================================
# VISUALIZATION
# =============================================
plt.figure(figsize=(8, 6), dpi=100)  # High resolution for publications

# Create bar plot with error bars
bars = plt.bar(['Experimental', 'Control'],
               [exp_mean, ctrl_mean],
               yerr=[exp_std, ctrl_std],
               capsize=15,
               width=0.6,
               color=['#1f77b4', '#ff7f0e'],  # Professional color scheme
               edgecolor='black',
               linewidth=1.2)

# Add significance annotation
y_max = max(exp_mean + exp_std, ctrl_mean + ctrl_std)
plt.plot([0, 0, 1, 1],
         [y_max*1.05, y_max*1.1, y_max*1.1, y_max*1.05],
         lw=1.5, c='black')
plt.text(0.5, y_max*1.12,
         f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001',
         ha='center', va='bottom')

# Customize plot
plt.ylim(0, y_max*1.15)
plt.ylabel('Electrodermal Activity (μS)', fontsize=12, labelpad=10)
plt.xlabel('Group', fontsize=12, labelpad=10)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
sns.despine()

# Add sample size information below x-axis
plt.text(0, -y_max*0.15, f'n = {len(experimental_group_eda)}',
         ha='center', fontsize=10)
plt.text(1, -y_max*0.15, f'n = {len(control_group_eda)}',
         ha='center', fontsize=10)

plt.tight_layout()  # Prevent label cutoff
plt.savefig('EDA_comparison.png', bbox_inches='tight')  # Save for publication
plt.show()


