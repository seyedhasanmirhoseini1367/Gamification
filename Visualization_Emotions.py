


import pandas as pd
import os

# List of all possible emotions
emotions = ['Neutral', 'Angry', 'Sadness', 'Happiness', 'Disgusted', 'Contempt', 'Surprise']


def process_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Initialize a list to store the data for each file
    data = []

    for file in files:
        # Extract the ID from the filename (assuming ID is the filename without extension)
        file_id = os.path.splitext(file)[0]

        # Initialize a dictionary to store the counts for this file
        emotion_counts = {emotion: 0 for emotion in emotions}

        # Read the file into a DataFrame
        df = pd.read_csv(os.path.join(folder_path, file), delim_whitespace=True, header=None,
                         names=['Time', 'Dominant Expression'])

        # Count the occurrences of each emotion
        counts = df['Dominant Expression'].value_counts()

        # Update the emotion counts dictionary
        for emotion in counts.index:
            if emotion in emotion_counts:
                emotion_counts[emotion] = counts[emotion]

        # Append the counts to the data list with the ID
        emotion_counts['ID'] = file_id
        data.append(emotion_counts)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data).set_index('ID')
    return df


# Define the folder paths (assuming they are correct in your environment)
free_folder = 'C:/Users/HP/Desktop/FaceReaderData(Eagle-edu)/Free_control group'
control_folder = 'C:/Users/HP/Desktop/FaceReaderData(Eagle-edu)/C&A _experimental groupl'

# Process each folder
free_df = process_folder(free_folder)
control_df = process_folder(control_folder)

# Save the DataFrames to CSV files
free_df.to_csv('......./Free_emotion_counts.csv')
control_df.to_csv('......./Control_emotion_counts.csv')

free_df.head(), control_df.head()

------------------------------------------------------------------


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

gpd = pd.read_csv('C:/Users/HP/Desktop/FaceReaderData(Eagle-edu)/Gender/Gender_Percentage_dataset.csv')
gpd = gpd.drop(['Neutral', 'Contempt'], axis=1)

gpd_female_mean = gpd[gpd['Gender'] == 'female'].drop('Gender', axis=1).mean()
gpd_male_mean = gpd[gpd['Gender'] == 'male'].drop('Gender', axis=1).mean()

emotions = gpd.columns[1:]  # Assuming the first column is 'Gender'
genders = ['male', 'female']

# Prepare data for plotting: calculate mean percentage for each emotion by gender
data = []
for gender in genders:
    gender_data = gpd[gpd['Gender'] == gender]
    mean_values = gender_data[emotions].mean()  # Calculate the mean only across emotion columns
    data.append(mean_values)

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.4
index = range(len(emotions))

bars = []
for i, gender in enumerate(genders):
    bars.extend(ax.bar([p + i * bar_width for p in index], data[i], width=bar_width, label=gender))

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

ax.set_xlabel('Emotions', labelpad=20)
ax.set_ylabel('Average Percentage', labelpad=10)
ax.set_title('Gender Comparison Across Different Emotions')
ax.set_xticks([p + bar_width / 2 for p in index])
ax.set_xticklabels(emotions)
ax.legend()
plt.tight_layout()
plt.show()

# =================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
gcd = pd.read_csv('C:/Users/HP/Desktop/FaceReaderData(Eagle-edu)/Gender/Gender_Count_dataset.csv')
gcd = gcd.drop(['Neutral', 'Contempt'], axis=1)

gcd_female = gcd[gcd['Gender']=='female'].drop('Gender', axis=1).sum()
gcd_male = gcd[gcd['Gender']=='male'].drop('Gender', axis=1).sum()

emotions = gcd_female.index
genders = ['Male', 'Female']
data = [gcd_male, gcd_female]

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = range(len(emotions))

for i, (gender, counts) in enumerate(zip(genders, data)):
    bars = ax.bar([p + i * bar_width for p in index], counts, width=bar_width, label=gender)

    # Annotate each bar with the respective count
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

ax.set_xlabel('Emotions', labelpad=15)
ax.set_ylabel('Total Counts', labelpad=10)
ax.set_title('Total Emotional Expressions by Gender')
ax.set_xticks([p + bar_width / 2 for p in index])
ax.set_xticklabels(emotions)
ax.legend()
plt.tight_layout()
plt.show()

# =================== emotions from the control group ===================


control_group = pd.read_csv('C:/Users/HP/Desktop/FaceReaderData(Eagle-edu)/Free_control group/free_data_percentage.csv')
control_group = control_group.drop(['Neutral', 'Contempt', 'Unnamed: 0'], axis=1)

emotions = control_group.columns
emotion_means = control_group.mean()

fig, ax = plt.subplots(figsize=(10, 6))

colors = sns.color_palette('husl', len(emotions))
bars = sns.barplot(x=emotion_means.index, y=emotion_means, palette=colors, ax=ax)

for bar in bars.patches:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

ax.set_xlabel('Emotions')
ax.set_ylabel('Average Percentage')
ax.set_title('Average Emotion Distribution in Control Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =================== emotions from the experimental group ===================

expe_group = pd.read_csv('C:/Users/HP/Desktop/FaceReaderData(Eagle-edu)/C&A _experimental group/expe_data_percentage.csv')
expe_group = expe_group.drop(['Neutral', 'Contempt', 'Unnamed: 0'], axis=1)

emotions = expe_group.columns
emotion_means = expe_group.mean()

fig, ax = plt.subplots(figsize=(10, 6))

colors = sns.color_palette('husl', len(emotions))

bars = sns.barplot(x=emotion_means.index, y=emotion_means, palette=colors, ax=ax)

for bar in bars.patches:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

ax.set_xlabel('Emotions')
ax.set_ylabel('Average Percentage')
ax.set_title('Average Emotion Distribution in Control Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =================== emotions from the control and experimental group ===================

control_group = pd.read_csv('C:/Users/HP/Desktop/FaceReaderData(Eagle-edu)/Free_control group/free_data_percentage.csv')
control_group = control_group.drop(['Neutral', 'Contempt', 'Unnamed: 0'], axis=1)

expe_group = pd.read_csv('C:/Users/HP/Desktop/FaceReaderData(Eagle-edu)/C&A _experimental group/expe_data_percentage.csv')
expe_group = expe_group.drop(['Neutral', 'Contempt', 'Unnamed: 0'], axis=1)

emotions = control_group.columns
control_means = control_group.mean()
expe_means = expe_group.mean()

fig, ax = plt.subplots(figsize=(12, 6))

index = range(len(control_means))
bar_width = 0.35

colors = sns.color_palette('husl', len(control_means.index))
control_bars = ax.bar(index, control_means, bar_width, label='Group in contact with the gamified version of Eagle-edu', color='b', alpha=0.6)
expe_bars = ax.bar([p + bar_width for p in index], expe_means, bar_width, label='Group in contact with the non-gamified version of Eagle-edu', color='r', alpha=0.6)

def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_values(control_bars)
add_values(expe_bars)

ax.set_xlabel('Emotions')
ax.set_ylabel('Average Percentage')
ax.set_title('Emotion Distribution Comparison')
ax.set_xticks([p + bar_width / 2 for p in index])
ax.set_xticklabels(control_means.index)
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
