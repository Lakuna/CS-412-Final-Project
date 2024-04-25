"""CS 412 final project."""

import csv
import pathlib

import numpy as np

# Load Twitter data into a NumPy array.
#
# Attributes:
# - Number of Created Discussions (NCD) (columns $[0,6]$): Measures the number of
#   discussions created at time step $t$ and involving the instance's topic.
#   Column $t$ represents the NCD at relative time $t$, abbreviated $NCD_t$.
# - Author Increase (AI) (columns $[7,13]$): Measures the number of new authors
#   interacting on the instance's topic at time $t$.
#   Column $t+7$ represents the AI at relative time $t$, abbreviated $AI_t$.
# - Attention Level Measured with Number of Authors (AS(NA)) (columns $[14,20]$):
#   Measures the attention paid to the instance's topic.
#   Column $t+14$ represents the AS(NA) at relative time $t$, abbreviated $AS(NA)_t$.
# - Burstiness Level (BL) (columns $[21,27]$): For a topic $z$ at time $t$, the
#   burstiness level is $NCD_t/NAD_t$.
#   Column $t+21$ represents the BL at relative time $t$, abbreviated $BL_t$.
# - Number of Atomic Containers (NAC) (columns $[28,34]$): Measures the total number of
#   atomic containers generated through the whole social media on the instance's topic
#   until time $t$.
#   Column $t+28$ represents the NAC at relative time $t$, abbreviated $NAC_t$.
# - Attention Level Measured with Number of Contributions (AS(NAC))
#   (columns $[35,41]$): Measures the attention paid to the instance's topic.
#   Column $t+35$ represents the AS(NAC) at relative time $t$, abbreviated $AS(NAC)_t$.
# - Contribution Sparseness (CS) (columns $[42,48]$): Measure of the spreading of
#   contributions over discussions for the instance's topic at time $t$.
#   Column $t+42$ represents the CS at relative time $t$, abbreviated $CS_t$.
# - Author Interaction (AT) (columns $[49,55]$): Measures the average number of authors
#   interacting on the instance's topic within a discussion.
#   Column $t+49$ represents the AT at relative time $t$, abbreviated $AT_t$.
# - Number of Authors (NA) (columns $[56,62]$): Measures the number of authors
#   interacting on the instance's topic at time $t$.
#   Column $t+56$ represents the NA at relative time $t$, abbreviated $NA_t$.
# - Average Discussion Length (ADL) (columns $[63,69]$): Measures the average length of
#   a discussion belonging to the instance's topic.
#   Column $t+63$ represents the ADL at relative time $t$, abbreviated $ADL_t$.
# - Number of Active Discussions (NAD) (columns $[70,76]$): Measures the number of
#   discussions involving the instance's topic until time $t$.
#   Column $t+70$ represents the NAD at relative time $t$, abbreviated $NAD_t$.
# - Feature to Predict (column $77$): Measures the mean NAD, which describes the
#   popularity of the instance's topic.
#   Column $77$ represents the mean NAD, abbreviated $NAD$.
with pathlib.Path.open("Twitter.data") as csvfile:
    twitter_data = np.array(list(csv.reader(csvfile)), dtype=np.float64)

# Load Tom's Hardware data into a NumPy array.
#
# Attributes:
# - Number of Created Discussions (NCD) (columns $[0,7]$): Measures the number of
#   discussions created at time step $t$ and involving the instance's topic.
#   Column $t$ represents the NCD at relative time $t$, abbreviated $NCD_t$.
# - Burstiness Level (BL) (columns $[8,15]$): For a topic $z$ at time $t$, the
#   burstiness level is the ratio of $NCD_t$ to $NAD_t$.
#   Column $t+8$ represents the BL at relative time $t$, abbreviated $BL_t$.
# - Number of Active Discussions (NAD) (columns $[16,23]$): Measures the number of
#   discussions involving the instance's topic until time $t$.
#   Column $t+16$ represents the NAD at relative time $t$, abbreviated $NAD_t$.
# - Author Increase (AI) (columns $[24,31]$): Measures the number of new authors
#   interacting on the instance's topic at time $t$ (its popularity).
#   Column $t+24$ represents the AI at relative time $t$, abbreviated $AI_t$.
# - Number of Atomic Containers (NAC) (columns $[32,39]$): Measures the total number of
#   atomic containers generated through the whole social media on the instance's topic
#   until time $t$.
#   Column $t+32$ represents the NAC at relative time $t$, abbreviated $NAC_t$.
# - Number of Displays (ND) (columns $[40,47]$): Gives the number of times that
#   discussions relying on the instance's topic has been displayed by users.
#   Column $t+40$ represents the ND at relative time $t$, abbreviated $ND_t$.
# - Contribution Sparseness (CS) (columns $[48,55]$): Measure of the spreading of
#   contributions over discussions for the instance's topic at time $t$.
#   Column $t+48$ represents the CS at relative time $t$, abbreviated $CS_t$.
# - Author Interaction (AT) (columns $[56,63]$): Measures the average number of authors
#   interacting on the instance's topic within a discussion.
#   Column $t+56$ represents the AT at relative time $t$, abbreviated $AT_t$.
# - Number of Authors (NA) (columns $[64,71]$): Measures the number of authors
#   interacting on the instance's topic at time $t$.
#   Column $t+64$ represents the NA at relative time $t$, abbreviated $NA_t$.
# - Average Discussion Length (ADL) (columns $[72,79]$): Measures the average length of
#   a discussion belonging to the instance's topic.
#   Column $t+72$ represents the ADL at relative time $t$, abbreviated $ADL_t$.
# - Attention Level Measured with Number of Authors (AS(NA)) (columns $[80,87]$):
#   Measures the attention paid to the instance's topic.
#   Column $t+80$ represents the AS(NA) at relative time $t$, abbreviated $AS(NA)_t$.
# - Attention Level Measured with Number of Contributions (AS(NAC))
#   (columns $[88,95]$): Measures the attention paid to the instance's topic.
#   Column $t+88$ represents the AS(NAC) at relative time $t$, abbreviated $AS(NAC)_t$.
# - Feature to Predict (column $96$): Measures the mean ND, which describes the
#   popularity of the instance's topic.
#   Column $96$ represents the mean ND, abbreviated $ND$.
with pathlib.Path.open("TomsHardware.data") as csvfile:
    toms_data = np.array(list(csv.reader(csvfile)), dtype=np.float64)

# At this point, there are a few issues that need to be resolved in order to create a
# merged dataset:
#
# 1. The order of the features is different in each dataset (and one is missing from
#    Twitter's):
#
#    | Feature | Twitter | Tom's Hardware |
#    | ------- | ------- | -------------- |
#    | NCD     | 1       | 1              |
#    | AI      | 2       | 4              |
#    | AS(NA)  | 3       | 11             |
#    | BL      | 4       | 2              |
#    | NAC     | 5       | 5              |
#    | AS(NAC) | 6       | 12             |
#    | CS      | 7       | 7              |
#    | AT      | 8       | 8              |
#    | NA      | 9       | 9              |
#    | ADL     | 10      | 10             |
#    | NAD     | 11      | 3              |
#    | ND      | N/A     | 6              |
#
# 2. The scale of Twitter's data is much larger. For example, the mean $NCD_0$ for
#    Twitter is $140.340$, while the mean $NCD_0$ for Tom's Hardware is $1.159$.
# 3. Twitter's data is taken over seven time periods, while Tom's Hardware's data is
#    taken over eight.
# 4. Each dataset was used to predict a different value, with Twitter's being used to
#    predict the mean NAD and Tom's Hardware's being used to predict the mean ND.
# 5. It is never explicitly stated how long each time period is or whether it's the
#    same length between datasets, but it is implied that both datasets use intervals
#    of one week. Additionally, since both datasets were uploaded in the same month,
#    it can be assumed that each $t$ represents the same time period between the
#    datasets.

# Start by removing extraneous columns from each dataset:
#
# From Tom's Hardware:
# - Each column for data in the eighth time period: 7, 15, 23, 31, 39, 47, 55, 63, 71,
#   79, 87, 95.
# - Each column of ND data: 40, 41, 42, 43, 44, 45, 46, 47.
# - The prediction column: 96.
#
# From Twitter:
# - The prediction column: 77.
twitter_data = np.delete(twitter_data, (77), axis=1)
toms_data = np.delete(
    toms_data,
    (7, 15, 23, 31, 39, 40, 41, 42, 43, 44, 45, 46, 47, 55, 63, 71, 79, 87, 95, 96),
    axis=1,
)

# Then, rearrange the columns in the Tom's Hardware dataset to match the order of the
# columns in the Twitter dataset. Refer to the table above to see how these values were
# determined.
# fmt: off
toms_data = toms_data[:, (
     0,  1,  2,  3,  4,  5,  6,
    21, 22, 23, 24, 25, 26, 27,
    70, 71, 72, 73, 74, 75, 76,
     7,  8,  9, 10, 11, 12, 13,
    28, 29, 30, 31, 32, 33, 34,
    42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69,
    14, 15, 16, 17, 18, 19, 20,
    35, 36, 37, 38, 39, 40, 41,
)]
# fmt: on

# Once the columns match, we can scale certain parts of the Tom's Hardware dataset so
# that it can be meaningfully compared to the Twitter dataset. There are two
# immediately apparent methods for this:
#
# 1. Scale the columns in the Tom's Hardware dataset by the ratio between the maximum
#    value in that column and the maximum value in the corresponding column from the
#    Twitter dataset. This approach will allow us to avoid modifying the (large)
#    Twitter dataset, and will keep the values of the Twitter dataset real.
# 2. Normalize all columns in both datasets to the same range. This approach will
#    allow us to more easily compare axes between themselves.
#
# We will prefer the second approach because we intend to merge the datasets.
twitter_data /= twitter_data.max(axis=0)
toms_data /= toms_data.max(axis=0)

# Finally, we can concatenate the datasets.
merged_data = np.concatenate((twitter_data, toms_data), axis=0)
