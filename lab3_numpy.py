"""CSC350 Lab 3: NumPy hands-on
Name: Gurria Shabir
Roll number:24-UON_0326

Run this file after every task:  py lab3_numpy.py
Commit after every task (git add, git commit -m "Task N done").s
"""
import numpy as np

print("NumPy version:", np.__version__)

# =====================================================================
# Task 1: Creating arrays (2 marks)
# =====================================================================
print("\n=== Task 1 ===")
# 1a. Make an array `temps` with this week's temperatures:
#     31, 28, 35, 30, 33, 29, 36
temps = np.array([31,28,35,30,33,29,36])
#     Print it, then print its dtype, shape and size.
print("Temperature",temps)
print("Dtype",temps.dtype)
print("Shape",temps.shape)
print("Size",temps.size)

# 1b. Make `months`: the numbers 1 to 12 as a 3 x 4 table
#     (use np.arange and reshape). Print it and its shape.
months = np.arange(1,13).reshape(3,4)
print(months)
print("Shape of months",months.shape)

# 1c. Print an array of five zeros, and a 2 x 3 array of ones.
print("Five Zeros Array",np.zeros(5))
print("Ones arrays", np.ones((2,3)))


# 1d. Make a random generator with seed 42 and roll 10 dice
#     (whole numbers from 1 to 6). Print the result.
rng =np.random.default_rng(42)
dice = rng.integers(1,7,size=10)
print("Dice",dice)

# =====================================================================
# Task 2: Indexing and masks (2.5 marks)  (use `temps` from Task 1)
# =====================================================================
print("\n=== Task 2 ===")
# 2a. Print the first and the last temperature.
print("First Temp",temps[0])
print("Last Temp",temps[-1])
# 2b. Print the weekend temperatures (the last two days).
print("last two days Temp",temps[-2])
# 2c. Print the temperatures in reverse order.
print("Reverse Temp",temps[::-1])
# 2d. Print only the hot days (above 32), how many there are,
#     and their average.  No loops!
hot = temps[temps > 32]
print("Hot days",hot)
print("No of Hot days",len(hot))
print("Average",hot.mean())
# 2e. Print the temperatures from 29 to 33 (both included). Hint: &
selected = temps[(temps >=29)& (temps <=33)]
print("Selected",selected)
# 2f. Use np.where to label each day "Hot" (33 or more) or "OK".

labels = np.where(temps >= 33 , "Hot" , "OK")
print(labels)
# =====================================================================ssz
# Task 3: Gradebook (3 marks)
# Rows = students, columns = 4 subjects.
# =====================================================================
print("\n=== Task 3 ===")
names = np.array(["Ali", "Sara", "Usman", "Hina", "Bilal"])
marks = np.array([[78, 85, 62, 90],
                  [92, 88, 95, 79],
                  [45, 60, 48, 70],
                  [88, 94, 91, 85],
                  [65, 42, 75, 58]])
# 3a. Print the shape of marks.
print(marks.shape)
# 3b. Print Usman's marks (one row) and all marks of subject 3
#     (column index 2).
usman_idx = np.where(names == "Usman")[0][0]
print(marks[usman_idx])
print(marks[:, 2])
# 3c. Print the average of each subject and the average of each student.
print(marks.mean(axis=0))
print(marks.mean(axis=1))
# 3d. Print the NAME of the student with the highest average.
print(names[marks.mean(axis=1).argmax()])
#     Hint: argmax gives a position; use it on `names`.
# 3e. Subject weights are 0.3, 0.3, 0.2, 0.2. Print each student's
#     weighted total, rounded to 1 decimal. (Broadcasting!)
weights = np.array([0.3, 0.3, 0.2, 0.2])
print(np.round((marks * weights).sum(axis=1), 1))
# 3f. Add 5 grace marks to everyone, but no mark may go above 100.
#     Print the new table. Hint: np.clip
print(np.clip(marks + 5, 0, 100))
# 3g. Print how many students failed (below 50) in each subject.
print((marks < 50).sum(axis=0))


# =====================================================================
# Task 4: An image is an array (2.5 marks)
# A 5 x 5 grayscale image: 0 = black, 255 = white.
# =====================================================================
print("\n=== Task 4 ===")
img = np.array([[ 10,  10, 200,  10,  10],
                [ 10,  10, 220,  10,  10],
                [210, 230, 250, 230, 210],
                [ 10,  10, 220,  10,  10],
                [ 10,  10, 200,  10,  10]])
# 4a. Print the shape, the darkest value and the brightest value.
print(img.shape, img.min(), img.max())
# 4b. Print the inverted image (white becomes black). Hint: 255 - ...
print(255 - img)
# 4c. Print a black-and-white version: 1 where the pixel is above 128,
#     else 0. Hint: a mask, then .astype(int)
print((img > 128).astype(int))
# 4d. Print row 2 normalized to the range 0 to 1, rounded to 2 decimals.
print(np.round(img[2] / 255, 2))
# 4e. Print the 3 x 3 centre of the image (crop).
print(img[1:4, 1:4])

