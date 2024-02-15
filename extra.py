# # Original data
# data = {'Word': ['Man', 'Eats', 'Meat']}
# words_list = data['Word']
#
# # Get unique words
# unique_words = []
#
# for words in words_list:
#     for word in words.split():
#         if word not in unique_words:
#             unique_words.append(word)
#
# # Create a binary matrix using lists and dictionaries
# original_matrix = []
#
# # Update matrix values based on word presence for the original data
# for words in words_list:
#     word_count = {word: 0 for word in unique_words}
#     for word in words.split():
#         word_count[word] += 1
#     original_matrix.append(word_count)
#
# # Print the original matrix with words in the desired order
# header = "\t".join(unique_words)
# print("Original Matrix:")
# print(header)
#
# for row in original_matrix:
#     row_str = "\t".join(str(row[word]) for word in unique_words)
#     print(row_str)
#
# # Handle user input
# user_input = input("Enter a string: ")
#
# # Calculate one-hot encoded vectors for each word in the user input
# user_input_vectors = []
#
# for word in user_input.split():
#     # Create a binary matrix for the updated data
#     updated_matrix = []
#
#     # Update matrix values based on word presence for the updated data
#     word_count = {word: 0 for word in unique_words}
#     word_count[word] = 1
#     updated_matrix.append(word_count)
#
#     # Print the updated matrix for the current word
#     # print(f"\nUpdated Matrix for {word}:")
#     header = "\t".join(unique_words)
#     # print(header)
#
#     for row in updated_matrix:
#         row_str = "\t".join(str(row[word]) for word in unique_words)
#         print(row_str)
#
#     # Create a one-hot encoded vector for the current word using a dictionary
#     user_input_vector_dict = {word: '1' if word == current_word else '0' for current_word in unique_words}
#
#     # Append the one-hot encoded vector to the list
#     user_input_vectors.append(user_input_vector_dict)
# # Create a one-hot encoded vector for each word in the user input
# user_input_vectors = []
#
# for word in user_input.split():
#     # Create a one-hot encoded vector for the current word using a list comprehension
#     user_input_vector = [int(current_word == word) for current_word in unique_words]
#
#     # Append the one-hot encoded vector to the list
#     user_input_vectors.append(user_input_vector)
#
# # Print the result for each word in the user input as vectors
# print("\nOne-Hot Encoded Vectors for User Input:")
# print(user_input_vectors)
# # This modification avoids using the map function and instead utilizes a list comprehension to create the one-hot encoded vectors for each word in the user input.
#
#
#
#
#
# #
# import numpy as np
# a=np.array([[1,2,3]])
# r1=np.repeat(a,3,axis=1)# repeating thr array or 1 axis
#
# print(r1)
import numpy as np
from numpy import random
# printing the patterns  of matrixs
# output = np.ones((5,5))
# z=np.zeros((3,3))
# z[1,1]=9
# print(output)
# print(z)
# print(" print the pattern ")
# output[1:4,1:4]=z
# print(output)

# be carefull while copying the arrays because if we directly copyinh the array it will reploicated to the original array also
# a=np.array([1,2,3])
# b=np.copy(a)# but hwen we use the copy method it wil not replicate the original one
# b[0]=1000
# print(a)
# print(f' printing the b\n {b}')


# Mathematics function in numpy
# a = np.array([1, 2, 3])
# print(np.tan(a)) # all the mathematics fuynctions are allowed sin coss

# linear algebra
# a= np.ones((2,3))
# print(a)
#
# b=np.full((3,2),2)
# print(b)
# # multiplication of matrics
# print("multiplication the matrics ")
# c=np.matmul(a,b)
# print(c)

# finding the determinant
# a=np.identity((3),dtype='int')
# print(np.linalg.det(a)) # getting the deteminant


# stastistics
# a=np.array([[1,2,3,4,5,6],
#             [7,8,9,10,11,12]])
# b=np.min(a,axis=1) # on the basis of axis values
# c=np.max(a,axis=0)
# print(f'array{a}')
# print(f'maximum{c}')
# print(f'minimum{b}')
# print(np.sum (a,axis=1)) # axis value


# Reorganizing arrays :>>>>>>>>>>>>.
# before=np.array([[1,2,3,4],[1,5,6,4]])
# print(before.shape)
# after=before.reshape((2,-1)) # rshaping the arrays
# print(after)


# # vertically stacking  the vectors
# v1=np.array([1,2,3,4])
# v2=np.array([1,5,6,4])
# print(np.hstack([v1,v2]))
# print(np.vstack([v1,v2]))

# Boolean masking and advanced indexing
# fd=np.genfromtxt('textdata.txt',delimiter=',') # reading thr data from file
# filedata=fd.astype('int')
# print(filedata) # this return the value in boolean fdatatyoe
# print(filedata[filedata<10]) # thius retur the values
# a=np.array([1,2,3,4,5,6,7])
# print(a[[1,3,6]]) # array indexing
# print("file data");
# b=np.all(filedata>10,axis=0)
# print(b)


# numpy data distribution:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
# from numpy import random
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# # a=random.choice([3,5,6,8],p=[0.1,0.1,0.2,0.6],size=(2,3))
# # print(a)
# #
# # arr=np.array([1,2,3,4])
# # # random.shuffle(arr) # shuffling the array
# # print(arr)
# # random.permutation(arr)
# # print(arr)
# # x=random.randint(20,40,size=(100))
# sns.distplot(random.normal(loc=50, scale=5, size=1000), hist=False, label='normal')
# sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')

# plt.show()
# print(x)






# pandas >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
# import numpy as np
# import pandas as pd
# dataset={
#     'name':['akash','sumit','rohan'],
#     'age':[10,29,33],
#     'marks':[122,123,133]
# }
# df=pd.DataFrame(dataset)
# # for saving the t file in the directory
# # df.to_csv('datafile',index=False) # and there is no index column becauser index = false
# # print(df.head(2)) # for printing the first two records
# # print(df.tail(2))# for printing the elast two record
# # print(df.describe()) # it tells the description of the data
# # new_age = 32  # Repace with the desired new age
# # df.loc[df['name'] == 'akash', 'age'] = new_age # for changing ther value
# print(df)
# ser=pd.Series(np.random.rand())
# print(ser)


#
# # date 18 january 2000
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Creating a 3D array
# array_3d = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#                     [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
#                     [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
#
# print( array_3d)
#
# depth_slice = array_3d[0, :, :]
# print("Depth slice:")
# print(depth_slice)
#
# row_slice = array_3d[:, 1, :]
# print("\nRow slice:")
# print(row_slice)
#
#
# column_slice = array_3d[:, :, 2]
# print("\nColumn slice:")
# print(column_slice)
# print(" printing a particular element ")
# ele=array_3d[1,1,:,]
# print(ele)

# # sns.distplot(column_slice)
# # plt.show()
#


# arr1=np.array([1,3,1,3])
# arr2=np.array([[1],
#               [2],
#               [3]])
# # print((arr1))
# sh=arr1+arr2
# print(sh)
# print(sh.shape)
# from numpy import random
# x=random.randint(10,100000000,size=(3,3),dtype='int32')
# print(x)



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)

y = 4 + 3 * X + np.random.randn(100, 1)
print(X,y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
plt.scatter(X_train,y_train)
plt.show()

# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

