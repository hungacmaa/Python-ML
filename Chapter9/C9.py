import pandas as pd
import numpy as np
import operator
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("knn.csv")
sns.lmplot('x', 'y', data=data,
hue='c', palette='Set1',
fit_reg=False, scatter_kws={"s": 70})
plt.show()

#---to calculate the distance between two points---
def euclidean_distance(pt1, pt2, dimension):
    distance = 0
    for x in range(dimension):
        distance += np.square(pt1[x] - pt2[x])
    return np.sqrt(distance)

#---our own KNN model---
def knn(training_points, test_point, k):
    distances = {}
    #---the number of axes we are dealing with---
    dimension = test_point.shape[1]
    #--calculating euclidean distance between each
    # point in the training data and test data
    for x in range(len(training_points)):
        dist = euclidean_distance(test_point, training_points.iloc[x], dimension)
        #---record the distance for each training points---
        distances[x] = dist[0]
    #---sort the distances---
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    #---to store the neighbors---
    neighbors = []
    #---extract the top k neighbors---
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    #---for each neighbor found, find out its class---
    class_counter = {}
    for x in range(len(neighbors)):
        #---find out the class for that particular point---
        cls = training_points.iloc[neighbors[x]][-1]
        if cls in class_counter:
            class_counter[cls] += 1
        else:
            class_counter[cls] = 1
    #---sort the class_counter in descending order---
    sorted_counter = sorted(class_counter.items(),
    key=operator.itemgetter(1),
    reverse=True)
    #---return the class with the most count, as well as the
    #neighbors found---
    return(sorted_counter[0][0], neighbors)

#---test point---
test_set = [[3,3.9]]
test = pd.DataFrame(test_set)
cls,neighbors = knn(data, test, 5)
print("Predicted Class: " + cls)


#---generate the color map for the scatter plot---
#---if column 'c' is A, then use Red, else use Blue---
colors = ['r' if i == 'A' else 'b' for i in data['c']]
ax = data.plot(kind='scatter', x='x', y='y', c = colors)
plt.xlim(0,7)
plt.ylim(0,7)
#---plot the test point---
plt.plot(test_set[0][0],test_set[0][1], "yo", markersize='9')
for k in range(7,0,-2):
    cls,neighbors = knn(data, test, k)
    print("============")
    print("k = ", k)
    print("Class", cls)
    print("Neighbors")
    print(data.iloc[neighbors])
    
    furthest_point = data.iloc[neighbors].tail(1)
    
    #---draw a circle connecting the test point
    #and the furthest point---
    radius = euclidean_distance(test, furthest_point.iloc[0], 2)
    
    #---display the circle in red if classification is A,
    # else display circle in blue---
    c = 'r' if cls=='A' else 'b'
    circle = plt.Circle((test_set[0][0], test_set[0][1]), radius, color=c, alpha=0.3)
    ax.add_patch(circle)
    
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from sklearn import svm, datasets
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = iris.data[:, :2] # take the first two features
y = iris.target
#---plot the points---
colors = ['red', 'green', 'blue']
for color, i, target in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X[y==i, 0], X[y==i, 1], color=color, label=target)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Scatter plot of Sepal width against Sepal length')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
k = 1
#---instantiate learning model---
knn = KNeighborsClassifier(n_neighbors=k)
#---fitting the model---
knn.fit(X, y)
#---min and max for the first feature---
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#---min and max for the second feature---
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#---step size in the mesh---
h = (x_max / x_min)/100
#---make predictions for each of the points in xx,yy---
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
#---draw the result using a color plot---
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Accent, alpha=0.8)
#---plot the training points---
colors = ['red', 'green', 'blue']
for color, i, target in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X[y==i, 0], X[y==i, 1], color=color, label=target)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title(f'KNN (k={k})')
plt.legend(loc='best', shadow=False, scatterpoints=1)
predictions = knn.predict(X)
#--classifications based on predictions---
print(np.unique(predictions, return_counts=True))