import pandas as pd

from scipy import misc
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 
samples = []
samples2 = []

#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 
for image_path in glob.glob("Datasets/ALOI/32/*.png"):
    image = misc.imread(image_path)
    image = image[::2,::2]
    X = image.reshape(-1)
    samples.append(X)
    
for image_path in glob.glob("Datasets/ALOI/32i/*.png"):
    image = misc.imread(image_path)
    image = image[::2,::2]
    X = image.reshape(-1)
    samples2.append(X)
    
df = pd.DataFrame(samples)
dfi = pd.DataFrame(samples2)
dfi['col'] = 'blue'
df_merge = pd.concat([df,dfi])

from sklearn.manifold import Isomap

iso = Isomap(n_components=3)
iso.fit(df_merge.drop('col',axis=1))
T = iso.fit_transform(df_merge.drop('col',axis=1))

iso2 = Isomap(n_components=3)
iso2.fit(df)
Ti = iso2.fit_transform(dfi)

#

# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 

plt.scatter(T[:72,0],T[:72,1],color='red')
plt.scatter(T[72:,0],T[72:,1],color='blue')

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(T[:72,0],T[:72,1],T[:72,2],color='red')
ax.scatter(T[72:,0],T[72:,1],T[72:,2],color='blue')
#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 



#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 



#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 




#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 



plt.show()

