# Hannah_Folse_Python_Portfolio
This is the portfolio of python code that I learned in BISC 450C.



## Using Jupyter Notebooks (1 and 2)

```python
%matplotlib inline 

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style = "darkgrid")
```


```python
df = pd.read_csv('/home/student/Desktop/classroom/myfiles/notebooks/fortune500.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25495</td>
      <td>2005</td>
      <td>496</td>
      <td>Wm. Wrigley Jr.</td>
      <td>3648.6</td>
      <td>493</td>
    </tr>
    <tr>
      <td>25496</td>
      <td>2005</td>
      <td>497</td>
      <td>Peabody Energy</td>
      <td>3631.6</td>
      <td>175.4</td>
    </tr>
    <tr>
      <td>25497</td>
      <td>2005</td>
      <td>498</td>
      <td>Wendy's International</td>
      <td>3630.4</td>
      <td>57.8</td>
    </tr>
    <tr>
      <td>25498</td>
      <td>2005</td>
      <td>499</td>
      <td>Kindred Healthcare</td>
      <td>3616.6</td>
      <td>70.6</td>
    </tr>
    <tr>
      <td>25499</td>
      <td>2005</td>
      <td>500</td>
      <td>Cincinnati Financial</td>
      <td>3614.0</td>
      <td>584</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ['year', 'rank', 'company', 'revenue', 'profit']
```


```python
df. head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    25500




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit      object
    dtype: object




```python
non_numeric_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numeric_profits].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>228</td>
      <td>1955</td>
      <td>229</td>
      <td>Norton</td>
      <td>135.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>290</td>
      <td>1955</td>
      <td>291</td>
      <td>Schlitz Brewing</td>
      <td>100.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>294</td>
      <td>1955</td>
      <td>295</td>
      <td>Pacific Vegetable Oil</td>
      <td>97.9</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>296</td>
      <td>1955</td>
      <td>297</td>
      <td>Liebmann Breweries</td>
      <td>96.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>352</td>
      <td>1955</td>
      <td>353</td>
      <td>Minneapolis-Moline</td>
      <td>77.4</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(df.profit[non_numeric_profits])
```




    {'N.A.'}




```python
len(df.profit[non_numeric_profits])
```




    369




```python
bin_sizes, _, _ = plt.hist(df.year[non_numeric_profits], bins = range(1955, 2006))
```


![png](output_11_0.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/113f06c8-1d51-42a7-8944-ed3d3a30dc09)



```python
df = df.loc[~non_numeric_profits]
df.profit = df.profit.apply(pd.to_numeric)
```


```python
len(df)
```




    25131




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit     float64
    dtype: object




```python
group_by_year = df.loc[:, ['year', 'revenue', 'profit']].groupby('year')

avgs = group_by_year.mean()

x = avgs.index
y1 = avgs.profit

def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x = 0, y = 0)
```


```python
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1955 to 2005', 'Profit (millions)')
```


![png](output_16_0.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/930eb5c2-53e7-4e9b-a0aa-e8453527d3ea)



```python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'Increase in mean Fortune 500 company revenue from 1955 to 2005', 'Revenue (millions)')
```


![png](output_17_0.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/e03adb8c-1d81-493b-84d7-5aeada43fad9)



```python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha = 0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots(ncols = 2)
title = 'Increase in mean and std fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'revenue', 'Revenue (millions)')

fig.set_size_inches(14,4)
fig.tight_layout()
```


![png](output_18_0.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/f5deafdc-53b0-4719-aca7-6016c56d2c02)







## Python Fundamentals 

```python
# any python interpreter can be used as a calculator: 
3 + 5 + 4
```




    12




```python
3 + 5 * 4
```




    23




```python
# lets save a value to a variable 
weight_kg = 60 #Interger 
```


```python
print(weight_kg)
```

    60



```python
# weight0 = valid 
# 0weight = invalid 
# weight and Weight are different 
```


```python
# types of data 
# There are 3 common types of data 
# Interger numbers 
# Flooting point numbers 
# Strings 
```


```python
weight_kg = 60.3 #Flooting point number 
```


```python
patient_name = "Jonh Smith" #String comprised of letters 

patient_id = '001' #String comprised of numbers  

```


```python
# Use variables in python 

weight_lb = 2.2 * weight_kg 
print(weight_lb)

```

    132.66



```python
# add a prefix to patient id 

patient_id = 'inflam_' + patient_id

print(patient_id)
```

    inflam_001



```python
# combine print statments 

print(patient_id, 'weight in kilograms:', weight_kg)
```

    inflam_001 weight in kilograms: 60.3



```python
# we can call a function inside another function 

print(type(60.3))

print(type(patient_id))

```

    <class 'float'>
    <class 'str'>



```python
# Calculations inside the print function 

print('weight in lbs:', 2.2 * weight_kg)
```

    weight in lbs: 132.66



```python
print(weight_kg)

```

    60.3



```python
weight_kg = 65.0 

print('weight in kilograms is now', weight_kg)
```

    weight in kilograms is now 65.0






## Analyzing Patient Data (1 and 2)
In this analysis we looked at inflamation data for multiple patients

```python
import numpy
```


```python
numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',') #saving as a varible 
```


```python
print(data)
```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]



```python
print(type(data))
```

    <class 'numpy.ndarray'>



```python
print(data.shape) # gives rows and columns 
```

    (60, 40)



```python
print('first value in data:', data[0,0]) # start counting at 0 not 1  
```

    first value in data: 0.0



```python
print('middle value indata:', data[29,19])
```

    middle value indata: 16.0



```python
print(data[0:4,0:10]) #first 4 rows and first 10 colums, goes up to 4 but does not include 4
```

    [[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
     [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
     [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
     [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]



```python
print(data[5:10,0:10])
```

    [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
     [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
     [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
     [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
     [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]



```python
print(data)
```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]



```python
small = data[:3, 36:] #anything to 3 and 36 to anthing (rows 0,1,2 and columns 37,38,39,40)
```


```python
print('small is:', small)
```

    small is: [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]



```python
print(small)
```

    [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]



```python
# use a numpy function 
print(numpy.mean(data))
```

    6.14875



```python
maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data) 

print('maximum inflammation:', maxval)
```

    maximum inflammation: 20.0



```python
print(maxval, minval, stdval)
```

    20.0 0.0 4.613833197118566



```python
print(maxval)
print(minval)
print(stdval)

```

    20.0
    0.0
    4.613833197118566



```python
print('maximum inflammation:',maxval)
print('minimum inflammation:',minval)
print('standard deviation inflammation:',stdval)
```

    maximum inflammation: 20.0
    minimum inflammation: 0.0
    standard deviation inflammation: 4.613833197118566



```python
# sometimes fwe want to look at variation in statistical values, 
# such as maximum inflammation oer patient, or average from day one 

patient_0 = data[0, :] # 0 on the first axis (rows), evergything on the second (columns)

print('maxium inflammation for patient 0:', numpy.amax(patient_0))
```

    maxium inflammation for patient 0: 18.0



```python
print('maximum inflammation for patient 2:',numpy.amax(data[2,:]))
```

    maximum inflammation for patient 2: 19.0



```python
print(numpy.mean(data, axis = 0)) #average per day (column) for all patients 
```

    [ 0.          0.45        1.11666667  1.75        2.43333333  3.15
      3.8         3.88333333  5.23333333  5.51666667  5.95        5.9
      8.35        7.73333333  8.36666667  9.5         9.58333333 10.63333333
     11.56666667 12.35       13.25       11.96666667 11.03333333 10.16666667
     10.          8.66666667  9.15        7.25        7.33333333  6.58333333
      6.06666667  5.95        5.11666667  3.6         3.3         3.56666667
      2.48333333  1.5         1.13333333  0.56666667]



```python
print(numpy.mean(data, axis = 0).shape)
```

    (40,)



```python
print(numpy.mean(data, axis = 1)) #average per patient (row)
```

    [5.45  5.425 6.1   5.9   5.55  6.225 5.975 6.65  6.625 6.525 6.775 5.8
     6.225 5.75  5.225 6.3   6.55  5.7   5.85  6.55  5.775 5.825 6.175 6.1
     5.8   6.425 6.05  6.025 6.175 6.55  6.175 6.35  6.725 6.125 7.075 5.725
     5.925 6.15  6.075 5.75  5.975 5.725 6.3   5.9   6.75  5.925 7.225 6.15
     5.95  6.275 5.7   6.1   6.825 5.975 6.725 5.7   6.25  6.4   7.05  5.9  ]



## Analyzing Patient Data (3)

```python
import numpy
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```


```python
# heat map of patient inflammation over time 
import matplotlib.pyplot # import plotting library 
image = matplotlib.pyplot.imshow(data) # makeing the image 
matplotlib.pyplot.show() # show 
```


![png](output_1_0.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/b0d2d372-f44f-4870-a626-0324fb0f8014)



```python
# average inflammation over time 

avg_inflammation = numpy.mean(data, axis = 0)
avg_plot = matplotlib.pyplot.plot(avg_inflammation)
matplotlib.pyplot.show()
```


![png](output_2_0.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/8bcecf2c-127e-472b-b3fb-4c8e70493ebb)



```python
max_plot = matplotlib.pyplot.plot(numpy.amax(data, axis = 0)) 
matplotlib.pyplot.show()
```


![png](output_3_0.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/5927ced7-5b70-4525-851d-49dccb0e0f00)



```python
min_plot = matplotlib.pyplot.plot(numpy.amin(data, axis = 0)) 
matplotlib.pyplot.show()
```


![png](output_4_0.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/472575b1-23b3-48c9-9097-3258dec54d04)



```python
fig = matplotlib.pyplot.figure(figsize = (10.0, 3.0)) # making a figure 
                                                    # and seting a figure size 10 by 3 


# making plots within the figure 
axes1 = fig.add_subplot(1, 3, 1) #first plot, 1 row by 3 columns first position 
axes2 = fig.add_subplot(1, 3, 2)
axes3 = fig.add_subplot(1, 3, 3)

axes1.set_ylabel('Inflammation')
axes1.set_xlabel('Time')
axes1.set_title('Average')
axes1.plot(numpy.mean(data, axis = 0))

axes2.set_ylabel('Inflammation')
axes2.set_xlabel('Time')
axes2.set_title('Max')
axes2.plot(numpy.amax(data, axis = 0))

axes3.set_ylabel('Inflammation')
axes3.set_xlabel('Time')
axes3.set_title('Min')
axes3.plot(numpy.amin(data, axis = 0))

fig.tight_layout()

matplotlib.pyplot.savefig('inflammation.png')
matplotlib.pyplot.show()
```


![png](output_5_0.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/1b7ff0b5-23eb-49a1-baf2-f29f94430c48)




## Storing Values in Lists 
```python
odds = [1, 3, 5, 7]
print('oods are:', odds)
```

    oods are: [1, 3, 5, 7]



```python
print('first element:', odds[0])
print('last element:', odds[3])
print('"-1" element:', odds[-1]) # counting from the end of a list 
print('"-2" element:', odds[-2])
```

    first element: 1
    last element: 7
    "-1" element: 7
    "-2" element: 5



```python
names = ['Curie', 'Darwing', 'Turing'] # Typo un Darwin's name 

print('names is orginally:', names )

names[1] = 'Darwin' # Correct the name

print('final value of names:', names)
```

    names is orginally: ['Curie', 'Darwing', 'Turing']
    final value of names: ['Curie', 'Darwin', 'Turing']



```python
#name = 'Darwin'
#name[0] = 'd'
```


```python
odds.append(11)
print('odds after adding a value:', odds)
```

    odds after adding a value: [1, 3, 5, 7, 11]



```python
remove_elements = odds.pop(0)

print('odds after removing the first:', odds)
print('removed element', remove_elements)
```

    odds after removing the first: [3, 5, 7, 11]
    removed element 1



```python
odds.reverse()
print('odds after reversing', odds)
```

    odds after reversing [11, 7, 5, 3]



```python
odds = [3, 5, 7]
primes = odds 
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7, 2]



```python
odds = [3, 5, 7]
primes = list(odds)
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7]



```python
binomial_name = "Drosophila melanogaster"
group = binomial_name[0:10]
print('group:', group)

species = binomial_name[11:23]
print('species:', species)

chromosomes = ['X', 'Y', '2', '3', '4']
autosomes  = chromosomes[2:5]
print('autosomes:', autosomes)

last = chromosomes[-1]
print('last', last)
```

    group: Drosophila
    species: melanogaster
    autosomes: ['2', '3', '4']
    last 4



```python
date = 'Monday 4 January 2023'
day = date[0:6]
print('Using 0 to begin range:', day)
day = date[:6] #assumes zero 
print('Omitting beginning index:', day)
```

    Using 0 to begin range: Monday
    Omitting beginning index: Monday



```python
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 
          'Oct',  'Nov', 'Dec']

sond = months[8:12]
print('With known last position:', sond)

sond = months[8:len(months)]
print('Using length to get last entry:', sond)

sond = months[8:]
print('Omitting ending index:', sond)
```

    With known last position: ['Sep', 'Oct', 'Nov', 'Dec']
    Using length to get last entry: ['Sep', 'Oct', 'Nov', 'Dec']
    Omitting ending index: ['Sep', 'Oct', 'Nov', 'Dec']



## Using Loops
```python
odds = [1, 3, 5, 7]
```


```python
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5
    7



```python
odds = [1, 3, 5]
print(odds[0])
print(odds[1])
print(odds[2])
#print(odds[3])
```

    1
    3
    5



```python
odds = [1, 3, 5, 7]

for num in odds: # for loop
    print(num)
```

    1
    3
    5
    7



```python
odds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

for num in odds: 
    print(num)
```

    1
    3
    5
    7
    9
    11
    13
    15
    17
    19



```python
length = 0 
names = ['Curie', 'Darwin', 'Turing']
for value in names: 
    length = length + 1 
print('There are', length, 'names in the list')
```

    There are 3 names in the list



```python
name = "Rosalind"

for name in ['Curie', 'Darwin', 'Turing']: 
    print(name)
    
print('after the loop, name is', name)
```

    Curie
    Darwin
    Turing
    after the loop, name is Turing



```python
print(len([0,1,2,3]))
```

    4



```python
name = ['Curie', 'Darwin', 'Turing']

print(len(name))
```

    3


## Using Multiple Files 
```python
import glob 
```


```python
print(glob.glob('inflammation*.csv'))
```

    ['inflammation-10.csv', 'inflammation-09.csv', 'inflammation-11.csv', 'inflammation-06.csv', 'inflammation-05.csv', 'inflammation-08.csv', 'inflammation-01.csv', 'inflammation-07.csv', 'inflammation-04.csv', 'inflammation-03.csv', 'inflammation-02.csv', 'inflammation-12.csv']



```python
import glob
import numpy 
import matplotlib.pyplot

filenames = sorted(glob.glob('inflammation*.csv')) # varible called filename and sorted inflammation files
filenames = filenames[0:3] # take the first 3 and saved over varible 

for filename in filenames: 
    print(filename)
    
    data = numpy.loadtxt(fname=filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize = (10.0, 3.0))
    
    axes1 = fig.add_subplot(1,3,1) #graph  1, i row, 3 columns, position 1
    axes2 = fig.add_subplot(1,3,2)
    axes3 = fig.add_subplot(1,3,3)
    
    axes1.set_ylabel('Inflammation')
    axes1.set_xlabel('Time')
    axes1.set_title('Average')
    axes1.plot(numpy.mean(data, axis =0))
    
    axes2.set_ylabel('Inflammation')
    axes2.set_xlabel('Time')
    axes2.set_title('Max')
    axes2.plot(numpy.amax(data, axis =0)) 
    
    axes3.set_ylabel('Inflammation')
    axes3.set_xlabel('Time')
    axes3.set_title('Min')
    axes3.plot(numpy.amin(data, axis =0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
    
```

    inflammation-01.csv



![png](output_2_1.png) 
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/369e6efe-a1d4-41d0-8e65-6d8aac407226)



    inflammation-02.csv



![png](output_2_3.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/d9a0b025-20d3-4bc8-957e-a5a4aedbfbbc)


    inflammation-03.csv



![png](output_2_5.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/cbb0d062-e1a3-4b07-87dd-ad10f0d20c58)



## Making Choices (1)
```python
num = 37
if num > 100:
    print('greater')
else:
    print('not greater')
print('done')
```

    not greater
    done



```python
num = 53
print('before conditional..')
if num > 100:
    print(num, 'is greater than 100')
print ('...after conditional')
```

    before conditional..
    ...after conditional



```python
num = 0

if num > 0:
    print(num, 'is positive')
elif num == 0:
    print(num, 'is zero')
else:
    print(num, 'is negative')
```

    0 is zero



```python
if (1 > 0) and (-1 >= 0):
    print('both parts are true')
else: 
    print('at least one part of false')
```

    at least one part of false



```python
if (1 > 0) or (-1 >= 0):
    print('at elast one parts is true')
else: 
    print('both are false')
```

    at elast one parts is true



## Making Choices (2)
```python
import numpy
```


```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',') #saving as a varible 

```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
max_inflammation_0 = numpy.amax(data, axis = 0)[0]

```


```python
max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Saspictious looking maxima')
    
    
elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
    print('Minima add up to zero')
    
    
else:
    print('Seems ok')
```

    Saspictious looking maxima



```python
data = numpy.loadtxt(fname = 'inflammation-03.csv', delimiter = ',')

max_inflammation_0 = numpy.amax(data, axis = 0)[0]
max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Saspictious looking maxima')
elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
    print('minima add up to zero -> includes a healthy participant')    
else:
    print('Seems ok')
```

    minima add up to zero -> includes a healthy participant





## Functions (1)

```python
fahrenheit_val = 99
celsius_val = ((fahrenheit_val - 32) * (5/9))

print(celsius_val) 

```

    37.22222222222222



```python
fahrenheit_val = 43
celsius_val2 = ((fahrenheit_val - 32) * (5/9))

print(celsius_val2) 
```

    6.111111111111112



```python
def explicit_fahr_to_celsius(temp): 
    # Assign the converted value to a variable
    converted = ((temp - 32) *(5/9))
    
    #Return the values of the new varisble 
    return converted 

```


```python
def fahr_to_celsius(temp): 
    # Assign the converted more effeciently using the return 
    # functions without creating a new varible 
    return((temp - 32) *(5/9))
    
```


```python
fahr_to_celsius(32)
```




    0.0




```python
explicit_fahr_to_celsius(32)
```




    0.0




```python
print('Freezing point of water:', fahr_to_celsius(32), 'C')
print('Boiling point of water:', fahr_to_celsius(212), 'C')
```

    Freezing point of water: 0.0 C
    Boiling point of water: 100.0 C



```python
def celsius_to_kelvin(temp_c): 
    return temp_c + 273.15

print('freezing point of water in Kelvin:', celsius_to_kelvin(0.))
```

    freezing point of water in Kelvin: 273.15



```python
def fahr_to_kelvin(temp_f):
    temp_c = fahr_to_celsius(temp_f) 
    temp_k = celsius_to_kelvin(temp_c)
    return temp_k

print('Boiling point of water in Kelvin:', fahr_to_kelvin(212.0))


```

    Boiling point of water in Kelvin: 373.15



```python
print('Again, temperature in Kelving was:', temp_k)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-23-be27d2dc3254> in <module>
    ----> 1 print('Again, temperature in Kelving was:', temp_k)
    

    NameError: name 'temp_k' is not defined



```python
temp_kelving = fahr_to_kelvin(212.0)
print('Temperature in Kelving was:', temp_kelving)
```

    Temperature in Kelving was: 373.15



```python
temp_kelvin
```




    373.15




```python
def print_temperatures():
    print('Temperature in Fahrenheit was:', temp_fahr)
    print('Temperature in Kelvin was:', temp_kelvin)
    
    
temp_fahr = 212
temp_kelvin = fahr_to_kelvin(temp_fahr)

print_temperatures()
```

    Temperature in Fahrenheit was: 212
    Temperature in Kelvin was: 373.15


## Functions (2-4)

```python
import numpy
import matplotlib.pyplot
import glob
```


```python
def visualize(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize=(10, 3))
    
    axes1 = fig.add_subplot(1, 3, 1)
    axes2 = fig.add_subplot(1, 3, 2)
    axes3 = fig.add_subplot(1, 3, 3)
    
    
    axes1.set_title('Average')
    axes1.set_xlabel('Time')
    axes1.set_ylabel('Inflammation')
    axes1.plot(numpy.mean(data, axis=0))
    
    axes2.set_title('Max')
    axes2.set_xlabel('Time')
    axes2.set_ylabel('Inflammation')
    axes2.plot(numpy.amax(data, axis=0))
    
    axes3.set_title('Min')
    axes3.set_xlabel('Time')
    axes3.set_ylabel('Inflammation')
    axes3.plot(numpy.amin(data, axis=0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
    
```


```python
def detect_problems(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    if numpy.amax(data, axis = 0)[0] == 0 and numpy.amax(data, axis = 0)[20] ==20: 
        print('Suspicious looking naxima')
    elif numpy.sum(numpy.amin(data, axis = 0)) == 0: 
        print('Minima add up to zero')
    else: 
        print('Seems ok')
```


```python
filenames = sorted(glob.glob('inflammation*.csv'))

for filename in filenames:
    print(filename)
    visualize(filename) 
    detect_problems(filename)
```

    inflammation-01.csv



![png](output_3_1.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/2f2fe4b2-460b-4759-b34d-ca8db2875298)


    Suspicious looking naxima
    inflammation-02.csv



![png](output_3_3.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/6dbf714c-434a-4043-914c-95941e2c197f)


    Suspicious looking naxima
    inflammation-03.csv



![png](output_3_5.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/d3b45039-46fc-47ec-9b82-219aae4cc3e6)


    Minima add up to zero
    inflammation-04.csv



![png](output_3_7.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/b7ab1b2a-682d-4bc0-80dd-9986f75f0d78)


    Suspicious looking naxima
    inflammation-05.csv



![png](output_3_9.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/aa43a67c-aed7-4e07-9c49-5ff0deb08ada)


    Suspicious looking naxima
    inflammation-06.csv



![png](output_3_11.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/3ffc8045-123e-4ec2-b30b-5f089c84836d)


    Suspicious looking naxima
    inflammation-07.csv



![png](output_3_13.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/e9a5e9b7-ef40-4ea8-86b3-9a1c67cbbc34)


    Suspicious looking naxima
    inflammation-08.csv



![png](output_3_15.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/53a83115-a299-422c-b7bc-b98e04144458)


    Minima add up to zero
    inflammation-09.csv



![png](output_3_17.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/1fe5d89f-013c-42e1-9198-6544be4a725a)


    Suspicious looking naxima
    inflammation-10.csv



![png](output_3_19.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/75e70db6-ec6a-47e9-9c93-4635062404e4)


    Suspicious looking naxima
    inflammation-11.csv



![png](output_3_21.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/288746e6-b237-479e-b69a-4a3249422468)


    Minima add up to zero
    inflammation-12.csv



![png](output_3_23.png)
![image](https://github.com/HannahFolse/Python_Portfolio/assets/150103987/d4f8cd9c-bf53-4be7-8c59-495ba35243c6)


    Suspicious looking naxima



```python
def offset_mean(data, target_mean_value):
    return(data - numpy.mean(data)) + target_mean_value
```


```python
z = numpy.zeros((2,2))
print(offset_mean(z,3))
```

    [[3. 3.]
     [3. 3.]]



```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')

print (offset_mean(data, 0))
```

    [[-6.14875 -6.14875 -5.14875 ... -3.14875 -6.14875 -6.14875]
     [-6.14875 -5.14875 -4.14875 ... -5.14875 -6.14875 -5.14875]
     [-6.14875 -5.14875 -5.14875 ... -4.14875 -5.14875 -5.14875]
     ...
     [-6.14875 -5.14875 -5.14875 ... -5.14875 -5.14875 -5.14875]
     [-6.14875 -6.14875 -6.14875 ... -6.14875 -4.14875 -6.14875]
     [-6.14875 -6.14875 -5.14875 ... -5.14875 -5.14875 -6.14875]]



```python
print('original min, mean, and max are:', numpy.amin(data), numpy.mean(data), numpy.amax(data))
offset_data = offset_mean(data, 0)
print('min, mean and max of offset data are:',
    numpy.amin(offset_data),
    numpy.mean(offset_data), 
    numpy.amax(offset_data))
                                                   
```

    original min, mean, and max are: 0.0 6.14875 20.0
    min, mean and max of offset data are: -6.14875 2.842170943040401e-16 13.85125



```python
print('std dev berfoer and after:', numpy.std(data), numpy.std(offset_data))
```

    std dev berfoer and after: 4.613833197118566 4.613833197118566



```python
print('difference in standard deviation before and after:',
      numpy.std(data) - numpy.std(offset_data))
```

    difference in standard deviation before and after: 0.0



```python
# offset_mean(data, target_mean_value):
# return a new array containing the original data with its
# mean offset to match the desired value
# this data should be imputed as a measurements in colums and samples in rows

def offset_mean(data, target_mean_value):
    return(data - numpy.mean(data)) + target_mean_value
```


```python
def offset_mean(data, target_mean_value):
    """Return a new array containing the oringal data with its mean offset to match the desired value"""
    return(data - numpy.mean(data)) + target_mean_value
```


```python
help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the oringal data with its mean offset to match the desired value
    



```python
def offset_mean(data, target_mean_value):
    """Return a new array contain the original data with its 
    mean offset to match the desired value. 
    
    Examples
    ----------
    >>> Offset_mean([1,2,3], 0)
    array([-1., 0., 1.])
    """
    return(data - numpy.mean(data)) + target_mean_value
```


```python
help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array contain the original data with its 
        mean offset to match the desired value. 
        
        Examples
        ----------
        >>> Offset_mean([1,2,3], 0)
        array([-1., 0., 1.])
    



```python
numpy.loadtxt('inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
numpy.loadtxt('inflammation-01.csv',  ',')
```


    Traceback (most recent call last):


      File "/home/student/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3326, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)


      File "<ipython-input-62-822caac50a2a>", line 1, in <module>
        numpy.loadtxt('inflammation-01.csv',  ',')


      File "/home/student/anaconda3/lib/python3.7/site-packages/numpy/lib/npyio.py", line 1087, in loadtxt
        dtype = np.dtype(dtype)


      File "/home/student/anaconda3/lib/python3.7/site-packages/numpy/core/_internal.py", line 201, in _commastring
        newitem = (dtype, eval(repeats))


      File "<string>", line 1
        ,
        ^
    SyntaxError: unexpected EOF while parsing




```python
def offset_mean(data, target_mean_value = 0.0):
    """Return a new array contain the original data with its 
    mean offset to match the desired value, (0 by default).  
    
    Examples
    ----------
    >>> Offset_mean([1,2,3])
    array([-1., 0., 1.])
    """
    return(data - numpy.mean(data)) + target_mean_value
```


```python
test_data = numpy.zeros((2,2))

print(offset_mean(test_data, 3))
```

    [[3. 3.]
     [3. 3.]]



```python
print(offset_mean(test_data))
```

    [[0. 0.]
     [0. 0.]]



```python
def display(a=1, b=2, c=3):
    print('a', a, 'b', b, 'c', c)
    
print('no parameters:')
display()
print('one parameters:')
display(55)
print('teo parameters:')
display(55,66)
```

    no parameters:
    a 1 b 2 c 3
    one parameters:
    a 55 b 2 c 3
    teo parameters:
    a 55 b 66 c 3



```python
print('only setting the value of c')
display(c = 77)
```

    only setting the value of c
    a 1 b 2 c 77



```python
help(numpy.loadtxt)
```

    Help on function loadtxt in module numpy:
    
    loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)
        Load data from a text file.
        
        Each row in the text file must have the same number of values.
        
        Parameters
        ----------
        fname : file, str, or pathlib.Path
            File, filename, or generator to read.  If the filename extension is
            ``.gz`` or ``.bz2``, the file is first decompressed. Note that
            generators should return byte strings for Python 3k.
        dtype : data-type, optional
            Data-type of the resulting array; default: float.  If this is a
            structured data-type, the resulting array will be 1-dimensional, and
            each row will be interpreted as an element of the array.  In this
            case, the number of columns used must match the number of fields in
            the data-type.
        comments : str or sequence of str, optional
            The characters or list of characters used to indicate the start of a
            comment. None implies no comments. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is '#'.
        delimiter : str, optional
            The string used to separate values. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is whitespace.
        converters : dict, optional
            A dictionary mapping column number to a function that will parse the
            column string into the desired value.  E.g., if column 0 is a date
            string: ``converters = {0: datestr2num}``.  Converters can also be
            used to provide a default value for missing data (but see also
            `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
            Default: None.
        skiprows : int, optional
            Skip the first `skiprows` lines, including comments; default: 0.
        usecols : int or sequence, optional
            Which columns to read, with 0 being the first. For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
            The default, None, results in all columns being read.
        
            .. versionchanged:: 1.11.0
                When a single column has to be read it is possible to use
                an integer instead of a tuple. E.g ``usecols = 3`` reads the
                fourth column the same way as ``usecols = (3,)`` would.
        unpack : bool, optional
            If True, the returned array is transposed, so that arguments may be
            unpacked using ``x, y, z = loadtxt(...)``.  When used with a structured
            data-type, arrays are returned for each field.  Default is False.
        ndmin : int, optional
            The returned array will have at least `ndmin` dimensions.
            Otherwise mono-dimensional axes will be squeezed.
            Legal values: 0 (default), 1 or 2.
        
            .. versionadded:: 1.6.0
        encoding : str, optional
            Encoding used to decode the inputfile. Does not apply to input streams.
            The special value 'bytes' enables backward compatibility workarounds
            that ensures you receive byte arrays as results if possible and passes
            'latin1' encoded strings to converters. Override this value to receive
            unicode arrays and pass strings as input to converters.  If set to None
            the system default is used. The default value is 'bytes'.
        
            .. versionadded:: 1.14.0
        max_rows : int, optional
            Read `max_rows` lines of content after `skiprows` lines. The default
            is to read all the lines.
        
            .. versionadded:: 1.16.0
        
        Returns
        -------
        out : ndarray
            Data read from the text file.
        
        See Also
        --------
        load, fromstring, fromregex
        genfromtxt : Load data with missing values handled as specified.
        scipy.io.loadmat : reads MATLAB data files
        
        Notes
        -----
        This function aims to be a fast reader for simply formatted files.  The
        `genfromtxt` function provides more sophisticated handling of, e.g.,
        lines with missing values.
        
        .. versionadded:: 1.10.0
        
        The strings produced by the Python float.hex method can be used as
        input for floats.
        
        Examples
        --------
        >>> from io import StringIO   # StringIO behaves like a file object
        >>> c = StringIO(u"0 1\n2 3")
        >>> np.loadtxt(c)
        array([[0., 1.],
               [2., 3.]])
        
        >>> d = StringIO(u"M 21 72\nF 35 58")
        >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
        ...                      'formats': ('S1', 'i4', 'f4')})
        array([(b'M', 21, 72.), (b'F', 35, 58.)],
              dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])
        
        >>> c = StringIO(u"1,0,2\n3,0,4")
        >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
        >>> x
        array([1., 3.])
        >>> y
        array([2., 4.])
    



```python
numpy.loadtxt('inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
def s(p): 
    a = 0 
    for v in p: 
        a += v
    m = a / len(p)
    d = 0
    for v in p: 
        d += (v - m) * (v - m)
    return numpy.sqrt(d / (len(p) - 1))

def std_dev(sample): 
    sample_sum = 0
    for vaule in smaple: 
        sample_sum += value 
        
        
    sample_mean = sample_sum / len(sample)
    
    sum_squared_devs = 0 
    for value in sample:
        sum_squared_devs ++ (value - smaple_mean) + (value - smaple_mean)
        
    return numpy.sqrt(sum_squared_devs / (len(sample) -1))
```




## Defensive Programming 

```python
numbers = [1.5, 2.3, 0.7, 0.001, 4.4]

total = 0

for num in numbers:
    assert num > 0, 'Data should only contain positive values'
    total += num 
print('total is:', total)
```

    total is: 8.901



```python
def normalize_rectangle(rect):
    """Normalizes a rectangle so that it is ar the origin and 1.0 units long on its logest axis. 
    input should be of the format (x0, y0, x1, y1). 
    (x0, y0) and (x1, y1) define the ;ower left and upper right corners of the rectangke respectively. """
    assert len(rect) == 4, 'Rectangles must catain 4 coordinates'
    x0, y0, x1, y1 = rect 
    assert x0 < x1, 'Invalid x coordinates'
    assert y0 < y1, 'Invalid y coordinates'
    
    dx = x1 - x0
    dy = y1 - y0 
    if dx > dy: 
        scaled = dy / dx 
        upper_x, upper_y = 1, scaled 
    else: 
        scaled = dx / dy 
        upper_x, upper_y = scaled, 1 
    assert 0 < upper_x <= 1, 'Calculated upper x coordinate invalid'
    assert 0 < upper_y <= 1, 'Calculated upper y coordinate invalid'
    
    return (0, 0, upper_x, upper_y)
```


```python
print(normalize_rectangle((4, 2, 1, 5)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-4-2ce04ad58996> in <module>
    ----> 1 print(normalize_rectangle((4, 2, 1, 5)))
    

    <ipython-input-2-315ddda71dfb> in normalize_rectangle(rect)
          5     assert len(rect) == 4, 'Rectangles must catain 4 coordinates'
          6     x0, y0, x1, y1 = rect
    ----> 7     assert x0 < x1, 'Invalid x coordinates'
          8     assert y0 < y1, 'Invalid y coordinates'
          9 


    AssertionError: Invalid x coordinates



```python
print(normalize_rectangle((0, 0, 1, 5)))
```

    (0, 0, 0.2, 1)



```python
print(normalize_rectangle((0,0,5,1)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-8-15c567e1c437> in <module>
    ----> 1 print(normalize_rectangle((0,0,5,1)))
    

    <ipython-input-7-c21a98cfa466> in normalize_rectangle(rect)
         17         upper_x, upper_y = scaled, 1
         18     assert 0 < upper_x <= 1, 'Calculated upper x coordinate invalid'
    ---> 19     assert 0 < upper_y <= 1, 'Calculated upper y coordinate invalid'
         20 
         21     return (0, 0, upper_x, upper_y)


    AssertionError: Calculated upper y coordinate invalid




## Transcribing DNA into RNA 

```python
# prompt the used to enter the input fasta file name 

input_file_name = input("Enter the name of the input fasta file:")
```

    Enter the name of the input fasta file: CFTR.txt



```python
# open the input of the input fasta file and read the DNA sequence 

with open(input_file_name, "r") as input_file: 
    dna_sequence = ""
    for line in input_file: 
        if line.startswith(">"):
            continue 
        dna_sequence += line.strip()
```


```python
# transcribe the DNA to RNA 

rna_sequence = ""

for nucleotide in dna_sequence: 
    if nucleotide == "T":
        rna_sequence += "U"
    else: 
        rna_sequence += nucleotide 
```


```python
# prompt the user to enter the output file name 

output_file_name = input("Enter the name of the output file: ")
```


```python
# save the RNA sequence to a text file 

with open(output_file_name, "w") as output_file: 
    output_file.write(rna_sequence)
    print("The RNA sequence had been saved to ",{output_file_name})
```

    The RNA sequence had been saved to  {'rna_sequence_of_CFTR'}



```python
print(rna_sequence)
```

    AUGCAGAGGUCGCCUCUGGAAAAGGCCAGCGUUGUCUCCAAACUUUUUUUCAGCUGGACCAGACCAAUUUUGAGGAAAGGAUACAGACAGCGCCUGGAAUUGUCAGACAUAUACCAAAUCCCUUCUGUUGAUUCUGCUGACAAUCUAUCUGAAAAAUUGGAAAGAGAAUGGGAUAGAGAGCUGGCUUCAAAGAAAAAUCCUAAACUCAUUAAUGCCCUUCGGCGAUGUUUUUUCUGGAGAUUUAUGUUCUAUGGAAUCUUUUUAUAUUUAGGGGAAGUCACCAAAGCAGUACAGCCUCUCUUACUGGGAAGAAUCAUAGCUUCCUAUGACCCGGAUAACAAGGAGGAACGCUCUAUCGCGAUUUAUCUAGGCAUAGGCUUAUGCCUUCUCUUUAUUGUGAGGACACUGCUCCUACACCCAGCCAUUUUUGGCCUUCAUCACAUUGGAAUGCAGAUGAGAAUAGCUAUGUUUAGUUUGAUUUAUAAGAAGACUUUAAAGCUGUCAAGCCGUGUUCUAGAUAAAAUAAGUAUUGGACAACUUGUUAGUCUCCUUUCCAACAACCUGAACAAAUUUGAUGAAGGACUUGCAUUGGCACAUUUCGUGUGGAUCGCUCCUUUGCAAGUGGCACUCCUCAUGGGGCUAAUCUGGGAGUUGUUACAGGCGUCUGCCUUCUGUGGACUUGGUUUCCUGAUAGUCCUUGCCCUUUUUCAGGCUGGGCUAGGGAGAAUGAUGAUGAAGUACAGAGAUCAGAGAGCUGGGAAGAUCAGUGAAAGACUUGUGAUUACCUCAGAAAUGAUUGAAAAUAUCCAAUCUGUUAAGGCAUACUGCUGGGAAGAAGCAAUGGAAAAAAUGAUUGAAAACUUAAGACAAACAGAACUGAAACUGACUCGGAAGGCAGCCUAUGUGAGAUACUUCAAUAGCUCAGCCUUCUUCUUCUCAGGGUUCUUUGUGGUGUUUUUAUCUGUGCUUCCCUAUGCACUAAUCAAAGGAAUCAUCCUCCGGAAAAUAUUCACCACCAUCUCAUUCUGCAUUGUUCUGCGCAUGGCGGUCACUCGGCAAUUUCCCUGGGCUGUACAAACAUGGUAUGACUCUCUUGGAGCAAUAAACAAAAUACAGGAUUUCUUACAAAAGCAAGAAUAUAAGACAUUGGAAUAUAACUUAACGACUACAGAAGUAGUGAUGGAGAAUGUAACAGCCUUCUGGGAGGAGGGAUUUGGGGAAUUAUUUGAGAAAGCAAAACAAAACAAUAACAAUAGAAAAACUUCUAAUGGUGAUGACAGCCUCUUCUUCAGUAAUUUCUCACUUCUUGGUACUCCUGUCCUGAAAGAUAUUAAUUUCAAGAUAGAAAGAGGACAGUUGUUGGCGGUUGCUGGAUCCACUGGAGCAGGCAAGACUUCACUUCUAAUGGUGAUUAUGGGAGAACUGGAGCCUUCAGAGGGUAAAAUUAAGCACAGUGGAAGAAUUUCAUUCUGUUCUCAGUUUUCCUGGAUUAUGCCUGGCACCAUUAAAGAAAAUAUCAUCUUUGGUGUUUCCUAUGAUGAAUAUAGAUACAGAAGCGUCAUCAAAGCAUGCCAACUAGAAGAGGACAUCUCCAAGUUUGCAGAGAAAGACAAUAUAGUUCUUGGAGAAGGUGGAAUCACACUGAGUGGAGGUCAACGAGCAAGAAUUUCUUUAGCAAGAGCAGUAUACAAAGAUGCUGAUUUGUAUUUAUUAGACUCUCCUUUUGGAUACCUAGAUGUUUUAACAGAAAAAGAAAUAUUUGAAAGCUGUGUCUGUAAACUGAUGGCUAACAAAACUAGGAUUUUGGUCACUUCUAAAAUGGAACAUUUAAAGAAAGCUGACAAAAUAUUAAUUUUGCAUGAAGGUAGCAGCUAUUUUUAUGGGACAUUUUCAGAACUCCAAAAUCUACAGCCAGACUUUAGCUCAAAACUCAUGGGAUGUGAUUCUUUCGACCAAUUUAGUGCAGAAAGAAGAAAUUCAAUCCUAACUGAGACCUUACACCGUUUCUCAUUAGAAGGAGAUGCUCCUGUCUCCUGGACAGAAACAAAAAAACAAUCUUUUAAACAGACUGGAGAGUUUGGGGAAAAAAGGAAGAAUUCUAUUCUCAAUCCAAUCAACUCUAUACGAAAAUUUUCCAUUGUGCAAAAGACUCCCUUACAAAUGAAUGGCAUCGAAGAGGAUUCUGAUGAGCCUUUAGAGAGAAGGCUGUCCUUAGUACCAGAUUCUGAGCAGGGAGAGGCGAUACUGCCUCGCAUCAGCGUGAUCAGCACUGGCCCCACGCUUCAGGCACGAAGGAGGCAGUCUGUCCUGAACCUGAUGACACACUCAGUUAACCAAGGUCAGAACAUUCACCGAAAGACAACAGCAUCCACACGAAAAGUGUCACUGGCCCCUCAGGCAAACUUGACUGAACUGGAUAUAUAUUCAAGAAGGUUAUCUCAAGAAACUGGCUUGGAAAUAAGUGAAGAAAUUAACGAAGAAGACUUAAAGGAGUGCUUUUUUGAUGAUAUGGAGAGCAUACCAGCAGUGACUACAUGGAACACAUACCUUCGAUAUAUUACUGUCCACAAGAGCUUAAUUUUUGUGCUAAUUUGGUGCUUAGUAAUUUUUCUGGCAGAGGUGGCUGCUUCUUUGGUUGUGCUGUGGCUCCUUGGAAACACUCCUCUUCAAGACAAAGGGAAUAGUACUCAUAGUAGAAAUAACAGCUAUGCAGUGAUUAUCACCAGCACCAGUUCGUAUUAUGUGUUUUACAUUUACGUGGGAGUAGCCGACACUUUGCUUGCUAUGGGAUUCUUCAGAGGUCUACCACUGGUGCAUACUCUAAUCACAGUGUCGAAAAUUUUACACCACAAAAUGUUACAUUCUGUUCUUCAAGCACCUAUGUCAACCCUCAACACGUUGAAAGCAGGUGGGAUUCUUAAUAGAUUCUCCAAAGAUAUAGCAAUUUUGGAUGACCUUCUGCCUCUUACCAUAUUUGACUUCAUCCAGUUGUUAUUAAUUGUGAUUGGAGCUAUAGCAGUUGUCGCAGUUUUACAACCCUACAUCUUUGUUGCAACAGUGCCAGUGAUAGUGGCUUUUAUUAUGUUGAGAGCAUAUUUCCUCCAAACCUCACAGCAACUCAAACAACUGGAAUCUGAAGGCAGGAGUCCAAUUUUCACUCAUCUUGUUACAAGCUUAAAAGGACUAUGGACACUUCGUGCCUUCGGACGGCAGCCUUACUUUGAAACUCUGUUCCACAAAGCUCUGAAUUUACAUACUGCCAACUGGUUCUUGUACCUGUCAACACUGCGCUGGUUCCAAAUGAGAAUAGAAAUGAUUUUUGUCAUCUUCUUCAUUGCUGUUACCUUCAUUUCCAUUUUAACAACAGGAGAAGGAGAAGGAAGAGUUGGUAUUAUCCUGACUUUAGCCAUGAAUAUCAUGAGUACAUUGCAGUGGGCUGUAAACUCCAGCAUAGAUGUGGAUAGCUUGAUGCGAUCUGUGAGCCGAGUCUUUAAGUUCAUUGACAUGCCAACAGAAGGUAAACCUACCAAGUCAACCAAACCAUACAAGAAUGGCCAACUCUCGAAAGUUAUGAUUAUUGAGAAUUCACACGUGAAGAAAGAUGACAUCUGGCCCUCAGGGGGCCAAAUGACUGUCAAAGAUCUCACAGCAAAAUACACAGAAGGUGGAAAUGCCAUAUUAGAGAACAUUUCCUUCUCAAUAAGUCCUGGCCAGAGGGUGGGCCUCUUGGGAAGAACUGGAUCAGGGAAGAGUACUUUGUUAUCAGCUUUUUUGAGACUACUGAACACUGAAGGAGAAAUCCAGAUCGAUGGUGUGUCUUGGGAUUCAAUAACUUUGCAACAGUGGAGGAAAGCCUUUGGAGUGAUACCACAGAAAGUAUUUAUUUUUUCUGGAACAUUUAGAAAAAACUUGGAUCCCUAUGAACAGUGGAGUGAUCAAGAAAUAUGGAAAGUUGCAGAUGAGGUUGGGCUCAGAUCUGUGAUAGAACAGUUUCCUGGGAAGCUUGACUUUGUCCUUGUGGAUGGGGGCUGUGUCCUAAGCCAUGGCCACAAGCAGUUGAUGUGCUUGGCUAGAUCUGUUCUCAGUAAGGCGAAGAUCUUGCUGCUUGAUGAACCCAGUGCUCAUUUGGAUCCAGUAACAUACCAAAUAAUUAGAAGAACUCUAAAACAAGCAUUUGCUGAUUGCACAGUAAUUCUCUGUGAACACAGGAUAGAAGCAAUGCUGGAAUGCCAACAAUUUUUGGUCAUAGAAGAGAACAAAGUGCGGCAGUACGAUUCCAUCCAGAAACUGCUGAACGAGAGGAGCCUCUUCCGGCAAGCCAUCAGCCCCUCCGACAGGGUGAAGCUCUUUCCCCACCGGAACUCAAGCAAGUGCAAGUCUAAGCCCCAGAUUGCUGCUCUGAAAGAGGAGACAGAAGAAGAGGUGCAAGAUACAAGGCUUUAG




## Translating RNA into Protein 

```python
# prompt the user to enter the name of a file that has an RNA sequence 

input_file_name = input("Enter the name of the input RNA file: ")
```

    Enter the name of the input RNA file:  rna_sequence_of_CFTR



```python
# open the input RNA file and read the RNA sequence 

with open(input_file_name, "r") as input_file: 
    rna_sequence = input_file.read().strip()
    
```


```python
# define the codon table 

codon_table = {

    "UUU":"F", "UUC":"F", "UUA":"L", "UUG":"L",
    "CUU":"L", "CUC":"L", "CUA":"L", "CUG":"L",
    "AUU":"I", "AUC":"I", "AUA":"I", "AUG":"M",
    "GUU":"V", "GUC":"V", "GUA":"V", "GUG":"V",
    "UCU":"S", "UCC":"S", "UCA":"S", "UCG":"S",
    "CCU":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "ACU":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "GCU":"A", "GCC":"A", "GCA":"A", "GCG":"A",  
    "UAU":"Y", "UAC":"Y", "UAA":"*", "UAG":"*",
    "CAU":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
    "AAU":"N", "AAC":"N", "AAA":"K", "AAG":"K",
    "GAU":"D", "GAC":"D", "GAA":"E", "GAG":"E",
    "UGU":"C", "UGC":"C", "UGA":"*", "UGG":"W",
    "CGU":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "AGU":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GGU":"G", "GGC":"G", "GGA":"G", "GGG":"G"
    }
```


```python
# translate RNA to protein 

protein_sequence = ""

for i in range(0, len(rna_sequence), 3):
    codon = rna_sequence[i:i +3]
    if len(codon) == 3: 
        amino_acid = codon_table[codon]
        if amino_acid == "*":
            break
        protein_sequence += amino_acid
```


```python
# prompt the user to enter the output file name 

output_file_name = input("Enter the name of the output file: ")
```

    Enter the name of the output file:  CFTR_protein.txt



```python
# save the protein sequence to a text file 

with open(output_file_name, "w") as output_file: 
    output_file.write(protein_sequence)
    print("The protein sequence has been saved to", {output_file_name})
```

    The protein sequence has been saved to {'CFTR_protein.txt'}



```python
print(protein_sequence)
```

    MQRSPLEKASVVSKLFFSWTRPILRKGYRQRLELSDIYQIPSVDSADNLSEKLEREWDRELASKKNPKLINALRRCFFWRFMFYGIFLYLGEVTKAVQPLLLGRIIASYDPDNKEERSIAIYLGIGLCLLFIVRTLLLHPAIFGLHHIGMQMRIAMFSLIYKKTLKLSSRVLDKISIGQLVSLLSNNLNKFDEGLALAHFVWIAPLQVALLMGLIWELLQASAFCGLGFLIVLALFQAGLGRMMMKYRDQRAGKISERLVITSEMIENIQSVKAYCWEEAMEKMIENLRQTELKLTRKAAYVRYFNSSAFFFSGFFVVFLSVLPYALIKGIILRKIFTTISFCIVLRMAVTRQFPWAVQTWYDSLGAINKIQDFLQKQEYKTLEYNLTTTEVVMENVTAFWEEGFGELFEKAKQNNNNRKTSNGDDSLFFSNFSLLGTPVLKDINFKIERGQLLAVAGSTGAGKTSLLMVIMGELEPSEGKIKHSGRISFCSQFSWIMPGTIKENIIFGVSYDEYRYRSVIKACQLEEDISKFAEKDNIVLGEGGITLSGGQRARISLARAVYKDADLYLLDSPFGYLDVLTEKEIFESCVCKLMANKTRILVTSKMEHLKKADKILILHEGSSYFYGTFSELQNLQPDFSSKLMGCDSFDQFSAERRNSILTETLHRFSLEGDAPVSWTETKKQSFKQTGEFGEKRKNSILNPINSIRKFSIVQKTPLQMNGIEEDSDEPLERRLSLVPDSEQGEAILPRISVISTGPTLQARRRQSVLNLMTHSVNQGQNIHRKTTASTRKVSLAPQANLTELDIYSRRLSQETGLEISEEINEEDLKECFFDDMESIPAVTTWNTYLRYITVHKSLIFVLIWCLVIFLAEVAASLVVLWLLGNTPLQDKGNSTHSRNNSYAVIITSTSSYYVFYIYVGVADTLLAMGFFRGLPLVHTLITVSKILHHKMLHSVLQAPMSTLNTLKAGGILNRFSKDIAILDDLLPLTIFDFIQLLLIVIGAIAVVAVLQPYIFVATVPVIVAFIMLRAYFLQTSQQLKQLESEGRSPIFTHLVTSLKGLWTLRAFGRQPYFETLFHKALNLHTANWFLYLSTLRWFQMRIEMIFVIFFIAVTFISILTTGEGEGRVGIILTLAMNIMSTLQWAVNSSIDVDSLMRSVSRVFKFIDMPTEGKPTKSTKPYKNGQLSKVMIIENSHVKKDDIWPSGGQMTVKDLTAKYTEGGNAILENISFSISPGQRVGLLGRTGSGKSTLLSAFLRLLNTEGEIQIDGVSWDSITLQQWRKAFGVIPQKVFIFSGTFRKNLDPYEQWSDQEIWKVADEVGLRSVIEQFPGKLDFVLVDGGCVLSHGHKQLMCLARSVLSKAKILLLDEPSAHLDPVTYQIIRRTLKQAFADCTVILCEHRIEAMLECQQFLVIEENKVRQYDSIQKLLNERSLFRQAISPSDRVKLFPHRNSSKCKSKPQIAALKEETEEEVQDTRL

