import numpy as np
print(np.__version__)

print("Hello", end=" ")
print("Abid")

print("Abid is",22,"years old")

x = "awesome"

def my_func():
    x = "not awesome"
    print("Python is " + x)

my_func()
print("Python is " + x)

arr = np.array(10)

arr2 = np.array([1,2,3,4,5])

lst = [1,3,4,5,6,87]

arr3 = np.array(lst)

print(arr3.ndim)

arr4 = np.array([[1,2,3],[4,5,6]])

print(arr4.ndim, arr4.shape)

arr5 = np.array([
    [ [1,2,3],[1,2,3] ] ,
    [ [10,20,30],[11,22,33] ] ,
    [ [10,20,30],[11,22,33] ] ,
    [ [10,20,30],[11,22,33] ]
])

print(arr5.ndim, arr5.shape)

a = np.array([1,2,3])
b = np.array([3,4,5])

print(a + b)
print(a-b)
print(a * b)
print(a ** b)

cg = np.array([3.88,3.92,2.5,4])

print(cg)

cg[cg < 3] = 0

print(cg)

zeroArray = np.zeros((10))

print(zeroArray)

zeroArray2 = np.zeros((2,4))

print(zeroArray2)

zeroArrray3 = np.zeros((2,4,6))

print(zeroArrray3)

anuNumArr = np.full((2,4),9)

print(anuNumArr)

i4 = np.eye(4)

print(i4)

divRange = np.linspace(1,10,4)
print(divRange)

divRange2 = np.linspace(1,10,4)
print(divRange2)

rng = np.random.default_rng()

print(rng.integers(1,7))

deserts = np.array(["🍰", "🍩", "🍫", "🍪","🍦","🍭","🧁","🍨","🎂","🧃"])
my_sweets = rng.choice(deserts, size = (3,3))
print(my_sweets)