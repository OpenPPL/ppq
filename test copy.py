import random

sum = 0
for i in range(1000):
    sum += max(random.random(), random.random())

print(sum / 1000)