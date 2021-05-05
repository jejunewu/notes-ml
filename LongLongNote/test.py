import datetime

now =  datetime.datetime.now()

for i in range(100000):
    pass
end = datetime.datetime.now()

print(end - now)
