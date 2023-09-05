def generator1():
    for i in range(3):
        yield f"gen01={i}"


def generator2():
    for val in generator1():
        yield f"gen02-{val}"


for val in generator2():
    print(val)
