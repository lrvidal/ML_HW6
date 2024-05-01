from sklearn.model_selection import train_test_split

input_file = "./rice.data"
mapping = {
    "Cammeo": 0,
    "Osmancik": 1
}

with open(input_file, 'r') as file:
    lines = file.readlines()

rice = []
for _, line in enumerate(lines):
    data = line.strip().split(',')
    typ = data[-1]
    data.pop(-1)
    if typ in mapping:
        data.append(mapping[typ])

    floats = [float(item) for item in data[:-1]]
    floats.append(int(data[-1]))
    rice.append(floats)

def getRiceData():
    rest, riceTest = train_test_split(rice, train_size=0.8, test_size=0.2) 
    riceTrain, riceVal = train_test_split(rest, train_size=0.75, test_size=0.25)
    return riceTrain, riceVal, riceTest