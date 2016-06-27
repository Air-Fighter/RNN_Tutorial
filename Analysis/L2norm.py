f = open('../data/qas_test/test01/output.txt', mode='r')

propabilities = []
right_one = []

for line in f:
    propabilities.append(float(line.split(" ")[0]))
    right_one.append(int(line.split(" ")[1]))

f.close()

size = len(propabilities) / 4

right_num = 0
wrong_num = 0

for index in xrange(size):
    right_index = -1
    min_index = -1
    min_p = 1

    for i in xrange(4):
        if propabilities[index * 4 + i] < min_p:
            min_index = i
            min_p = propabilities[index * 4 + i]
        if right_one[index * 4 + i] == 1:
            right_index = i

    if (right_index == min_index):
        right_num += 1
    else:
        wrong_num += 1

print right_num, wrong_num