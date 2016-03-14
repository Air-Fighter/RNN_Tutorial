f = open('data/test_3/output.txt', mode='r')

pred = []
right = []

for line in f:
    pred.append(int(line.split(" ")[0]))
    right.append(int(line.split(" ")[1]))

f.close()

size = len(pred) / 4

right_num = 0
cover_num = 0
wrong_num = 0
for index in xrange(size):
    flag = False
    pred_sum = 0
    for i in xrange(4):
        if (pred[index * 4 + i]==right[index * 4 + i] and \
            right[index * 4 + i] == 1 ):
            flag = True
        pred_sum += pred[index * 4 + i]

    print pred_sum

    if (flag and pred_sum == 1):
        right_num += 1
    elif (flag and pred_sum > 1):
        cover_num += 1
    else:
        wrong_num += 1

print right_num, cover_num, wrong_num