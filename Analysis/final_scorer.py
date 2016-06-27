# coding=utf-8
class question(object):
    def __init__(self):
        pass
    id = -1
    real_type = -1 # 0:EQ 1:SQ
    c_type = -1 # 0:EQ 1:SQ
    answer = -1
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    a_z = 0.0
    b_z = 0.0
    c_z = 0.0
    d_z = 0.0
    a_g = 0.0
    b_g = 0.0
    c_g = 0.0
    d_g = 0.0

def load_zscores(file='../data/zxr/allScores.txt'):

    f = open('../data/zxr/allScores.txt', mode='r')
    num = 0
    q_list = []

    for line in f:
        if num % 5 == 0:
            tem_q = question()
            strs = line.split('\t')[:-1]
            tem_q.id = int(strs[0])
            if strs[1] == "句子题":
                tem_q.real_type = 1
            elif strs[1] == "实体题":
                tem_q.real_type = 0
            if strs[2] == "句子题":
                tem_q.c_type = 1
            elif strs[2] == "实体题":
                tem_q.c_type = 0
            q_list.append(tem_q)
        elif num % 5 == 1:
            tem_q.a_z = float(line)
        elif num % 5 == 2:
            tem_q.b_z = float(line)
        elif num % 5 == 3:
            tem_q.c_z = float(line)
        elif num % 5 == 4:
            tem_q.d_z = float(line)
        num += 1

    f.close()
    return q_list

def load_gscores(file='../data/gru_test/gaokao_all_test/output.txt', q_list=[]):
    f = open('../data/gru_test/gaokao_all_test/output.txt', mode='r')
    num = 0
    for line in f:
        tem_q = q_list[num / 4]
        s = float(line.strip().split(' ')[0])
        flag = int(line.strip().split(' ')[1])
        if num % 4 == 0:
            tem_q.a_g = float(s)
        elif num % 4 == 1:
            tem_q.b_g = float(s)
        elif num % 4 == 2:
            tem_q.c_g = float(s)
        elif num % 4 == 3:
            tem_q.d_g = float(s)
        if flag == 1:
            tem_q.answer = num % 4
        num += 1
    f.close()
    return q_list

def regularize_scores(q_list):
    for q in q_list:
        z_sum = q.a_z + q.b_z + q.c_z + q.d_z
        g_sum = q.a_g + q.b_g + q.c_g + q.d_g
        q.a_z = q.a_z / z_sum
        q.b_z = q.b_z / z_sum
        q.c_z = q.c_z / z_sum
        q.d_z = q.d_z / z_sum
        q.a_g = q.a_g / g_sum
        q.b_g = q.b_g / g_sum
        q.c_g = q.c_g / g_sum
        q.d_g = q.d_g / g_sum
    return q_list

def combine_scores(q_list, W):
    for q in q_list:
        if q.c_type == 0:
            q.a = W[0][0] * q.a_z + W[0][1] * q.a_g
            q.b = W[0][0] * q.b_z + W[0][1] * q.b_g
            q.c = W[0][0] * q.c_z + W[0][1] * q.c_g
            q.d = W[0][0] * q.d_z + W[0][1] * q.d_g
        elif q.c_type == 1:
            q.a = W[1][0] * q.a_z + W[1][1] * q.a_g
            q.b = W[1][0] * q.b_z + W[1][1] * q.b_g
            q.c = W[1][0] * q.c_z + W[1][1] * q.c_g
            q.d = W[1][0] * q.d_z + W[1][1] * q.d_g
    return q_list

def count_right(q_list):
    right = 0
    wrong = 0
    eq_right = 0
    eq_wrong = 0
    sq_right = 0
    sq_wrong = 0
    for q in q_list:
        max_p = -1.0
        tmp_a = -1
        if q.a > max_p:
            max_p = q.a
            tmp_a = 0
        if q.b > max_p:
            max_p = q.b
            tmp_a = 1
        if q.c > max_p:
            max_p = q.c
            tmp_a = 2
        if q.d > max_p:
            max_p = q.d
            tmp_a = 3
        if tmp_a == q.answer:
            if q.real_type == 0:
                eq_right += 1
            else:
                sq_right += 1
            right += 1
        else:
            if q.real_type == 0:
                eq_wrong += 1
            else:
                sq_wrong += 1
            wrong += 1
    return right, wrong, eq_right, eq_wrong, sq_right, sq_wrong

if __name__ == '__main__':
    q_list = load_zscores()
    q_list = load_gscores(q_list=q_list)
    q_list = regularize_scores(q_list)
    # W = [[0.9695, 0.0305], [0.0053, 0.9947]]
    W = [[0, 1], [0, 1]]
    q_list = combine_scores(q_list, W)
    right, wrong, eq_right, eq_wrong, sq_right, sq_wrong= count_right(q_list)

    print right, wrong, eq_right, eq_wrong, sq_right, sq_wrong
