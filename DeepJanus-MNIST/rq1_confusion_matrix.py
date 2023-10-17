import pandas
import numpy as np

ADAPTIVE = False
df = pandas.read_csv('stats_2_converted_matrix.csv', sep=';')

if ADAPTIVE:
    attention = df['#Iterations_att_adaptive']
    random = df['#Iterations_Normal_adaptive']
else:
    attention = df['#Iterations_Att']
    random = df['#Iterations_Normal']

att_no_rnd_no = 0
att_yes_rnd_no = 0
att_no_rnd_yes = 0
att_yes_rnd_yes = 0

for idx in range(len(attention)):
    att = attention.iloc[idx]
    rnd = random.iloc[idx]

    if np.isnan(att) and np.isnan(rnd):
        att_no_rnd_no += 1
    elif np.isnan(att) and not np.isnan(rnd):
        att_no_rnd_yes += 1
    elif not np.isnan(att) and np.isnan(rnd):
        att_yes_rnd_no += 1
    elif not np.isnan(att) and not np.isnan(rnd):
        att_yes_rnd_yes += 1
    else:
        print("something is going on")
        exit()

print("ATT yes RND yes: %d (%.2f%%)" % (att_yes_rnd_yes, att_yes_rnd_yes / len(attention) * 100))

print("ATT no RND yes: %d (%.2f%%)" % (att_no_rnd_yes, att_no_rnd_yes / len(attention) * 100))

print("ATT yes RND no: %d (%.2f%%)" % (att_yes_rnd_no, att_yes_rnd_no / len(attention) * 100))

print("ATT no RND no: %d (%.2f%%)\n" % (att_no_rnd_no, att_no_rnd_no / len(attention) * 100))
