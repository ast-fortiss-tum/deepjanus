import pandas as pd
import matplotlib.pyplot as plt
from numpy import trapz

for idx in reversed(range(1, 5)):
    print("attention threshold =", idx)
    FILEPATH_RANDOM = 'runs/run_random_at' + str(idx) + '/stats.csv'
    FILEPATH_ATT = 'runs/run_heatmap_at' + str(idx) + '/stats.csv'

    df_random = pd.read_csv(FILEPATH_RANDOM)
    df_att = pd.read_csv(FILEPATH_ATT)

    area_attention = trapz(df_att['iteration'].values)
    print("area (attention)=", area_attention)
    area_random = trapz(df_random['iteration'].values)
    print("area (random)=", area_random)
    print("incr. % (att vs random)=", round((area_attention / area_random * 100) - 100))
    print()

    # plt.plot(df_random['iteration'].values, la
    # bel='random (at=' + str(idx) + ')')
    # plt.plot(df_att['iteration'].values, label='attention (at=' + str(idx) + ')')
    # plt.axis([0, 400, 0, 1 + max(max(df_att['iteration'].values.tolist()), max(df_random['iteration'].values.tolist()))])
    #
    # plt.legend()
    # # plt.savefig('at=' + str(idx))
    # plt.show()
    # plt.clf()
