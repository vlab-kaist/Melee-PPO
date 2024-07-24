import pandas as pd
import matplotlib.pyplot as plt


csv_files = [f"/home/tgkang/multi-env/skrl-ssbm/LeagueSelfPlay/agent{i}.csv" for i in range(5)]

dataframes = []

for file in csv_files:
    df = pd.read_csv(file, header=None, names=['epoch', 'elo'])
    dataframes.append(df)


plt.figure(figsize=(10, 6))


for i, df in enumerate(dataframes):
    plt.plot(df['epoch'], df['elo'], label=f'Agent {i+1}')


plt.title('ELO Over epoch')
plt.xlabel('Epoch')
plt.ylabel('ELO')
plt.legend()
plt.grid(True)

plt.savefig('elo_over_epoch.png')
