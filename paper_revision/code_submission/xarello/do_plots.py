import pathlib
import gzip
import matplotlib.pyplot as plt
import pickle

data_path = pathlib.Path.home() / 'data' / 'xarello' / 'experiments' / 'EX7'
out_path = pathlib.Path.home() / 'data' / 'xarello'
cache_path = out_path / 'datas.pic'

victims = ['GEMMA'] #['BiLSTM', 'BERT']
tasks = ['PR2', 'FC', 'RD', 'HN']

prefixes = {"Mean train tries: ": "train", "Mean eval tries: ": "eval"}

if cache_path.exists():
    with open(cache_path, 'rb') as handle:
        datas= pickle.load(handle)
else:
    datas = {}
    for victim in victims:
        for task in tasks:
            print(victim + '-' + task)
            with gzip.open(data_path / victim / task / 'log.txt.gz', 'r') as fin:
                for line in fin:
                    line = line.decode('utf-8')
                    for prefix in prefixes:
                        if line.startswith(prefix):
                            number = float(line[(len(prefix)):].strip())
                            key = victim + '_' + task + '_' + prefixes[prefix]
                            if key not in datas:
                                datas[key] = []
                            datas[key].append(number)
    with open(cache_path, 'wb') as handle:
        pickle.dump(datas, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.clf()
result = plt.subplots(len(victims), 4, sharey='row', sharex='col')
if len(victims)==2:
    fig, (sub1, sub2) = result
else:
    fig, sub1 = result
fig.set_size_inches(10, 2.5*len(victims))
fig.subplots_adjust(wspace=0)
fig.subplots_adjust(hspace=0)
for victim in victims:
    for task, ax in zip(tasks, sub1 if victim == victims[0] else sub2):
        y = datas[victim + '_' + task + '_train']
        x = list(range(1, len(y) + 1))
        ax.plot(x, y, label='train', linewidth=2, color='black')
        y = datas[victim + '_' + task + '_eval']
        x = list(range(0, len(y)))
        ax.plot(x, y, label='eval', linewidth=2, color='darkgray')
        if task == tasks[0]:
            ax.yaxis.set_label_text('Steps until success')
            ax.set_yticks([10, 15, 20, 25])
        ax.set_ylim([7.0, 26])
        location = 'upper right' if task in ['PR2', 'FC'] else 'lower right'
        if victim == victims[-1]:
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.xaxis.set_label_text('Epochs')
        ax.legend(title=victim + '/' + task[:2], loc=location)
        
        

plt.savefig(out_path / 'plot.pdf', bbox_inches='tight')

print('END')
