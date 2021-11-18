import matplotlib.pyplot as plt
import numpy as np
import matplotlib
font = {'family' : 'normal',
        'size'   : 30}
matplotlib.rc('font', **font)


def version_3():
    # labels = ['Baseline', 'DCT Only', 'Channel-wise', 'Temporal-wise']
    # labels = ['Baseline', 'DCT Only', 'Channel-wise', 'Temporal-wise', 'TTE (Ours)']
    labels = ['Jnt', '+Local', '+Center', '+Part', '+Finger']
    # labels = ['BSL', 'L=1', 'L=2', 'L=3', 'L=4']
    # baseline = 81.9
    baseline = 79.3
    # baseline = 88.2

    base_values = [baseline, baseline, baseline, baseline]
    # men_means = [81.9, 73.7, 82.2, 72.8, 83.1]
    # men_means = [baseline, 82.3, 82.9, 82.2, 82.6]
    men_means = [baseline, 80.1, 81.3, 80.8, 80.8]
    # men_means = [baseline, 89.0, 89.4, 89.3, 89.1]

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars


    fig, ax = plt.subplots(figsize=(7, 5))
    # rects1 = ax.bar(x - width/2, base_values, width, label='Men', bottom=0, color="g")
    # rects2 = ax.bar(x + width/2, base_values, width, label='Women', bottom=0, color="g")
    rects3 = ax.bar(x, men_means, width, label='TTE',
                    color='#6d99d6')  # 6d99d6 7addd1 7addd1 f1e785
    # rects4 = ax.bar(x + width / 2, women_means, width, label='STE',
    #                 color='#f1e785')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Accuracy (%)')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    # ax.set_ylim((70, 87.5))
    ax.set_ylim((78, 84))
    # ax.legend(loc='upper left', fontsize=35)

    plt.axhline(y=baseline, linewidth=2, color='#8c93a6',
                linestyle='--')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            print('rect: ', rect)
            height = rect.get_height()
            if height == max(men_means):
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color='red',
                            weight='normal',
                            fontsize=20)
            else:
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color='black',
                            fontsize=20)


    autolabel(rects3)

    fig.tight_layout()

    plt.show()

    fig.savefig("k-hyper-square.png", dpi=200)



def version_1():
    # labels = ['Jnt', '+Local', '+Center', '+Part', '+Finger', '+All']
    labels = ['Jnt', '+Local', '+Center', '+Part', '+Finger']

    # Baseline
    baseline_static = 81.9
    baseline_velocity = 79.1

    # Decouple GCN
    # baseline_static = 80.7
    # baseline_velocity = 78.2

    # Shift GCN
    # baseline_static = 79.2
    # baseline_velocity = 77.3

    # Baseline
    men_means = [baseline_static, 82.2, 82.9, 82.1, 82.6]
    women_means = [baseline_velocity, 80.1, 81.3, 80.8, 80.8]

    # DecoupleGCN
    # men_means = [baseline_static, 81.5, 82.1, 81.5, 81.5, 82.3]
    # women_means = [baseline_velocity, 81.3, 81.9, 81.6, 81.7, 82.0]

    # ShiftGCN
    # men_means = [baseline_static, 79.6, 80.5, 79.7, 79.8, 80.8]
    # women_means = [baseline_velocity, 77.6, 79.8, 78.7, 78.8, 80.3]

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars


    fig, ax = plt.subplots(figsize=(12,8))
    # rects1 = ax.bar(x - width/2, base_values, width, label='Men', bottom=0, color="g")
    # rects2 = ax.bar(x + width/2, base_values, width, label='Women', bottom=0, color="g")
    rects3 = ax.bar(x - width / 2, men_means, width, label='Static',
                    color='#225095')
    rects4 = ax.bar(x + width / 2, women_means, width, label='Velocity',
                    color='#fac901')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Angular Encoding Type')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    # ax.set_ylim((76.5, 84.5))
    ax.set_ylim((76.5, 85.5))
    ax.legend(loc='upper left', fontsize=30, ncol=2)

    # plt.axhline(y=baseline_static, linewidth=2, color='#8c93a6',
    #             linestyle='--')

    # plt.axhline(y=baseline_velocity, linewidth=2, color='#8c93a6',
    #             linestyle='--')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            print('rect: ', rect)
            height = rect.get_height()
            if height == max(men_means) or height == max(women_means):
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color='#dd0100',
                            # color='red',
                            weight='normal',
                            fontsize=20)
            else:
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color='black',
                            fontsize=20)


    autolabel(rects3)
    autolabel(rects4)

    fig.tight_layout()

    plt.show()
    fig.savefig("simple_static_velocity.png", dpi=200)


def version_2():
    import numpy as np
    import matplotlib.pyplot as plt

    # some example data
    threshold = 43.0
    values = np.array([30., 87.3, 99.9, 3.33, 50.0])
    x = range(len(values))

    # split it up
    above_threshold = np.maximum(values - threshold, 0)
    below_threshold = np.minimum(values, threshold)

    # and plot it
    fig, ax = plt.subplots()
    ax.bar(x, below_threshold, 0.35, color="g")
    ax.bar(x, above_threshold, 0.35, color="r",
           bottom=below_threshold)

    # horizontal line indicating the threshold
    ax.plot([0., 4.5], [threshold, threshold], "k--")

    fig.savefig("look-ma_a-threshold-plot.png")


if __name__ == '__main__':
    version_1()
