import matplotlib.pyplot as plt
import numpy as np
import matplotlib
font = {'family' : 'normal',
        'size'   : 30}
matplotlib.rc('font', **font)


def plot_bar_chart():
    labels = ['No Enc', 'Rand Enc', 'TTE']
    bar_values = [13.12, 11.28, 0.19]

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars


    fig, ax = plt.subplots(figsize=(7, 5))
    # rects1 = ax.bar(x - width/2, base_values, width, label='Men', bottom=0, color="g")
    # rects2 = ax.bar(x + width/2, base_values, width, label='Women', bottom=0, color="g")
    rects3 = ax.bar(x, bar_values, width, label=labels,
                    color=['#fac901', '#225095', '#dd0100'])  # 6d99d6 7addd1 7addd1 f1e785
    # rects4 = ax.bar(x + width / 2, women_means, width, label='STE',
    #                 color='#f1e785')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Accuracy (%)')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    # ax.set_ylim((70, 87.5))
    ax.set_ylim((0, 16))
    ax.legend(loc='upper right', fontsize=35)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            print('rect: ', rect)
            height = rect.get_height()
            if height == min(bar_values):
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color='#dd0100',
                            weight='normal',
                            fontsize=30)
            else:
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color='black',
                            fontsize=30)


    autolabel(rects3)
    fig.tight_layout()
    fig.set_size_inches(7, 10)

    plt.show()

    fig.savefig('../../test_fields/bar_chart_plotters/chronological_order_bars.png', bbox_inches='tight')


if __name__ == '__main__':
    plot_bar_chart()
