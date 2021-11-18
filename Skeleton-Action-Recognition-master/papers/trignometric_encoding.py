def plot_multiple_lines_chronological_loss(lines, save_name=None, labels=None, every_n=5):
    font = {'size': 30}
    import matplotlib
    matplotlib.rc('font', **font)

    if labels is not None:
        assert len(lines) == len(labels)
    markers = ['^', '.', '*']
    # colors = ['#dd0100', '#225095', '#fac901']
    colors = ['#225095']

    plt.xlim(0, len(lines[0]))  # Chronological loss value
    plt.ylim(-0.25, 1.25)  # Chronological loss value

    x_axis_list = list([x for x in range(0, len(lines[0]), every_n)])
    for line_idx, a_line in enumerate(lines):
        a_line_plot = list([a_line[i] for i in range(0, len(lines[0]), every_n)])
        plt.plot(x_axis_list, a_line_plot,
                 color=colors[line_idx % len(colors)],
                 marker=markers[line_idx % len(markers)],
                 markersize=30,
                 label=labels[line_idx] if labels is not None else None)

    # plt.xticks(x_axis_list, a_line_plot)
    plt.grid()
    plt.legend(loc=4)  # Chronological loss value

    fig = matplotlib.pyplot.gcf()

    fig.set_size_inches(7, 10)  # Chronological loss value

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()
        plt.show()
