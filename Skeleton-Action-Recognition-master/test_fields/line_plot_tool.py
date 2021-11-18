from utils_dir.utils_visual import plot_multiple_lines

if __name__ == '__main__':
    line_1 = [1, 2, 3, 4, 5, 6, 7, 8]
    line_2 = [1, 6, 4, 5, 4, 5, 4, 8]

    value_sum = 0
    for i in range(len(line_2)-1):
        value_sum += line_2[i] - line_2[i+1]
    print('value sum: ', value_sum)

    plot_multiple_lines([line_2], every_n=1, save_name='oscillation.png')
    # plot_multiple_lines([line_2], every_n=1)
