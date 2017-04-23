import numpy as np
import matplotlib.pyplot as plt

def draw(points, options):
    plt.figure(1)
    g = np.linspace(-20, 20, 100)
    # label = ["A", "B", "C", "D", "E"]
    # points = [[10.33220959, -14.26762009],
    #           [-14.97896004, 3.90642428],
    #           [9.86806202, -5.64979553],
    #           [22.52350044, -12.22364902],
    #           [-7.54985332, -5.67611361]]
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    for i in range(len(points)):
        distance = y[i] - x[i]
        print distance
    plt.plot(x, y, "o")
    for i in range(len(options)):
        plt.annotate(options[i], fontsize=16, xy=(points[i][0], points[i][1]), xytext=(points[i][0] + 1, points[i][1] + 4),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                     )
    plt.plot(g, g)
    plt.show()