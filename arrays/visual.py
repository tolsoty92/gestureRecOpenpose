import numpy as np
import matplotlib.pyplot as plt


gesture = '1finger/'

for i in range(0, 340):


    array = np.load(gesture+str(i)+'.npy')
    print(array)
    x = []
    y = []
    for i in array:
        x.append(i[0])
        y.append(i[1])

    # print(x)
    # print(y)

    plt.plot([x[0], x[1], x[2], x[3], x[4]], [y[0], y[1], y[2], y[3], y[4]], 'r')
    plt.plot([x[0], x[5], x[6], x[7], x[8]], [y[0], y[5], y[6], y[7], y[8]], 'y')
    plt.plot([x[0], x[9], x[10], x[11], x[12]], [y[0], y[9], y[10], y[11], y[12]], 'g')
    plt.plot([x[0], x[13], x[14], x[15], x[16]], [y[0], y[13], y[14], y[15], y[16]], 'b')
    plt.plot([x[0], x[17], x[18], x[19], x[20]], [y[0], y[17], y[18], y[19], y[20]], 'brown')

    plt.show()