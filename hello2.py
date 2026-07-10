import matplotlib.pyplot as plt
import numpy as np
"""
x = np.array([12, 34, 54, 65, 77])
y = np.array([55, 77, 34, 76, 87])
y2 = np.array([23, 12, 45, 76, 11])
y3 = np.array([44, 34, 12, 43, 33])

line_style = dict(marker = "o", 
            markersize = 10,
            markerfacecolor = "black",
            markeredgecolor = "black",
            linestyle = "dashed",
            linewidth = 4)
plt.title("Spouse ages")
plt.xlabel("Man age")
plt.ylabel("Woman age")

plt.xticks(x)

plt.grid(axis = "both",
         color = "lightgray",
         linestyle = "dashed")

plt.plot(x,y, **line_style)
plt.plot(x,y2, color = "red", **line_style)
plt.plot(x,y3, color = "navy", **line_style)
plt.show()
"""
categories = np.array(["10","20","30","40","50","60","70","80","90","100"])
values = np.array([2,3,1,5,6,8,4,6,1,4])

plt.bar(categories,values)
plt.show()