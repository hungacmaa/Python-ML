# import matplotlib.pyplot as plt
# plt.plot(
# [1,2,3,4,5,6,7,8,9,10],
# [2,4.5,1,2,3.5,2,1,2,3,2]
# )

# import matplotlib.pyplot as plt
# plt.plot(
# [1,2,3,4,5,6,7,8,9,10],
# [2,4.5,1,2,3.5,2,1,2,3,2]
# )
# plt.title("Results") # sets the title for the chart
# plt.xlabel("Semester") # sets the label to use for the x-axis
# plt.ylabel("Grade") # sets the label to use for the y-axis

# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")
# plt.plot(
# [1,2,3,4,5,6,7,8,9,10],
# [2,4.5,1,2,3.5,2,1,2,3,2]
# )
# plt.title("Results") # sets the title for the chart
# plt.xlabel("Semester") # sets the label to use for the x-axis
# plt.ylabel("Grade") # sets the label to use for the y-axis

# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")
# plt.plot(
# [1,2,3,4,5,6,7,8,9,10],
# [2,4.5,1,2,3.5,2,1,2,3,2]
# )
# plt.plot(
# [1,2,3,4,5,6,7,8,9,10],
# [3,4,2,5,2,4,2.5,4,3.5,3]
# )
# plt.title("Results") # sets the title for the chart
# plt.xlabel("Semester") # sets the label to use for the x-axis
# plt.ylabel("Grade") # sets the label to use for the y-axis

# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")
# plt.plot(
# [1,2,3,4,5,6,7,8,9,10],
# [2,4.5,1,2,3.5,2,1,2,3,2],
# label="Jim"
# )
# plt.plot(
# [1,2,3,4,5,6,7,8,9,10],
# [3,4,2,5,2,4,2.5,4,3.5,3],
# label="Tom"
# )
# plt.title("Results") # sets the title for the chart
# plt.xlabel("Semester") # sets the label to use for the x-axis
# plt.ylabel("Grade") # sets the label to use for the y-axis
# plt.legend()

# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")
# plt.bar(
# [1,2,3,4,5,6,7,8,9,10],
# [2,4.5,1,2,3.5,2,1,2,3,2],
# label = "Jim",
# color = "m", # m for magenta
# align = "center"
# )
# plt.title("Results")
# plt.xlabel("Semester")
# plt.ylabel("Grade")
# plt.legend()
# plt.grid(True, color="y")

# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")
# plt.bar(
# [1,2,3,4,5,6,7,8,9,10],
# [2,4.5,1,2,3.5,2,1,2,3,2],
# label = "Jim",
# color = "m", # for magenta
# align = "center",
# alpha = 0.5
# )

# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")
# plt.bar(
# [1,2,3,4,5,6,7,8,9,10],
# [2,4.5,1,2,3.5,2,1,2,3,2],
# label = "Jim",
# color = "m", # for magenta
# align = "center",
# alpha = 0.5
# )
# plt.bar(
# [1,2,3,4,5,6,7,8,9,10],
# [1.2,4.1,0.3,4,5.5,4.7,4.8,5.2,1,1.1],
# label = "Tim",
# color = "g", # for green
# align = "center",
# alpha = 0.5
# )
# plt.title("Results")
# plt.xlabel("Semester")
# plt.ylabel("Grade")
# plt.legend()
# plt.grid(True, color="y")

# import matplotlib.pyplot as plt
# rainfall = [17,9,16,3,21,7,8,4,6,21,4,1]
# months = ['Jan','Feb','Mar','Apr','May','Jun',
# 'Jul','Aug','Sep','Oct','Nov','Dec']
# plt.bar(months, rainfall, align='center', color='orange' )
# plt.show()

# import matplotlib.pyplot as plt
# rainfall = [17,9,16,3,21,7,8,4,6,21,4,1]
# months = ['Jan','Feb','Mar','Apr','May','Jun',
# 'Jul','Aug','Sep','Oct','Nov','Dec']
# plt.bar(range(len(rainfall)), rainfall, align='center', color='orange' )
# plt.xticks(range(len(rainfall)), months, rotation='vertical')
# plt.show()

# import matplotlib.pyplot as plt
# labels = ["Chrome", "Internet Explorer",
# "Firefox", "Edge","Safari",
# "Sogou Explorer","Opera","Others"]
# marketshare = [61.64, 11.98, 11.02, 4.23, 3.79, 1.63, 1.52, 4.19]
# explode = (0,0,0,0,0,0,0,0)
# plt.pie(marketshare,
# explode = explode, # fraction of the radius with which to
# # offset each wedge
# labels = labels,
# autopct="%.1f%%", # string or function used to label the
# # wedges with their numeric value
# shadow=True,
# startangle=45) # rotates the start of the pie chart by
# # angle degrees counterclockwise from the
# # x-axis
# plt.axis("equal") # turns off the axis lines and labels
# plt.title("Web Browser Marketshare - 2018")
# plt.show()

# import matplotlib.pyplot as plt
# labels = ["Chrome", "Internet Explorer",
# "Firefox", "Edge","Safari",
# "Sogou Explorer","Opera","Others"]
# marketshare = [61.64, 11.98, 11.02, 4.23, 3.79, 1.63, 1.52, 4.19]
# explode = (0,0,0.8,0,0.8,0,0,0)
# colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
# plt.pie(marketshare,
# explode = explode, # fraction of the radius with which to
# # offset each wedge
# labels = labels,
# colors = colors,
# autopct="%.1f%%", # string or function used to label the
# # wedges with their numeric value
# shadow=True,startangle=45) # rotates the start of the pie chart by
# # angle degrees counterclockwise from the
# # x-axis
# plt.axis("equal") # turns off the axis lines and labels
# plt.title("Web Browser Marketshare - 2018")
# plt.show()

# Chúng ta cũng có thể lưu lại biểu đồ vẽ được dưới dạng ảnh
# ví dụ:
# import matplotlib.pyplot as plt
# labels = ["Chrome", "Internet Explorer",
# "Firefox", "Edge","Safari",
# "Sogou Explorer","Opera","Others"]
# ...
# plt.axis("equal") # turns off the axis lines and labels
# plt.title("Web Browser Marketshare - 2018")
# plt.savefig("Webbrowsers.png", bbox_inches="tight")
# plt.show()

# import matplotlib.pyplot as plt
# plt.plot([1,2,3,4], # x-axis
# [1,8,27,64], # y-axis
# 'bo') # blue circle marker
# plt.axis([0, 4.5, 0, 70]) # xmin, xmax, ymin, ymax
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# a = np.arange(1,4.5,0.1) # 1.0, 1.1, 1.2, 1.3...4.4
# plt.plot(a, a**2, 'y^', # yellow triangle_up marker
#          a, a**3, 'bo', # blue circle
# a, a**4, 'r--',) # red dashed line
# plt.axis([0, 4.5, 0, 70]) # xmin, xmax, ymin, ymax
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
a = np.arange(1,5,0.1)
plt.subplot(121) # 1 row, 2 cols, chart 1
plt.plot([1,2,3,4,5],
[1,8,27,64,125],
'y^')
plt.subplot(122) # 1 row, 2 cols, chart 2
plt.plot(a, a**2, 'y^',
a, a**3, 'bo',
a, a**4, 'r--',)
plt.axis([0, 4.5, 0, 70]) # xmin, xmax, ymin, ymax
plt.show()
         