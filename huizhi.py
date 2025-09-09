import json
import matplotlib.pyplot as plt

# 假设你的JSON文件名为data.json，并且它包含一个名为"ReferenceLine"的数组
json_file = 'data.json'

# 读取JSON文件
with open(json_file, 'r') as file:
    data = json.load(file)

# 提取ReferenceLine数据
reference_line = data['ReferenceLine']

# 假设ReferenceLine是一个包含(x, y)元组的列表
x_values = [point[0] for point in reference_line]
y_values = [point[1] for point in reference_line]

# 绘制图像
plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
plt.title('Reference Line Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()