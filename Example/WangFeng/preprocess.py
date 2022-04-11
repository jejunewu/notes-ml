import os

PROJECT_ROOT_PATH = os.path.join(os.path.abspath(__file__), '..', '..', '..')
file_path = os.path.join(PROJECT_ROOT_PATH, 'Datasets', 'wangfeng.txt')

f = open(file_path, 'r', encoding='utf-8')
lines = f.readlines()
f.close()
new_lines = []

for i in range(len(lines)):
    if (lines[i] == '\n') | (lines[i] == '演唱：汪峰\n'):
        continue
    else:
        new_lines.append(lines[i][:-1] + '。\n')
f = open('new_wangfeng.txt', 'w', encoding='utf-8')
f.writelines(new_lines)
f.close()
