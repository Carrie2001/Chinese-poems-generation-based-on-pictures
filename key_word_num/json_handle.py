# coding UTF-8
# 该程序用于把https://github.com/chinese-poetry/chinese-poetry上的古诗json数据转化成txt并从中筛选五言诗句
import json
import os


# 5言诗
length = 5
length += 1
length *= 2
tfilename = '唐诗'
files = os.listdir(tfilename)
all_list = []
# 把这些诗句写入list
for name in files:
    filename = tfilename + '/'
    filename += name
    with open(filename, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
        for i in range(len(data_json)):
            tmp_list = []
            title = []
            title.append(data_json[i]['title'])
            author = []
            author.append(data_json[i]['author'])
            tmp_list.append(title)
            tmp_list.append(author)
            tmp_list.append(data_json[i]['paragraphs'])
            # print(tmp_list[2][0])
            # print(len(tmp_list[2][0]))
            # print(len(tmp_list[2]))
            if len(tmp_list[2]) != 2 or len(tmp_list[2][0]) != length:
                continue
            all_list.append(tmp_list)
# 把list存入文档
with open('五言绝句唐诗', 'w', encoding='utf-8') as file:
    for i in range(len(all_list)):
        file.write(all_list[i][0][0])
        file.write('::')
        file.write(all_list[i][1][0])
        file.write('::')
        file.write(all_list[i][2][0])
        file.write(all_list[i][2][1])
        file.write('\n')
