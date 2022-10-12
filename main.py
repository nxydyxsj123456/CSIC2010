import urllib.parse

normal_file_raw = 'D:/DATASET/CSIC2010/normalTrafficTraining.txt'
anomalous_file_raw = 'D:/DATASET/CSIC2010/anomalousTrafficTest.txt'

normal_file_pre = 'normal.txt'
anomalous_file_pre = 'anomalous.txt'


def pre_file(file_in, file_out=None):
    with open(file_in, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    res = []
    for i in range(len(lines)):
        line = lines[i].strip()#去前后空格
        # 提取 GET类型的数据
        if line.startswith("GET"):
            res.append("GET " + line.split(" ")[1])
        # 提取 POST类型的数据
        elif line.startswith("POST") or line.startswith("PUT"):
            method = line.split(' ')[0]
            url = line.split(' ')[1]
            j = 1
            # 提取 POST包中的数据
            while True:
                # 定位消息正文的位置
                if lines[i + j].startswith("Content-Length"):
                    break
                j += 1
            j += 2
            data = lines[i + j].strip()
            url += '?'+data
            res.append(method + ' ' + url)

    with open(file_out, 'w', encoding='utf-8') as f_out:
        for line in res:
            line = urllib.parse.unquote(line, encoding='ascii', errors='ignore').replace('\n', '').lower()
            f_out.writelines(line + '\n')

    print("{}数据预提取完成 {}条记录".format(file_out, len(res)))


if __name__ == '__main__':
    pre_file(normal_file_raw, normal_file_pre)
    pre_file(anomalous_file_raw, anomalous_file_pre)