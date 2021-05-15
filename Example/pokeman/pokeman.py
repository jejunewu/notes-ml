import os, glob, random,csv

def load_csv(root, filename, name2label):
    # 从csv文件返回images,labels列表
    # root:数据集根目录，filename:csv文件名， name2label:类别名编码表
    if not os.path.exists(os.path.join(root, filename)):
        # 如果csv文件不存在，则创建
        images = []
        for name in name2label.keys():
            # 只考虑后缀为png,jpg,jpeg的图片：'pokemon\\mewtwo\\00001.png
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))
        # 打印数据集信息：1167, 'pokemon\\bulbasaur\\00000000.png'
        print(len(images), images)
        random.shuffle(images) # 随机打散顺序
        # 创建csv文件，并存储图片路径及其label信息
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                name = img.split(os.sep)[-2]
                label = name2label[name]
                # 'pokemon\\bulbasaur\\00000000.png', 0
                writer.writerow([img, label])
            print('written into csv file:', filename)
    # 此时已经有csv文件，直接读取
    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            # 'pokemon\\bulbasaur\\00000000.png', 0
            img, label = row
            label = int(label)
            images.append(img)
            labels.append(label)
    # 返回图片路径list和标签list
    return images, labels


def load_pokeman(root, mode='train'):
    name2label = {}
    # 遍历根目录下的子文件夹，并排序，保证映射关系固定
    for name in sorted(os.listdir(os.path.join(root))):
        # 跳过非文件夹
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())

    # 读取Label信息
    # [file1,file2,], [3,1]
    images, labels = load_csv(root, 'images.csv', name2label)

    prcent = 0.8
    if mode == 'train':  # 60%
        images = images[:int(prcent * len(images))]
        labels = labels[:int(prcent * len(labels))]
    elif mode == 'val':  # 20% = 60%->80%
        images = images[int(prcent * len(images)):int((prcent+0.2) * len(images))]
        labels = labels[int(prcent * len(labels)):int((prcent+0.2) * len(labels))]
    else:  # 20% = 80%->100%
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]

    return images, labels, name2label
