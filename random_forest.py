from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import cv2
from random import shuffle
from sklearn.externals import joblib
from os import listdir as ld

def random_forest():
    workspace = os.getcwd()
    classnames = ['OCEAN', 'MOUNTAIN', 'LAKE', 'FARMLAND', 'DESERT', 'CITY']

    f = workspace+'/data_label.txt'
    with open(f,'r') as f:
        lines = f.readlines()
    data_num = len(lines)

    def transform(img):
        hist = cv2.calcHist([img], [0, 1, 2], None, [8]*3, [0, 256]*3)
        return hist.ravel()

    def oneHot(filelabel):
        label_matrix = np.eye(len(classnames))
        for i in range(len(classnames)):
            if classnames[i] == filelabel:
                return label_matrix[i]

    list_pic = []
    list_labels = []
    shuffle(lines)
    cnt = 0
    for i in range(data_num):
        tmp_filename = lines[i].split(',')
        tmp_filelabel = tmp_filename[1].strip('\n')
        tmp_filepath = tmp_filename[0]
        tmp_pic = cv2.imread(tmp_filepath)
        list_labels.append(oneHot(tmp_filelabel))
        tmp_repic = cv2.resize(tmp_pic, (256, 256))
        # tmp_repic = tmp_repic*1.0/255
        tmp_repic = transform(tmp_pic)
        list_pic.append(tmp_repic)
        cnt += 1
    print('=== processed pic', cnt, 'of', data_num, '===')
    array_pic = np.array(list_pic)
    # print(array_pic)
    array_labels = np.array(list_labels)
    # print(array_labels)

    rf=RandomForestRegressor(n_estimators = 10) # 这里使用了默认的参数设置
    rf.fit(array_pic,array_labels) # 进行模型的训练
    joblib.dump(rf, 'model.pkl')

    # listdir中所有图片变成np.array并append进result
    def accessfile(filename):
        tmp_batch = []
        tmp_pic = cv2.imread(filename)
        tmp_repic = cv2.resize(tmp_pic, (256, 256))
        # tmp_repic = tmp_repic*1.0/255
        tmp_repic = transform(tmp_pic)
        tmp_batch.append(tmp_repic)
        result = np.array(tmp_batch)
        return result

    tmp_filepath = os.path.join(workspace, 'datas/testset/')
    tmp_filenames = sorted(os.listdir(tmp_filepath))
    with open('result.csv', 'w') as res:
        for tmp_filename in tmp_filenames:
            result = accessfile(tmp_filepath+tmp_filename)
            pred = rf.predict(result)
            # print(tmp_filename, ':', pred)
            results = np.argmax(pred, axis=1)
            results_max = np.max(pred, axis=1)
            pred_label = classnames[results[0]]
            if results_max >= 0.9:
                tmp_img = cv2.imread(tmp_filepath+tmp_filename)
                cv2.imwrite(workspace+'/'+pred_label+'/'+tmp_filename, tmp_img)
                # print('path:', workspace+'/'+pred_label+'/'+tmp_filename)
            res.write(tmp_filename)
            res.write(',')
            res.write(pred_label)
            res.write('\n')
            # print(max_prob)

    def get_path(classname, workspace):
      list_names = ld(workspace+'/'+classname)
      # print(list(map(lambda x: workspace+classname+'/'+x, list_names)))
      return list(map(lambda x: workspace+'/'+classname+'/'+x+','+classname, list_names))

    list_names = []

    for classname in classnames:
      list_name = get_path(classname, workspace)
      print(classname, ':', len(list_name))
      for name in list_name:
          list_names.append(name)
    f = open('data_label.txt', 'w')
    for list_name in list_names:
      f.write(list_name+'\n')
    f.close()
    print('Total number:', len(list_names))
    return len(list_names)

length = []
for i in range(50):
    number = random_forest()
    length.append(number)
    err = length[i]-length[i-1]
    if err == 1:
        break
print(length[-1] - length[0])
