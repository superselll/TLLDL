import os
import numpy as np
import pandas as pd
import xlwt

def getresult(result_dir):
    res = []
    for i in os.listdir(result_dir):
        a = pd.read_excel(os.path.join(result_dir,i),header=None)
        res.append(np.expand_dims(a,0))
        #break
    res = np.concatenate(res)
    mean_value = np.mean(res,0,keepdims=False)
    max_value = np.max(res,0,keepdims=False)
    min_value = np.min(res,0,keepdims=False)
    deta1 = max_value-mean_value
    deta2 = mean_value-min_value
    output = np.ones_like(mean_value)
    row = output.shape[0]
    col = output.shape[1]
    file = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet1 = file.add_sheet('sheet1')
    for i in range(row):
        for j in range(col):
            deta = max(deta1[i][j],deta2[i][j])
            message = f'{round(mean_value[i][j],4)}±{round(deta,4)}'
            sheet1.write(i, j,message)
    file.save(os.path.join(result_dir,'total1.xls'))

def main():
    result_dir = 'bfgsuse'
    getresult(result_dir)


if __name__ == '__main__':
    main()