import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from env import *
import sys
sys.setrecursionlimit(10000)
#输入原始图像array，输出色块区域array，色块颜色array，色块数量
def FindColorPatch(rgb,thre=6):
    row, col = len(rgb), len(rgb[0])
    binary = [[0] * col for i in range(row)]
    for i in range(row-1):
        for j in range(col-1):
            if sum(rgb[i][j]) < 740 and sum(rgb[i][j]) >120:
                if abs(sum(rgb[i][j]) - sum(rgb[i][j+1])) < thre and abs(sum(rgb[i][j]) - sum(rgb[i+1][j])) < thre:
                      binary[i][j] = 1
    cells = []
    def dfs(x,y,binary,path):
        dirs = [[-1, 0], [0, 1], [0, -1], [1, 0]]
        path.append([x,y])
        binary[x][y] = 0
        for dir in dirs:
            x_, y_ = x + dir[0], y + dir[1]
            if x_ >= 0 and x_ < row and y_ >= 0 and y_ < col:
                if binary[x_][y_] == 1:
                    dfs(x_,y_,binary,path)
    binary_cp = binary.copy()
    for i in range(row):
        for j in range(col):
            if binary[i][j] == 1:
                path = []
                dfs(i,j,binary_cp,path)
                cells.append(path)
    count = 0
    new_binary = [[0] * col for i in range(row)]
    for item in cells:
        if len(item)>1500:
            for sub in item:
                new_binary[sub[0]][sub[1]] = 1
            count+=1
    #plt.imshow(new_binary)
    return new_binary,count,cells

#get center & mean color of each color patches
def GetCenterAndMeanColor(cells,rgb):
    result = []
    for item in cells:
        if len(item)>1500:
            r, g, b = 0, 0, 0
            for sub in item:
                rgb_value = rgb[sub[0]][sub[1]]
                r_, g_, b_ = rgb_value[0], rgb_value[1], rgb_value[2]
                r += r_
                g += g_
                b += b_
            r, g, b = r/len(item), g/len(item), b/len(item)
            result.append([r,g,b])
    center=[]
    for item in cells:
        if len(item)>1500:
            a,b=0,0
            for sub in item:
                a+=sub[0]
                b+=sub[1]
            a,b=a/len(item),b/len(item)
            center.append([a,b])
    return center,result

 #补全色卡,输入已识别出色块的center，平均rgb，
def MeanSquareColor(center,length,image):
    x,y=center
    bottom_x=int(x+length/2)
    top_x=int(x-length/2) if x-length/2>0 else 0
    left_y=int(y-length/2) if y-length/2>0 else 0
    right_y=int(y+length/2)
    pixels=image[top_x:bottom_x+1,left_y:right_y+1,:]
    return list(pixels.mean(axis=1).mean(axis=0))

def CompleteCard(center,result,image,label):
  center_new=center.copy()
  result_new=result.copy()
  label_new=list(label.copy())
  row_counts={}
  for i in label_new:
    if i not in row_counts:
      row_counts[i]=1
    else:
      row_counts[i]+=1
  max_row=max(row_counts.values())
  max_ind=max(row_counts,key=row_counts.get)
  #create a list to store not complete rows
  not_comp=[]
  for key,val in row_counts.items():
    if val<max_row:
      not_comp.append(key)
  max_row_y=[center_new[x][1] for x in np.where(label_new==max_ind)[0]]
  distance=(max(max_row_y)-min(max_row_y))/5
  #判断有缺失的row中具体哪一个column是缺失的，并把[row,column] append进center中
  while not_comp:
    cur_row=not_comp.pop()
    cur_row_y=[center_new[x][1] for x in np.where(label_new==cur_row)[0]]
    max_pointer=0
    cur_pointer=0
    sort_cur_row=sorted(cur_row_y)
    sort_max_row=sorted(max_row_y)
    row_ind=np.average([center_new[x][0] for x in np.where(label_new==cur_row)[0]])
    while max_pointer<len(max_row_y):
      if abs(sort_cur_row[cur_pointer]-sort_max_row[max_pointer])<0.4*distance:
        max_pointer+=1
        if cur_pointer<len(cur_row_y)-1:
          cur_pointer+=1
      else:
        center_new.append([row_ind,sort_max_row[max_pointer]])
        #print([row_ind,sort_max_row[max_pointer]])
        label_new.append(cur_row)
        result_new.append(MeanSquareColor([row_ind,sort_max_row[max_pointer]],distance*0.85,image))
        max_pointer+=1
      #print([cur_pointer,max_pointer])
  return center_new,result_new,label_new

#Find the location of black
##black patch always has minimum sum of rgb
def FindBlackLocation(center,result,label):
    black_row=label[np.argmin(np.array(result).sum(axis=1))]
    black_col=center[np.argmin(np.array(result).sum(axis=1))][1]
    black_row_all_col=[center[x][1] for x in np.where(label==black_row)[0]]
    if min(black_row_all_col)==black_col:
        return str(black_row+1)+'_1'
    elif max(black_row_all_col)==black_col:
        return str(black_row+1)+'_'+str(len(black_row_all_col))	

#after finding the location of black patch, we need to determine which color card we should use
def DetermineColorCard(center,result,label):
    location=FindBlackLocation(center,result,label)
    if location=='1_1':
        return color_black_1st
    elif location=='1_4':
        return color_green_1st
    elif location=='6_1':
        return color_white_1st
    elif location=='6_4':
        return color_brown_1st

#给图中的color result打标（已有label作为行）
def PatchLabel(center,label):
    import pandas as pd
    df=pd.DataFrame({'row':label,'col_ind':np.array(center)[:,1]})
    df['col']=df.groupby('row')['col_ind'].rank('dense',ascending=True).astype(int)
    df['row']=df.row.apply(lambda x:x+1)
    index=df.row.apply(str)+'_'+df.col.apply(str)
    return index

###pipeline
def color_matching(rgb):
    #输入原始图像array，输出色块区域array，色块颜色array，色块数量
    new_binary,count,cells=FindColorPatch(rgb, 4)
    #get center & mean color of each color patches
    center,result=GetCenterAndMeanColor(cells,rgb)
    #get row label for each patches
    from sklearn.cluster import DBSCAN
    min_eps=min(rgb.shape[0],rgb.shape[1])
    clustering = DBSCAN(eps=int(min_eps/6), min_samples=2).fit(np.array(center)[:,0].reshape(-1,1))
    labels=clustering.labels_
    #determine if the color card is complete or not, if not, complete the card
    print('count: '+str(count))
    if count<24:
        center,result,labels=CompleteCard(center,result,rgb,labels)
    
    #determine which one of four color cards we should use
    color_card=DetermineColorCard(center,result,labels)
    #label the location of each color patch in color card
    location=PatchLabel(center,labels)
    card_res=[]
    for i in location:
        card_res.append(color_card[i])
    return location,color_card,result,card_res
