# Scripher 2024.6.4
import os
from PIL import Image

class matrix:
    def __init__(self, L: list[list[int]]):
        self.row = len(L)
        self.col = len(L[0])
        self.data = [[L[i][j] for j in range(0, self.col)] for i in range(0, self.row)]

def fread(pathinp: str, pathoutImage: str, pathoutLabel: str, cnt: int):
    img = Image.open(pathinp).convert('L')
    w = img.size[0]
    h = img.size[1]
    if (w < 256 or h < 256):
        return -1
    mw = (w - 256) >> 1
    mh = (h - 256) >> 1
    box = (mw, mh, mw + 256, mh + 256)
    imgLabel = img.crop(box)

    imgLabel.save(pathoutLabel + str(cnt) + ".jpg")
    
    imgImage = imgLabel.resize((64, 64)).resize((256, 256))
    imgImage.save(pathoutImage + str(cnt) + ".jpg")

    dataLabel = imgLabel.getdata()
    dataImage = imgImage.getdata()
    return [matrix([[dataLabel[(i << 8) + j] for j in range(0, 256)] for i in range(0, 256)]), 
            matrix([[dataImage[(i << 8) + j] for j in range(0, 256)] for i in range(0, 256)])]
    
def fwrite(path: str, num: int):
    # 1-4字节是magic_number：2051
    # 5-8字节是图像张数num
    # 9-12字节是图像的行数：28
    # 13-16字节是图像的列数：28

    f = open(path, "wb")
    # magic_number
    f.write(int.to_bytes(2051, 4, "big"))

    # 图像张数
    f.write(int.to_bytes(num, 4, "big"))

    # 行数列数
    f.write(int.to_bytes(256, 4, "big"))
    f.write(int.to_bytes(256, 4, "big"))
        
    return f

def fwriteSingle(img: matrix, f) -> None:
    for i in range(0, 256):
        for j in range(0, 256):
            f.write(int.to_bytes(img.data[i][j], 1, "big"))
    return

# training set
pathImage = ".\\train-images.bin"
pathLabel = ".\\train-labels.bin"
# folder_path = "D:\\dataset\\test2014"
folder_path = ".\\train"
pathImageOutput = ".\\trainImage\\"
pathLabelOutput = ".\\trainLabel\\"
numimg = 40000

# testing set
# pathImage = ".\\test-images.bin"
# pathLabel = ".\\test-labels.bin"
# folder_path = ".\\test"
# pathImageOutput = ".\\testImage\\"
# pathLabelOutput = ".\\testLabel\\"
# numimg = 320

pointer = 0
betshow = 50
show = True

cnt = 0

fImage = fwrite(pathImage,numimg)
fLabel = fwrite(pathLabel,numimg)
end = False
for foldername, subfolders, filenames in os.walk(folder_path):
    if end:
        break
    for fname in filenames:
        if show and pointer % betshow == betshow - 1:
            print(pointer + 1)
        pointer = pointer + 1

        temp = fread(folder_path + "\\" + fname, pathImageOutput, pathLabelOutput, cnt)
        if (temp != -1):
            cnt += 1
            if(cnt == numimg + 1):
                end = True
                break
            fwriteSingle(temp[0], fImage)
            fwriteSingle(temp[1], fLabel)
if(cnt != numimg + 1):
    print("Warning : Not enough image read. ", cnt, " images have read.")
else:
    print(pointer, "images used,", cnt - 1, "images have read.")
fImage.close()
fLabel.close()