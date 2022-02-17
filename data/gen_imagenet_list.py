import glob
import os
import numpy as np
folders =  'n01558993 n01601694 n01669191 n01751748 n01755581 n01756291 n01770393 n01855672 n01871265 n02018207 n02037110 n02058221 n02087046 n02088632 n02093256 n02093754 n02094114 n02096177 n02097130 n02097298 n02099267 n02100877 n02104365 n02105855 n02106030 n02106166 n02107142 n02110341 n02114855 n02120079 n02120505 n02125311 n02128385 n02133161 n02277742 n02325366 n02364673 n02484975 n02489166 n02708093 n02747177 n02835271 n02906734 n02909870 n03085013 n03124170 n03127747 n03160309 n03255030 n03272010 n03291819 n03337140 n03450230 n03483316 n03498962 n03530642 n03623198 n03649909 n03710721 n03717622 n03733281 n03759954 n03775071 n03814639 n03837869 n03838899 n03854065 n03929855 n03930313 n03954731 n03956157 n03983396 n04004767 n04026417 n04065272 n04200800 n04209239 n04235860 n04311004 n04325704 n04336792 n04346328 n04380533 n04428191 n04443257 n04458633 n04483307 n04509417 n04515003 n04525305 n04554684 n04591157 n04592741 n04606251 n07583066 n07613480 n07693725 n07711569 n07753592 n11879895'

IMG_EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']

if __name__ == '__main__':
    np.random.seed(0)
    cls_num = 50
    ratio = 0.5
    fout_train_label = open('ImageNet100_label_%d_%.1f.txt'%(cls_num, ratio), 'w')
    fout_train_unlabel = open('ImageNet100_unlabel_%d_%.1f.txt'%(cls_num, ratio), 'w')

    folders = folders.split(' ')
    for i, folder_name in enumerate(folders):
        files = []
        for extension in IMG_EXTENSIONS:
            files.extend(glob.glob(os.path.join('xxx', folder_name, '*' + extension) ))
        for filename in files: 
            if i < cls_num and np.random.rand() < ratio:
                fout_train_label.write('%s %d\n'%(filename[38:], i))
            else:
                fout_train_unlabel.write('%s %d\n'%(filename[38:], i))      

    fout_train_label.close()
    fout_train_unlabel.close()