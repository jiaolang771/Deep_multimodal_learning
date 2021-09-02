import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def KNN_aug(bin):
    #bin = np.concatenate((bin,data[0:5,:]),axis=0)
    # create a new empty data
    n,m = bin.shape
    new = np.zeros((1, bin.shape[1]))
    if(bin.shape[0]<=5):
        # this statement will generate a list of random numbers and add up to 1
        w = np.random.dirichlet(np.ones(bin.shape[0]), size=1)[0]
        for i in range(bin.shape[0]):
            new = new + w[i] * bin[i, :].reshape(1,m)
    else:
        # randomly select a subjects x0
        index = int(np.random.rand()*n)
        x0 = bin[index, 1:8101].reshape(1,m-1)
        y0 = bin[index,0]
        # use KNN to find 4 nearest neighbour to x0
        KNN = KNeighborsClassifier(n_neighbors=4)
        X = bin[:,1:m]
        y = bin[:,0].astype(int)
        #print(y)
        KNN.fit(X, y)
        # return a list of probabilities of the sub belongs to
        proba = KNN.predict_proba(x0)
        selected = np.append(y0,x0).reshape(1,m)
        while(selected.shape[0]<5):
            index_max = proba.argmax()
            unique_scores = np.unique(bin[:,0])
            score = unique_scores[index_max]
            index = np.where(bin[:,0]==score)[0]
            selected = np.concatenate((selected,bin[index]), axis=0)
            np.delete(proba,index_max)

        w = np.random.dirichlet(np.ones(5), size=1)[0]
        for i in range(5):
            #new = new + w[i] * selected[i].reshape(1, m)
            new = new + w[i] * bin[int(np.random.rand()*n)].reshape(1, m)

    bin = np.concatenate((new,bin), axis=0)
    #bin = np.concatenate((bin, new), axis=0)
    return bin

def augmentation(data):
    index = np.where(data[:,0] < 70)
    bin1 = data[index[0]]

    index = np.where(data[:,0] < 80)
    temp = data[index[0]]
    index = np.where(70 <= temp[:,0])
    bin2 = temp[index[0]]

    index = np.where(data[:,0] < 90)
    temp = data[index[0]]
    index = np.where(80 <= temp[:,0])
    bin3 = temp[index[0]]

    index = np.where(data[:,0] < 100)
    temp = data[index[0]]
    index = np.where(90 <= temp[:,0])
    bin4 = temp[index[0]]

    index = np.where(100 <= data[:,0])
    bin5 = data[index[0]]

    print('before augmentation: ')
    print('bin1: ', bin1.shape[0])
    print('bin2: ', bin2.shape[0])
    print('bin3: ', bin3.shape[0])
    print('bin4: ', bin4.shape[0])
    print('bin5: ', bin5.shape[0])

    total= bin1.shape[0]+bin2.shape[0]+bin3.shape[0]+bin4.shape[0]+bin5.shape[0]

    while(bin1.shape[0]+bin2.shape[0]+bin3.shape[0]+bin4.shape[0]+bin5.shape[0]<total*10):
        cd_num = bin1.shape[0]+bin2.shape[0]+bin3.shape[0]
        hc_num = bin4.shape[0]+bin5.shape[0]

        if(cd_num < hc_num):
            # augmentation cd
            if bin1.shape[0]<bin2.shape[0]:
                if bin1.shape[0]<bin3.shape[0]:
                    bin1 = KNN_aug(bin1)
                else:
                    bin3 = KNN_aug(bin3)
            else:
                if bin2.shape[0]<bin3.shape[0]:
                    bin2 = KNN_aug(bin2)
                else:
                    bin3 = KNN_aug(bin3)
        else:
            # sugmentation hc
            if bin4.shape[0] < bin5.shape[0]:
                bin4 = KNN_aug(bin4)
            else:
                bin5 = KNN_aug(bin5)

    print('\nafter augmentation: ')
    print('bin1: ', bin1.shape[0])
    print('bin2: ', bin2.shape[0])
    print('bin3: ', bin3.shape[0])
    print('bin4: ', bin4.shape[0])
    print('bin5: ', bin5.shape[0])

    aug_data = np.concatenate((bin1, bin2), axis=0)
    aug_data = np.concatenate((aug_data, bin3), axis=0)
    aug_data = np.concatenate((aug_data, bin4), axis=0)
    aug_data = np.concatenate((aug_data, bin5), axis=0)
    np.random.shuffle(aug_data)
    return aug_data
    # for i in range(10):
    #     plt.imshow(bin1[i,1:8101].reshape(90, 90))
    #     plt.savefig('images/bin1_'+str(i)+'_'+str(bin1[i,0])+'.png')
    #
    #     plt.imshow(bin2[i, 1:8101].reshape(90, 90))
    #     plt.savefig('images/bin2_' + str(i) + '_' + str(bin2[i, 0]) + '.png')
    #
    #     plt.imshow(bin3[i, 1:8101].reshape(90, 90))
    #     plt.savefig('images/bin3_' + str(i) + '_' + str(bin3[i, 0]) + '.png')
    #
    #     plt.imshow(bin4[i, 1:8101].reshape(90, 90))
    #     plt.savefig('images/bin4_' + str(i) + '_' + str(bin4[i, 0]) + '.png')
    #
    #     plt.imshow(bin5[i, 1:8101].reshape(90, 90))
    #     plt.savefig('images/bin5_' + str(i) + '_' + str(bin5[i, 0]) + '.png')

if __name__ == '__main__':
    print('')

    #data = np.loadtxt('./data_reg/columbus_col_8100_harmonized.txt')
    #aug_data = augmentation(data)