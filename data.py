import os

import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import random
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# %%
def findfiles(path):
    return glob.glob(path)


def convert_to_tensor(index: list, lst_of_char: list):
    image_set = []
    for i in index:
        image = Image.open(lst_of_char[i])
        image_ = transforms.ToTensor()(image)
        image_set.append(image_)

    return torch.stack(image_set, dim=0)


# mission of this data loader is to load images from different classes
def chinese_character_loader(opts):
    # return a dataloader and the number of classes
    working_directory = os.getcwd() + "\\data\\augmentation\\trainset_ch\\"
    font_list = findfiles(working_directory + '*')
    # below is to get me the class names
    labels = []
    for i in range(len(font_list)):
        category = os.path.splitext(os.path.basename(font_list[i]))[0]
        labels.append(category)

    # convert labels to one-hot encoding
    label_df = pd.DataFrame(labels, columns=["font"])
    dum_df = pd.get_dummies(label_df, columns=["font"], prefix='', prefix_sep='')

    # to randomly choose batch_size of chars and its labels
    image_set = []
    image_set_labels = []

    # Suppose the numbers of images in each folder are equal!!!
    number_chars_per_font = len(os.listdir(working_directory + '%s' % labels[0] + '\\'))
    for font_index in range(len(font_list)):
        # print(font_index)
        picking_directory = working_directory + '%s' % labels[font_index] + '\\'
        charNames_list = os.listdir(picking_directory)
        image_indices = random.sample(range(number_chars_per_font), min(number_chars_per_font, 300))
        # image_indices = range(number_chars_per_font)
        for i in image_indices:
            # print(i, len(image_indices))
            r = Image.open(picking_directory + charNames_list[i]).convert('L')
            t = transforms.ToTensor()(r)
            image_set.append(t)

            label_num = dum_df.columns.get_loc(labels[font_index])
            image_set_labels.append(label_num)
    '''

    for i in range(opts['batch_size']*3):
        # to randomly pick a font and then randomly pick a char
        font = labels[random.randint(0, len(labels) -1)]
        picking_directory = working_directory + '%s' %font + '\\'
        # char_list = findfiles(picking_directory + '*.png')
        charPaths_list = os.listdir(picking_directory)


        picked_char = charPaths_list[random.randint(0, len(charPaths_list)-1)]
        # convert the image to list
        image = Image.open(picked_char).convert('L')
        image_ = transforms.ToTensor()(image)
        image_set.append(image_)
        # convert its label to list

        # font_tensor = torch.tensor(dum_df[font], dtype=torch.long)
        # image_set_labels.append(font_tensor)

        label_num = dum_df.columns.get_loc(font)
        image_set_labels.append(label_num)
    '''
    train_data = []
    for i in range(len(image_set)):
        train_data.append([image_set[i], image_set_labels[i]])
    # the first return is the training set, the second return is the validation set
    return DataLoader(train_data, batch_size=opts['batch_size']), len(labels)


def test_data_loader(opts):
    working_directory = os.getcwd() + "\\data\\augmentation\\testset_ch\\"
    font_list = findfiles(working_directory + '*')
    # below is to get me the class names
    labels = []
    for i in range(len(font_list)):
        category = os.path.splitext(os.path.basename(font_list[i]))[0]
        labels.append(category)

    label_df = pd.DataFrame(labels, columns=["font"])
    dum_df = pd.get_dummies(label_df, columns=["font"], prefix='', prefix_sep='')
    image_set = []
    image_set_labels = []
    for i in labels:
        current_directory = working_directory + '%s\\' % i
        # load all characters in that directory
        char_list = findfiles(current_directory + '*')
        for j in range(len(char_list)):
            image = Image.open(char_list[j]).convert('L')
            image_ = transforms.ToTensor()(image)
            image_set.append(image_)
            # now we convert the label to tensor
            # font_tensor = dum_df[font].to_numpy()
            # font_tensor = np.long()
            # font_tensor = torch.tensor(dum_df[i], dtype=torch.long)
            font_tensor = torch.from_numpy(dum_df[i].to_numpy())

            # image_set_labels.append(font_tensor)
            label_num = dum_df.columns.get_loc(i)
            image_set_labels.append(label_num)

    test_data = []
    for i in range(len(image_set)):
        test_data.append([image_set[i], image_set_labels[i]])

    return DataLoader(test_data, batch_size=opts['validation_batch_size'])


# %%
def chinese_advanced_loader(opts, test = False):
    working_directory = os.getcwd() + '\\data\\augmentation\\trainset_ch\\'
    font_list = findfiles(working_directory + '*')
    labels = []
    for i in range(len(font_list)):
        category = os.path.splitext(os.path.basename(font_list[i]))[0]
        labels.append(category)

    if not test:
        image_set = []
        target_set = []
        y_set = []
        # below is balanced

        for i in range(len(font_list)):
            current_path = font_list[i]
            char_list = os.listdir(working_directory + labels[i])
            image_indices = random.sample(range(len(char_list)), 2 * opts['data_size'])
            count = 0
            for j in image_indices:
                if count % 2 == 0:
                    image = Image.open(working_directory + labels[i] + "\\" + char_list[j]).convert('L')
                    image = transforms.ToTensor()(image)
                    image_set.append(image)

                    random_dataset = random.randint(0, len(font_list)-1)
                    # might be resource-consuming, optimize?
                    random_indice = random.randint(0, len(char_list) - 1)
                    image_ = Image.open(working_directory + labels[random_dataset] + '\\' + char_list[random_indice]).convert('L')
                    image_ = transforms.ToTensor()(image_)
                    target_set.append(image_)
                    if random_dataset == i:
                        y_set.append(1)
                    else:
                        y_set.append(0)
                else:
                    image = Image.open(working_directory + labels[i] + "\\" + char_list[j]).convert('L')
                    image = transforms.ToTensor()(image)
                    image_set.append(image)
                    random_source = (random.sample(image_indices, 1))
                    random_source = random_source[0]
                    image_ = Image.open(working_directory + labels[i] + "\\" + char_list[random_source]).convert('L')
                    image_ = transforms.ToTensor()(image_)
                    target_set.append(image_)
                    y_set.append(1)
                count += 1

        shuff = list(zip(image_set, target_set, y_set))

        random.shuffle(shuff)
        image_set, target_set, y_set = zip(*shuff)
        y_set = torch.tensor(y_set, dtype=torch.float)
        image_loader = DataLoader(image_set, batch_size=opts['image_size'])
        target_loader = DataLoader(target_set, batch_size = opts['image_size'])
        y_loader = DataLoader(y_set, batch_size=opts['image_size'])

        return image_loader, target_loader, y_loader

    else:
        # below is the test part
        # working_directory_test = os.getcwd() + '\\data\\augmentation\\testset_ch\\'
        working_directory_test = os.getcwd() + '\\data\\augmentation\\testset_ch\\'
        random_test_indice = random.randint(0, len(labels) - 1)
        char_list_image = os.listdir(working_directory_test + labels[random_test_indice])
        test_target_set = []
        test_image_set = []
        test_y_set = []
        for i in range(60):
            random_ = int(random.randint(0, len(labels)-1))
            i_ = labels[random_]
            # target
            char_list = os.listdir(working_directory_test + i_)
            random_target_indice = int(random.randint(0, len(char_list) - 1))
            image = Image.open(working_directory_test + i_ + '\\' + char_list[random_target_indice]).convert('L')
            image = transforms.ToTensor()(image)
            test_target_set.append(image)
            # image
            random_image_indice = int(random.randint(0, len(char_list_image) - 1))
            image = Image.open(working_directory_test + labels[random_test_indice] + '\\' + char_list_image[random_image_indice]).convert('L')
            image = transforms.ToTensor()(image)
            test_image_set.append(image)
            if i_ == labels[random_test_indice]:
                test_y_set.append(1)
            else:
                test_y_set.append(0)

            # below is balanced test
            # target
            random_target_indice = int(random.randint(0, len(char_list_image) - 1))
            image = Image.open(working_directory_test + labels[random_test_indice] + '\\' + char_list_image[random_target_indice]).convert('L')
            image = transforms.ToTensor()(image)
            test_target_set.append(image)
            # image
            random_image_indice = int(random.randint(0, len(char_list_image) - 1))
            image = Image.open(working_directory_test + labels[random_test_indice] + '\\' + char_list_image[random_image_indice]).convert('L')
            image = transforms.ToTensor()(image)
            test_image_set.append(image)
            test_y_set.append(1)

        shuff = list(zip(test_image_set, test_target_set, test_y_set))
        random.shuffle((shuff))
        test_image_set, test_target_set, test_y_set = zip(*shuff)
        test_y_set = torch.tensor(test_y_set, dtype=torch.float)
        test_imageloader = DataLoader(test_image_set, batch_size=400)
        test_targetloader = DataLoader(test_target_set, batch_size=400)
        test_yloader = DataLoader(test_y_set, batch_size=400)

        return test_imageloader, test_targetloader, test_yloader

# %%
def chinese_advanced_char_loader(test_flag, opts, font_A_index, font_B_index):
    # A refers to the ref_imgs, B refers to the target_imgs
    # return ref_imgs, target_imgs, y
    working_directory = os.getcwd() + '\\data\\augmentation\\trainset_ch\\'
    font_list = findfiles(working_directory + '*')
    labels = []
    for i in range(len(font_list)):
        category = os.path.splitext(os.path.basename(font_list[i]))[0]
        labels.append(category)
    # below we are doing A
    image_font = labels[font_A_index]
    current_directory = working_directory + '%s' % image_font + '\\'
    char_list = findfiles(current_directory + '*.png')
    image_indices = random.sample(range(len(char_list)), 300)
    image_set = []
    image_set_labels = []
    # convert to onehot_encoding
    label_df = pd.DataFrame(labels, columns=["font"])
    dum_df = pd.get_dummies(label_df, columns=["font"], prefix='', prefix_sep='')

    for i in image_indices:
        image = Image.open(char_list[i]).convert('L')
        image_ = transforms.ToTensor()(image)
        image_set.append(image_)
        # label
        label_num = dum_df.columns.get_loc(image_font)
        image_set_labels.append(label_num)
    train_image_data = []
    for i in range(300):
        train_image_data.append(image_set[i])
    imageloader = DataLoader(train_image_data, batch_size=opts['image_size'])

    # below we are doing B
    target_font = labels[font_B_index]
    current_directory = working_directory + '%s' % target_font + '\\'
    char_list = findfiles(current_directory + '*.png')
    target_indices = random.sample(range(len(char_list)), 300)
    target_set = []
    target_set_labels = []
    # convert to onehot_encoding
    label_df = pd.DataFrame(labels, columns=["font"])
    dum_df = pd.get_dummies(label_df, columns=["font"], prefix='', prefix_sep='')
    for i in target_indices:
        image = Image.open(char_list[i]).convert('L')
        image_ = transforms.ToTensor()(image)
        target_set.append(image_)
        # label
        label_num = dum_df.columns.get_loc(target_font)
        target_set_labels.append(label_num)
    target_image_data = []
    for i in range(300):
        target_image_data.append(target_set[i])
    targetloader = DataLoader(target_image_data, batch_size=opts['ref_size'])

    # below we are doing y
    if target_font == image_font:
        y = 300 * [1]
        y = torch.tensor(y, dtype=torch.float)
    else:
        y = 300 * [0]
        y = torch.tensor(y, dtype=torch.float)
    yloader = DataLoader(y, batch_size=opts['image_size'])
    if not test_flag:

        return imageloader, targetloader, yloader
    else:
        # caution! Here y is a dataloader, so it needs extra steps in the test procedure
        image_loader, target_loader, y = advanced_test_dataloader(opts, font_A_index)
        return image_loader, target_loader, y


# Trialed as the train dataloader
# %%
def advanced_test_dataloader(opts, font_A_index):
    # the idea is to make sure that the image set and target set have the same batch number
    working_directory = os.getcwd() + "\\data\\augmentation\\testset_ch\\"
    font_list = findfiles(working_directory + '*')
    labels = []
    for i in range(len(font_list)):
        category = os.path.splitext(os.path.basename(font_list[i]))[0]
        labels.append(category)

    image_set = []
    working_directory_A = os.getcwd() + '\\data\\augmentation\\trainset_ch\\'
    char_list = os.listdir(working_directory_A + labels[font_A_index])
    # char_list = findfiles(working_directory_A + labels[font_A_index] + '\\*.png')
    # for now we are keeping the size of target images to be 200

    #below is the image
    for i in range(2 * len(font_list)):
        image_indices = random.sample(range(len(char_list)), 2 * len(font_list) * opts['image_size'])
        for k in image_indices:
            image = Image.open(working_directory_A + labels[font_A_index] + "\\" + char_list[k]).convert('L')
            image = transforms.ToTensor()(image)
            image_set.append(image)
    image_set *= (200//opts['ref_size'])
    image_loader = DataLoader(image_set, batch_size=opts['image_size'])

    # below is the reference
    # for now we load 200 pictures for each target class
    target_set = []
    target_set_ = []
    # for every target, we append 200 pictures to the target_set
    for j in range(len(labels)):
        # this return a list of images
        char_list_tar = os.listdir(working_directory + labels[j])
        image_indices = random.sample(range(len(char_list_tar)), 200)
        for k in image_indices:
            image = Image.open(working_directory + labels[j] + '\\' + char_list_tar[k]).convert('L')
            image = transforms.ToTensor()(image)
            target_set.append(image)
    for i in range(200//opts['ref_size']):
        # for every target, we append 200 pictures to the target_set
        # regroup the data so that they can be fed into the dataloader in a correct sequence
        for n in range(len(labels)):
            # add B
            starting_pos = i*(len(target_set)/(200//opts['ref_size'])) + n * opts['ref_size']
            all_pos = [int(starting_pos) + size for size in range(opts['ref_size'])]
            for pos in all_pos:
                pos = int(pos)
                target_set_.append(target_set[pos])
            # add A
            starting_pos_a = i * (len(target_set) / (200 // opts['ref_size'])) + font_A_index * opts['ref_size']
            all_pos_a = [starting_pos_a + size for size in range(opts['ref_size'])]
            for pos in all_pos_a:
                pos = int(pos)
                target_set_.append(target_set[pos])
    target_loader = DataLoader(target_set_, batch_size = opts['ref_size'])
    #below is y
    y = []
    for i in range(2 * (200//opts['ref_size']) * len(font_list)):
        if i % len(font_list) == font_A_index:
            for j in range(2 * opts['image_size']):
                y.append(1)
        else:
            for k in range(2 * opts['image_size']):
                if k <= opts['image_size'] - 1:
                    y.append(0)
                else:
                    y.append(1)

    y = torch.tensor(y, dtype=torch.float)
    y_loader = DataLoader(y, batch_size = opts['image_size'])

    return image_loader, target_loader, y_loader

# %%


if __name__ == "__main__":
    opts = {'image_size': 30, 'ref_size': 50, 'validation_batch_size': 50, 'train': False, 'batch_size': 50, 'data_size': 400}
    a, b, c = chinese_advanced_loader(opts, True)

    for data1, data2, data3 in zip(a,b,c):
        c1 = data1
        c2 = data2
        c3 = data3
        print(c1.shape, c2.shape, c3.shape)
        break
