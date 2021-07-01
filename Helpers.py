import glob
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
# import random
import math
import os

err_not_np_img = "not a numpy array or list of numpy array"
err_img_arr_empty = "Image array is empty"
err_row_zero = "No. of rows can't be <=0"
err_column_zero = "No. of columns can't be <=0"
err_invalid_size = "Not a valid size tuple (x,y)"
err_caption_array_count = "Caption array length doesn't matches the image array length"


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def is_tuple(x):
    return type(x) is tuple


def is_list(x):
    return type(x) is list


def is_numeric(x):
    return type(x) is int


def is_numeric_list_or_tuple(x):
    for i in x:
        if not is_numeric(i):
            return False
    return True

    old_png_path = os.path.join(source_mask_dir, old_basename + '.png')
    new_png_path = os.path.join(dest_mask_dir, old_basename + '_' + mode_label + '.png')
    if os.path.isfile(old_png_path):
        cmd = 'cp {} {}'.format(old_png_path, new_png_path)
        os.system(cmd)


def save(image_array_dict, output_dir: str, mode_label):
    for filename, image_array in image_array_dict.items():

        file_name = '_'.join([os.path.splitext(filename)[0], mode_label]) + '.jpg'
        file_path = os.path.join(output_dir, file_name)
        plt.imsave(file_path, image_array)
        print('save {}: {}'.format(mode_label, file_name))


def visualize(image_array, column=1, v_gap=0.1, h_gap=0.1, fig_size=(20, 20), color_map=None, caption_array=-1, fname=None):
    if not (is_tuple(fig_size) and len(fig_size) == 2):
        raise Exception(err_invalid_size)
    if column <= 0:
        raise Exception(err_column_zero)
    if (is_list(image_array)):
        for img in image_array:
            if not is_numpy_array(img):
                raise Exception(err_not_np_img)

        if caption_array != -1:
            if len(caption_array) != len(image_array):
                raise Exception(err_caption_array_count)

        column = math.ceil(column)
        row = math.ceil(len(image_array) / column)
        column = min(column, len(image_array))
        f, axes = plt.subplots(row, column, figsize=fig_size)
        f.subplots_adjust(hspace=h_gap, wspace=v_gap)

        n_row = 0
        n_col = 0
        index = 0
        for img in image_array:
            if column == 1:
                axes[n_row].imshow(img, cmap=color_map)
                if (caption_array != -1):
                    axes[n_row].set_title(caption_array[index])
            elif row == 1:
                axes[n_col].imshow(img, cmap=color_map)
                if (caption_array != -1):
                    axes[n_col].set_title(caption_array[index])
            else:
                axes[n_row, n_col].imshow(img, cmap=color_map)
                if (caption_array != -1):
                    axes[n_row, n_col].set_title(caption_array[index])
            n_col += 1
            if (n_col) % column == 0:
                n_row += 1
                n_col = 0
            index += 1
            if (n_row == row):
                break
#         plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if isinstance(image_array, list):
        image_array_list = image_array
        # for image_array in  image_array_list:
        #     if is_numpy_array(image_array):
        #         print('show')
        #         plt.figure(figsize = fig_size)
        #         # plt.imshow(image_array, cmap=color_map)
        #         if(caption_array!=-1):
        #             plt.title(caption_array)
        #         plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(fname)
        # plt.show()


#         pad=0.4, w_pad=0.5, h_pad=1.0

def create_filesets(path, n):
    '''
    @description: 将所有jpg文件名，分为n份
    @param path {str} jpg所在路径
    @param n {n} n processes
    @return fileset_list {list}
    '''    
    assert(isinstance(n, int) and n>0)
    all_image_paths = glob.glob(path)
    cnt = int(len(all_image_paths)/n) + 1  # 单个数据集应有数据个数
    fileset_list = []
    for i in range(n):
        st = i*cnt
        end = min((i+1)*cnt, len(all_image_paths))
        print("{} -- {}".format(st, end))
        fileset_list.append(all_image_paths[st:end])
    return fileset_list
    

def load_images(fileset):
    '''Load images with RGB order'''
    for image_path in fileset:
        print(image_path)
        image_dict = {}
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        filename = os.path.basename(image_path)
        image_dict[filename] = img
        yield image_dict


class InputImages:
    def __init__(self, path) -> None:
        self.path = path

    def __iter__(self) -> None:
        self.files_path = glob.glob(self.path)
        self.file_cnt = len(self.files_path)
        self.idx = 0
        self.buffer_dict = {}
        self.buffer_start_idx = -1
        self.buffer_end_idx = -1
        return self

    def _do_buffer(self, idx: int, buffer_size=3000):
        '''
        load 1000 张图在内容中
        '''
        self.buffer_dict.clear()
        start_idx = idx
        if (start_idx + (buffer_size - 1)) > (self.file_cnt - 1):
            end_idx = self.file_cnt - 1
        else:
            end_idx = start_idx + (buffer_size - 1)
        print('Load buffer from: {} to {}'.format(start_idx, end_idx))
        for file_path in self.files_path[start_idx:(end_idx + 1)]:
            filename = os.path.basename(file_path)
            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            self.buffer_dict[filename] = img
        self.buffer_start_idx = start_idx
        self.buffer_end_idx = end_idx

    def _is_in_buffer(self, idx):
        print('idx = ', idx)
        if self.buffer_start_idx <= idx <= self.buffer_end_idx:
            return True
        else:
            return False

    def load_data(self):
        file_path = self.files_path[self.idx]
        filename = os.path.basename(file_path)
        img = self.buffer_dict[filename]
        self.idx += 1
        return {'filename': filename, 'img': img}

    def __next__(self):
        if self.idx >= self.file_cnt:
            raise StopIteration

        if self._is_in_buffer(self.idx):
            data_dict = self.load_data()
        else:
            self._do_buffer(self.idx)
            data_dict = self.load_data()
        return data_dict
