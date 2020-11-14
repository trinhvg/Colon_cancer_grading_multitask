import os
import argparse
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
import torch.utils.data as data
from collections import OrderedDict
import importlib
from config import Config
from misc.infer_wsi_utils import *
from imgaug import augmenters as iaa
import pandas as pd
from PIL import Image
import openslide
from progress.bar import Bar as ProgressBar  # Easy progress reporting for Python


def compute_acc(pred_, ano_):
    pred, ano = pred_.copy(), ano_.copy()
    pred = pred[ano > 0]
    ano = ano[ano > 0]
    acc = np.mean(pred == ano)
    return np.round(acc, 4)


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())

def export_figure_matplotlib(arr, ano, f_name, dpi=200, resize_fact=0.5, plt_show=False):
    """
     Export array as figure in original resolution
     :param arr: array of image to save in original resolution
     :param f_name: name of file where to save figure
     :param resize_fact: resize facter wrt shape of arr, in (0, np.infty)
     :param dpi: dpi of your screen
     :param plt_show: show plot or not
     """
    # arr = cv2.resize(arr, (0.5, 0.5), interpolation=cv2.INTER_CUBIC)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1] / dpi, arr.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr)
    # ax.imshow(ano, alpha=0.3)
    plt.contour(ano, levels=[0.9, 1.9, 2.9, 3.9], colors=['dodgerblue', 'lightgreen', 'orange', 'darkred'],
                linewidths=50) #20
    plt.savefig(f_name, dpi=(dpi * resize_fact))

    im = fig2img(fig)
    im = np.array(im)
    im = cv2.resize(np.array(im),  (512,512), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f_name, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    print(f_name)
    if plt_show:
        plt.show()
    else:
        plt.close()

class Inferer(Config):
    def __init__(self):
        self.project_path = '/media/vtltrinh/Data2/JL_pred/Effi_pred/'
        self.in_img_path = '/media/vtltrinh/Data1/COLON_PATCHES_1000/CVPR/1010714_final'
        self.in_ano_path = self.in_img_path
        self.in_patch = self.in_img_path
        self.out_img_path = f'{self.project_path}/Effi_pred_colon_tma_v3/'
        self.patch_size = 1024
        self.patch_stride = self.patch_size//8
        self.infer_batch_size = 128
        self.nr_procs_valid = 12
        self.nr_classes = 4


    def resize_save(self, svs_code, save_name, img, scale=1.0):
        ano = img.copy()
        cmap = plt.get_cmap('jet')
        path = f'{self.out_img_path}/{svs_code}/'
        img = (cmap(img / scale)[..., :3] * 255).astype('uint8')
        img[ano == 0] = [10, 10, 10]
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'{path}/{save_name}.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return 0

    def infer_step_m(self, net, batch, net_name):
        net.eval()  # infer mode
        imgs = batch  # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW
        # push data to GPUs and convert to float32
        imgs = imgs.to('cuda').float()

        with torch.no_grad():  # dont compute gradient
            logit_class, _ = net(imgs)  # forward
            prob = nn.functional.softmax(logit_class, dim=1)
            return prob.cpu().numpy()

    def infer_step_c(self, net, batch, net_name):
        net.eval()  # infer mode

        imgs = batch  # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to('cuda').float()

        with torch.no_grad():  # dont compute gradient
            logit_class = net(imgs)  # forward
            prob = nn.functional.softmax(logit_class, dim=1)
            return prob.cpu().numpy()

    def infer_step_r(self, net, batch, net_name):
        net.eval()  # infer mode

        imgs = batch  # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to('cuda').float()

        with torch.no_grad():  # dont compute gradient
            logit_regres = net(imgs)  # forward
            label = torch.tensor([0., 1., 2., 3.]).repeat(len(logit_regres), 1).permute(1, 0).cuda()
            idx = torch.argmin(torch.abs(logit_regres - label), 0)
            return idx.cpu().numpy()

    def predict_one_model(self, net, svs_code, net_name="Multi_512_mse"):
        print(net_name)
        infer_step = Inferer.__getattribute__(self, f'infer_step_{net_name[0].lower()}')

        svs_code = svs_code.replace('_ano', '')
        slide = cv2.imread(f'{self.in_img_path}/{svs_code}.jpg')
        slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)

        ano = np.float32(np.load(f'{self.in_ano_path}/{svs_code}_ano.npy'))  # [h, w]
        patch_list = generate_patch_list(ano, self.patch_size, self.patch_stride)

        inf_output_dir = f'{self.out_img_path}/{svs_code}/'
        if not os.path.isdir(inf_output_dir):
            os.makedirs(inf_output_dir)

        infer_dataset = dataset.DatasetSerialPatch_tma_colon(slide, patch_list, self.patch_size, shape_augs=None)
        dataloader = data.DataLoader(infer_dataset,
                                     num_workers=self.nr_procs_valid,
                                     batch_size=self.infer_batch_size,
                                     shuffle=False,
                                     drop_last=False)

        out_prob = np.zeros([self.nr_classes, ano.shape[0], ano.shape[1]], dtype=np.float32)  # [h, w]
        out_prob_count = np.zeros([ano.shape[0], ano.shape[1]], dtype=np.float32) # [h, w]

        for batch_data in dataloader:
            imgs_input, imgs_path = batch_data
            output_prob = infer_step(net, imgs_input, net_name)
            # print(output_prob.shape)
            for idx, patch_loc in enumerate(imgs_path):
                patch_loc = np.array(eval(patch_loc))
                for grade in range(self.nr_classes):

                    out_prob[grade][patch_loc[0]:patch_loc[0] + self.patch_size, patch_loc[1]:patch_loc[1] + self.patch_size] += output_prob[idx][grade]
                    out_prob_count[patch_loc[0]:patch_loc[0] + self.patch_size, patch_loc[1]:patch_loc[1] + self.patch_size] += 1

        out_prob_count[out_prob_count == 0.] = 4.
        out_prob_count /= 4.
        out_prob /= out_prob_count
        predict = np.argmax(out_prob, axis=0) + 1

        for c in range(self.nr_classes):
            out_prob[c][ano == 0] = 0

        for grade in range(self.nr_classes):
            self.resize_save(svs_code, f'prob_grade{grade}_mse_ceo', out_prob[grade])
        predict[ano == 0] = 0

        acc = compute_acc(predict, ano)
        self.resize_save(svs_code, f'predict_{net_name}_{acc}', predict, scale=4.0)
        slide_ = cv2.resize(slide, (9216, 9216), interpolation=cv2.INTER_CUBIC)

        export_figure_matplotlib(slide_, ano, f'{self.out_img_path}/{svs_code}/slide_2.png')
        slide = cv2.resize(slide, (512, 512), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'{self.out_img_path}/{svs_code}/slide.png', cv2.cvtColor(slide, cv2.COLOR_BGR2RGB))
        np.save(f'{self.out_img_path}/{svs_code}/predict_{net_name}', predict)
        np.save(f'{self.out_img_path}/{svs_code}/ano_npy', ano)
        if not os.path.isfile(f'{self.out_img_path}/{svs_code}/ano.npy'):
            # np.save(f'{self.out_img_path}/{svs_code}/ano', ano)
            self.resize_save(svs_code, 'ano', ano, scale=4.0)
        print('done')
        return 0

    def predict_one_model_regress(self, net, svs_code, net_name="Multi_512_mse"):
        infer_step = Inferer.__getattribute__(self, f'infer_step_{net_name[0].lower()}')
        svs_code = svs_code.replace('_ano', '')
        slide = cv2.imread(f'{self.in_img_path}/{svs_code}.jpg')
        slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)

        ano = np.float32(np.load(f'{self.in_ano_path}/{svs_code}_ano.npy'))  # [h, w]
        patch_list = generate_patch_list(ano, self.patch_size, self.patch_stride)

        inf_output_dir = f'{self.out_img_path}/{svs_code}/'
        if not os.path.isdir(inf_output_dir):
            os.makedirs(inf_output_dir)

        infer_dataset = dataset.DatasetSerialPatch_tma_colon(slide, patch_list, self.patch_size, shape_augs=None)
        dataloader = data.DataLoader(infer_dataset,
                                     num_workers=self.nr_procs_valid,
                                     batch_size=self.infer_batch_size,
                                     shuffle=False,
                                     drop_last=False)


        out_prob = np.zeros([self.nr_classes, ano.shape[0], ano.shape[1]], dtype=np.float32)  # [h, w]

        for batch_data in dataloader:
            imgs_input, imgs_path = batch_data
            output_prob = infer_step(net, imgs_input, net_name)
            for idx, patch_loc in enumerate(imgs_path):
                patch_loc = np.array(eval(patch_loc))
                for grade in range(self.nr_classes):
                    if grade == output_prob[idx]:
                        out_prob[grade][patch_loc[0]:patch_loc[0] + self.patch_size, patch_loc[1]:patch_loc[1] + self.patch_size] += 1

        predict = np.argmax(out_prob, axis=0) + 1
        predict[ano == 0] = 0

        acc = compute_acc(predict, ano)
        self.resize_save(svs_code, f'predict_{net_name}_{acc}', predict, scale=4.0)

        slide_ = cv2.resize(slide, (9216, 9216), interpolation=cv2.INTER_CUBIC)

        export_figure_matplotlib(slide_, ano, f'{self.out_img_path}/{svs_code}/slide_2.png')
        slide = cv2.resize(slide, (512, 512), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'{self.out_img_path}/{svs_code}/slide.png', cv2.cvtColor(slide, cv2.COLOR_BGR2RGB))
        np.save(f'{self.out_img_path}/{svs_code}/predict_{net_name}', predict)
        if not os.path.isfile(f'{self.out_img_path}/{svs_code}/ano.npy'):
            self.resize_save(svs_code, 'ano', ano, scale=4.0)
        print('done')
        return 0

    def run_wsi(self, csv_acc):
        device = 'cuda'
        net_list = pd.read_csv(f'{os.getcwd()}/data/{csv_acc}')
        name_list = net_list['name_list'].values
        for net_name in name_list:
            if "CLASS" in self.task_type:
                net_def = importlib.import_module('model.class_dense')  # dynamic import
            if "REGRESS" in self.task_type:
                net_def = importlib.import_module('model.regres_dense')  # dynamic import
            if "MULTI" in self.task_type:
                net_def = importlib.import_module('model.multitask_net2')  # dynamic import
            net = net_def.densenet121()

            net = torch.nn.DataParallel(net).to(device)
            inf_model_path = os.path.join(net_dir, net_name, f'_net_{net_idx}.pth')
            saved_state = torch.load(inf_model_path)
            net.load_state_dict(saved_state)

            name_wsi_list = findExtension(self.in_ano_path, '.npy')
           for name in name_wsi_list:
                svs_code = name[:-4]
                print(svs_code)
                acc_wsi = []
                try:
                    if 'REGRESS' in net_name:
                        acc_one_model = self.predict_one_model_regress(net, svs_code, net_name=net_name)
                    else:
                        acc_one_model = self.predict_one_model(net, svs_code, net_name=net_name)
                    acc_wsi.append(acc_one_model)
                except:
                    print('error')

####
if __name__ == '__main__':
    #colon
    net_dir = "/media/vtltrinh/Data2/JL_pred/model/COLON_TMA_CMPB2020"
    csv_acc = 'Colon_DenseNet_Multitask.csv'
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()
    inferer = Inferer()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    inferer.run_wsi(csv_acc)
