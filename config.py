import imgaug  # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa
import imgaug as ia


####
class Config(object):
    def __init__(self):
        if _args is not None:
            self.__dict__.update(_args.__dict__)
        self.seed = 5
        self.init_lr = 1.0e-3
        self.lr_steps = 40  # decrease at every n-th epoch
        self.train_batch_size = 8
        self.infer_batch_size = 16 #24/32
        self.nr_epochs = 100
        self.nr_classes = 4

        # nr of processes for parallel processing input
        self.nr_procs_train = 8
        self.nr_procs_valid = 8

        self.nr_fold = 5
        self.fold_idx = 4
        self.cross_valid = False

        self.load_network = False
        self.save_net_path = ""

        self.data_size = [1024, 1024]
        self.input_size = [512, 512]

        #
        self.dataset = 'colon_manual'
        # v1.0.3.0 test classifying cancer only
        self.logging = True  # True for debug run only
        self.log_path = '/media/vtltrinh/Data1/COLON_PATCHES_1000/log_result/'
        self.chkpts_prefix = 'model'
        self.task_type = self.run_info.split('_')[0]
        self.loss_type = self.run_info.replace(self.task_type + "_", "")
        self.model_name = f'/SoftTarget_{self.task_type}_{self.loss_type}'
        print(self.model_name)
        self.log_dir = self.log_path + self.model_name

    def train_augmentors(self):
        shape_augs = [
            iaa.Resize((512, 512), interpolation='nearest'),
            # iaa.CropToFixedSize(width=800, height=800),
        ]
        #
        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        input_augs = [
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),  # gaussian blur with random sigma
                iaa.MedianBlur(k=(3, 5)),  # median with random kernel sizes
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            ]),
            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
            # move pixels locally around (with random strengths)
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
            iaa.Sequential([
                iaa.Add((-26, 26)),
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.LinearContrast((0.75, 1.25), per_channel=1.0),
            ], random_order=True),
            sometimes([
                iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode="reflect",
                    pad_cval=(0, 255)
                ),
            ]),
        ]
        return shape_augs, input_augs

    ####
    def infer_augmentors(self):
        shape_augs = [
            iaa.Resize((512, 512), interpolation='nearest'),
            # iaa.CropToFixedSize(width=800, height=800, position="center"),
        ]
        return shape_augs, None

############################################################################
