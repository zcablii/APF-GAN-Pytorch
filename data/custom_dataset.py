"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='scale_width_and_crop')
        load_size = 572 if is_train else 512
        parser.set_defaults(load_size=load_size) # 256 or 512 for diff. input size
        parser.set_defaults(crop_size=512) # 256 or 512 for diff. input size
        parser.set_defaults(aspect_ratio=4/3)
        parser.set_defaults(display_winsize=256)

        parser.add_argument('--remove_gray_imgs', action='store_true', help='ignore gray training imgs')
        parser.set_defaults(remove_gray_imgs=True) 
        parser.add_argument('--remove_img_txt_path', type=str, default='',
                            help='txt file of hard img names')
                            
        parser.add_argument('--brightness', type=tuple, default=(1,1), help='training image brightness augment. Tuple of float (min, max) in range(0,inf)')
        parser.add_argument('--contrast', type=tuple, default=(1,1), help='training image contrast augment. Tuple of float (min, max) in range(0,inf)')
        parser.add_argument('--saturation', type=tuple, default=(1,1), help='training image saturation augment. Tuple of float (min, max) in range(0,inf)')
        # parser.set_defaults(brightness=(0.8,1.25))
        # parser.set_defaults(contrast=(0.8,1.25))
        # parser.set_defaults(saturation=(0.8,1.25))

        parser.set_defaults(label_nc=29)
        parser.set_defaults(batchSize=20) # 32 or 10 for diff. input size
        parser.set_defaults(contain_dontcare_label=False)

        
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)

        parser.set_defaults(no_instance=True)
        parser.add_argument('--label_dir', type=str, default='../data/train/labels', required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, default='../data/train/imgs',
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            print('set num_upsampling_layers to more')
            parser.set_defaults(num_upsampling_layers='more')
        return parser


    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True, remove_gray_imgs = opt.remove_gray_imgs,is_image=False,remove_img_txt_path=opt.remove_img_txt_path)

        if not opt.isTrain:
            image_paths = label_paths
        else:
            image_dir = opt.image_dir
            image_paths = make_dataset(image_dir, recursive=False, read_cache=True, remove_gray_imgs = opt.remove_gray_imgs,remove_img_txt_path=opt.remove_img_txt_path)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True, remove_gray_imgs = opt.remove_gray_imgs,is_image=False,remove_img_txt_path=opt.remove_img_txt_path)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths
