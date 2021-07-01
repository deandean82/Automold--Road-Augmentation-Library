from multiprocessing import Process

import Automold as am
import Helpers as hp

IMAGE_PATH = './input/*.jpg'
XML_PATH = './input/xml'
LABEL_EXT = '.xml'
N_PROCESS = 5


def create_dataloaders(path, n):
    '''
    @description: create dataloaders 
    @param path {str} jpg path，example './input/*.jpg'
    @return loaders {list} 
    '''
    fileset_list = hp.create_filesets(IMAGE_PATH, n)
    loaders = []
    for fileset in fileset_list:
        loaders.append(hp.load_images(fileset))
    return loaders


def create_new_dataset(loaders: list):
    for loader in loaders:
        task = Process(group=None, target=_do_augment, name='create_augmentation', args=(loader, ), kwargs={}, daemon=False)
        task.start()


def _do_augment(image_dicts: iter):
    '''
    @description: do augment really
    @param image_dicts {generator} every element {'jpgfilename':img} , img RGB order
    @return {*}
    '''
    for i, image_dict in enumerate(image_dicts):
        print("Image count: ", i)
        # Bright
        bright_images_dict = am.brighten(image_dict)  ## if brightness_coeff is undefined brightness is random in each image
        hp.save(bright_images_dict, './output', mode_label='bright1')

        # Dark
        dark_images_dict = am.darken(image_dict, darkness_coeff=0.5)  ## darkness_coeff is between 0.0 and 1.0
        hp.save(dark_images_dict, './output', mode_label='dark2')

        # Dark_bright
        dark_bright_images_dict = am.random_brightness(image_dict)
        hp.save(dark_bright_images_dict, './output', mode_label='darkbright')

        # Rain
        rainy_images_dict = am.add_rain(image_dict, rain_type='heavy', slant=20)  # TODO 调参数
        hp.save(rainy_images_dict, './output', mode_label='rain')

        # Exposure
        exposure_images_dict = am.correct_exposure(image_dict)
        hp.save(exposure_images_dict, './output', mode_label='exposure')

        # Shadow
        shadowy_images_dict = am.add_shadow(image_dict, no_of_shadows=2, shadow_dimension=8)
        shadowy_images_dict = am.add_shadow(image_dict)
        hp.save(shadowy_images_dict, './output', mode_label='shadow')

        # Sonw
        # snowy_images_dict = am.add_snow(images_dict, snow_coeff=-1)
        # hp.save(snowy_images_dict, './output', mode_label='snow')
        # snowy_images= am.add_snow(images_dict, snow_coeff=0.3)
        # hp.visualize(snowy_images, column=3, fname='./output/snow1.jpg')
        # snowy_images= am.add_snow(images_dict, snow_coeff=0.8)
        # hp.visualize(snowy_images, column=3, fname='./output/snow2.jpg')

        # Fog
        foggy_images_dict = am.add_fog(image_dict, fog_coeff=0.05)
        hp.save(foggy_images_dict, './output', mode_label='fog')

        # # Flare
        # flare_images= am.add_sun_flare(images)
        # hp.visualize(flare_images, column=3, fname='./output/flare1.jpg')
        # import math
        # flare_images= am.add_sun_flare(images, flare_center=(100,100), angle=-math.pi/4) ## fixed src center
        # hp.visualize(flare_images, column=3, fname='./output/flare2.jpg')

        # # Speed
        # speedy_images_dict = am.add_speed(images_dict, speed_coeff=0.1)  ##random speed
        # hp.save(speedy_images_dict, './output', mode_label='speed')

        # speedy_images= am.add_speed(images) ##random speed
        # hp.visualize(speedy_images, column=3, fname='./output/speed1.jpg')
        # speedy_images= am.add_speed(images, speed_coeff=0.9) ##random speed
        # hp.visualize(speedy_images, column=3, fname='./output/speed2.jpg')

        # # Autumn
        # fall_images= am.add_autumn(images)
        # hp.visualize(fall_images, column=3, fname='./output/fall1.jpg')
    print('done')


def labels_duplicate(label_ext: str, ori_label_dir: str):
    '''
    @description: Find the corresponding xml, copy and rename it, with the output image files
    @param label_ext {str} Label file extension
    @param ori_label_dir {str} optinal, Dir of orignal label files
    @return {*}
    '''
    import os
    import glob
    files_path = glob.glob('./output/*.jpg')
    dest_mask_dir = './output/'
    for file_path in files_path:
        print(file_path)
        filename = os.path.basename(file_path)
        name = os.path.splitext(filename)[0]
        ori_name = '_'.join(name.split('_')[:-1])
        source_mask_path = os.path.join(ori_label_dir, ori_name + label_ext)
        dest_mask_path = os.path.join(dest_mask_dir, name + label_ext)
        if os.path.isfile(source_mask_path):
            cmd = 'cp {} {}'.format(source_mask_path, dest_mask_path)
            print(cmd)
            os.system(cmd)


if __name__ == "__main__":
    loaders = create_dataloaders(IMAGE_PATH, N_PROCESS)
    create_new_dataset(loaders)
    # labels_duplicate(LABEL_EXT, XML_PATH)
