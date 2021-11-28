import GPUtil
import glob
import os
from utils import prepare_result_dir
import configs
from time import sleep
import sys
import run_ZSSR_single_input
from skimage.measure import compare_psnr, compare_ssim
 

def main(conf_name, gpu):
    # Initialize configs and prepare result dir with date
    print(f'conf_name : {conf_name} | gpu : {gpu}')
    if conf_name is None:
        conf = configs.Config()
    else:
        if conf_name.strip() == 'X2_ONE_JUMP_IDEAL_CONF':
            conf = configs.X2_ONE_JUMP_IDEAL_CONF
        if conf_name.strip()  == 'X2_IDEAL_WITH_PLOT_CONF':
            conf = configs.X2_IDEAL_WITH_PLOT_CONF
        if conf_name.strip()  == 'X2_GRADUAL_IDEAL_CONF':
            conf = configs.X2_GRADUAL_IDEAL_CONF
        if conf_name.strip()  == 'X2_GIVEN_KERNEL_CONF':
            conf = configs.X2_GIVEN_KERNEL_CONF
        if conf_name.strip()  == 'X2_REAL_CONF':
            conf = configs.X2_REAL_CONF

    res_dir = prepare_result_dir(conf)
    local_dir = os.path.abspath(__file__)
    print('current path : ',conf.input_path)
    # We take all png files that are not ground truth
    # files = [file_path for file_path in glob.glob('%s/*.png' % conf.input_path)
            #  if not file_path[-7:-4] == '_gt']
    files = [file_path for file_path in glob.glob('%s/*_bicubic.png' % conf.input_path)]
            
    # print(files)
    # Loop over all the files
    for file_ind, input_file in enumerate(files):
        # Ground-truth file needs to be like the input file with _gt (if exists)
        ground_truth_file = input_file.replace('_bicubic','_HR') 
        if not os.path.isfile(ground_truth_file):
            print(f'NO {ground_truth_file}')
            ground_truth_file = '0'

        # Numeric kernel files need to be like the input file with serial number
        kernel_files = ['%s_%d.mat;' % (input_file[:-4], ind) for ind in range(len(conf.scale_factors))]
        kernel_files_str = ''.join(kernel_files)
        for kernel_file in kernel_files:
            if not os.path.isfile(kernel_file[:-1]):
                kernel_files_str = '0'
                print('no kernel loaded')
                break
            print(f'kernel_files : {kernel_files}')

        # This option uses all the gpu resources efficiently
        if gpu == 'all':

            # Stay stuck in this loop until there is some gpu available with at least half capacity
            gpus = []
            while not gpus:
                gpus = GPUtil.getAvailable(order='memory')

            # Take the gpu with the most free memory
            cur_gpu = gpus[-1]

            # Run ZSSR from command line, open xterm for each run
            print(f'RUN COMMAND : '+"xterm -hold -e " + {conf.python_path} +
                      " %s/run_ZSSR_single_input.py '%s' '%s' '%s' '%s' '%s' '%s' alias python &"\
                          % (local_dir, input_file, ground_truth_file, kernel_files_str, cur_gpu, conf_name, res_dir))
            os.system("xterm -hold -e " + conf.python_path +
                      " %s/run_ZSSR_single_input.py '%s' '%s' '%s' '%s' '%s' '%s' alias python &"
                      % (local_dir, input_file, ground_truth_file, kernel_files_str, cur_gpu, conf_name, res_dir))

            # Verbose
            print('Ran file #%d: %s on GPU %d\n' % (file_ind, input_file, cur_gpu))

            # Wait 5 seconds for the previous process to start using GPU. if we wouldn't wait then GPU memory will not
            # yet be taken and all process will start on the same GPU at once and later collapse.
            sleep(5)

        # The other option is just to run sequentially on a chosen GPU.
        else: #weakness:파일 한개마다 다 run시켜서 학습하고 잇음=>un-efficient
            print('input_file, ground_truth_file, kernel_files_str, gpu, conf, res_dir " ' + input_file, ground_truth_file, kernel_files_str, gpu, conf, res_dir)
            run_ZSSR_single_input.main(input_file, ground_truth_file, kernel_files_str, gpu, conf, res_dir)
            print("Done")

if __name__ == '__main__':
    conf_str = sys.argv[1] if len(sys.argv) > 1 else None
    gpu_str = sys.argv[2] if len(sys.argv) > 2 else None
    main(conf_str, gpu_str)
