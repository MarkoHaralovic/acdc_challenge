# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.

project_root = '/home/s3758869/acdc_challenge/acdc_segmenter'
data_root = '/home/s3758869/acdc_challenge/database/training'
test_data_root = '/home/s3758869/acdc_challenge/database/testing'
local_hostnames = ['localhost']  # used to check if on cluster or not,
# enter the name of your local machine

##################################################################################

log_root = os.path.join(project_root, 'acdc_logdir')
preproc_folder = os.path.join(project_root,'preproc_data')

def setup_GPU_environment():

    if socket.gethostname() not in local_hostnames:
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            logging.warning('We are on a cluster but CUDA_VISIBLE_DEVICES is not set. Setting it to the default GPU.')
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['SGE_GPU'] = os.environ['CUDA_VISIBLE_DEVICES']
        else:
            logging.info('Setting SGE_GPU environment variable to {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
            os.environ['SGE_GPU'] = os.environ['CUDA_VISIBLE_DEVICES']
