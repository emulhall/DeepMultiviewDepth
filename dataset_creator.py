import argparse
import os
import pickle
import fnmatch
import glob
from associate import read_file_list
from associate import associate
from natsort import natsorted


def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='TUM pickle creator')
    parser.add_argument('--data_path', type=str, default='~/TUM/',
                        help='The path to the TUM data.')
    parser.add_argument('--output_path', type=str, default='pickles',
                        help='The path to save the TUM.pkl file.')
    return parser.parse_args()


def create_split(ROOT_DIR):
    folder = os.path.join(ROOT_DIR)

    final_split = [[], [], [], []]
    num_frames = 0
    test=0
    #TODO Update this to however we get the color images
    #categories = fnmatch.filter(os.listdir(folder), '*/*')
    categories = glob.glob(os.path.join(folder, '*/*'))
    for c in range(len(categories)):
    	f1 = os.path.join(categories[c], 'rgb.txt')
    	f2 = os.path.join(categories[c], 'depth.txt')
    	first_list = read_file_list(f1)
    	second_list = read_file_list(f2)
    	matches = associate(first_list,second_list,0.0,0.04)
    	color_filelist = natsorted(glob.glob(os.path.join(categories[c], 'p*/images/*')))

    	#for img_idx in range(len(color_filelist)):
    	for img_idx in range(min(len(color_filelist),900)):
    		#Get color path
    		color_path = color_filelist[img_idx]

    		head,frame=os.path.split(color_path)

    		#Get the corresponding COLMAP prediction
    		colmap_filelist = glob.glob(os.path.join(head.replace('images', 'dense'),'*/stereo/depth_maps',frame+'.geometric.bin'))

    		for f in range(len(colmap_filelist)):
    			#Get the ground truth depth time index
    			gt = [match[1] for match in matches if str(match[0])==frame[:-4]]
    			if(len(gt)==0):
    				test+=1
    				continue
    			else:
    				gt_path = os.path.join(categories[c],'depth',str(gt[0])+'.png')
    				final_split[1].append(gt_path)
    			#Append color path
    			final_split[0].append(color_path)

    			#Get colmap prediction path
    			colmap_path=colmap_filelist[f]
    			final_split[2].append(colmap_path)

    			#Get normal prediction path
    			normal_path=colmap_path.replace('depth_maps','normal_maps')
    			final_split[3].append(normal_path)

    			num_frames+=1


    print('Number of frames: %d' % num_frames)
    print('Number missing gt: %d' % test)

    return final_split


def main():
    args = ParseCmdLineArguments()
    final_dict = {'train': create_split(args.data_path)}
    with open(os.path.join(args.output_path,'TUM.pkl'), 'wb') as f:
        pickle.dump(final_dict, f)

if __name__ == "__main__":
    main()
