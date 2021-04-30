import argparse
import os
import pickle
import fnmatch
import glob
from associate import read_file_list
from associate import associate
from natsort import natsorted


def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='Test pickle creator')
    parser.add_argument('--data_path', type=str, default='./dataset/Test',
                        help='The path to the test data.')
    parser.add_argument('--output_path', type=str, default='./pickles',
                        help='The path to save the test.pkl file.')
    return parser.parse_args()


def create_split(ROOT_DIR):

    final_split = [[], [], [], []]
    num_frames = 0

    color_vid_list = glob.glob(os.path.join(ROOT_DIR, 'Color/*'))
    for c in range(len(color_vid_list)):
        head, vid=os.path.split(color_vid_list[c])
        #TODO change this once we re-run COLMAP on stairs2 and walking5
        if(vid=='stairs2' or vid=='walking5'):
            continue

        color_frame_list=natsorted(glob.glob(os.path.join(color_vid_list[c],'*')))
        for f in range(len(color_frame_list)):
            color_path = color_frame_list[f]
            head2, color_num = os.path.split(color_path)
            num=color_num[5:-4]
            #fix for bees2 - bees2 depth is missing a leading 0 that color has
            depth_path = color_path.replace('Color', 'Depth')
            if vid=='bees2':
                depth_path=depth_path.replace(num,str(int(num)))

            colmap_root=head.replace('Color', 'COLMAP_Depth')
            colmap = glob.glob(os.path.join(colmap_root, vid,'frame_'+"{:0>6d}".format(int(num))+'.png.geometric.bin'))
            colmap_path = colmap[0]

            normal_path = colmap_path.replace('COLMAP_Depth', 'Normals')

            #Append paths
            final_split[0].append(color_path)
            final_split[1].append(depth_path)
            final_split[2].append(colmap_path)
            final_split[3].append(normal_path)
            num_frames+=1


    print('Number of frames: %d' % num_frames)

    return final_split


def main():
    args = ParseCmdLineArguments()
    final_dict = {'train': create_split(args.data_path)}
    with open(os.path.join(args.output_path,'test.pkl'), 'wb') as f:
        pickle.dump(final_dict, f)

if __name__ == "__main__":
    main()
